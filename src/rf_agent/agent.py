"""The agent loop: plan -> call MCP tools -> observe -> answer, with a full execution trace.

Every step is recorded in `AgentResult.trace`, so answers can be audited and evaluated
against the tool outputs they claim to be based on (see `rf_agent.grounding`).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from fastmcp import Client, FastMCP

from .config import AgentConfig
from .llm import LLMClient

SYSTEM_PROMPT = """You are an RF test-data analyst for wireless hardware engineers.
Rules:
- Answer ONLY from tool results. Never invent or estimate numbers.
- Start with list_options to learn valid builds and temperatures.
- For ACLR, higher (less negative) is worse.
- You have read-only access. If asked to modify or delete data, refuse and explain why.
- When you have enough evidence, give a short answer that quotes the numbers you used."""


@dataclass
class TraceStep:
    step: int
    tool: str
    arguments: str
    ok: bool
    result: str


@dataclass
class AgentResult:
    question: str
    answer: str
    trace: list[TraceStep] = field(default_factory=list)
    steps_used: int = 0
    stopped_reason: str = "answered"      # answered | step_limit
    providers_used: list[str] = field(default_factory=list)

    @property
    def tool_outputs(self) -> list[str]:
        return [s.result for s in self.trace if s.ok]


def _to_openai_tool(t) -> dict:
    schema = getattr(t, "input_schema", None) or t.inputSchema
    return {"type": "function",
            "function": {"name": t.name, "description": t.description or "", "parameters": schema}}


class RFAgent:
    def __init__(self, llm: LLMClient, server: FastMCP | str, config: AgentConfig | None = None):
        """`server` is an in-process FastMCP instance or anything fastmcp.Client accepts (URL, script path)."""
        self.llm = llm
        self.server = server
        self.cfg = config or AgentConfig()

    async def run(self, question: str) -> AgentResult:
        result = AgentResult(question=question, answer="")
        async with Client(self.server) as client:
            tools = [_to_openai_tool(t) for t in await client.list_tools()]
            messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT},
                                    {"role": "user", "content": question}]

            for step in range(1, self.cfg.max_steps + 1):
                reply = self.llm.complete(messages, tools)
                result.steps_used = step
                if reply.provider:
                    result.providers_used.append(reply.provider)

                if not reply.tool_calls:                       # no tool requested -> final answer
                    result.answer = reply.content or ""
                    return result

                messages.append({
                    "role": "assistant", "content": reply.content or "",
                    "tool_calls": [{"id": tc.id, "type": "function",
                                    "function": {"name": tc.name, "arguments": tc.arguments}}
                                   for tc in reply.tool_calls],
                })
                for tc in reply.tool_calls:
                    text, ok = await self._call_tool(client, tc.name, tc.arguments)
                    result.trace.append(TraceStep(step, tc.name, tc.arguments, ok, text))
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": text})

        result.answer = "Stopped: step limit reached without an answer."
        result.stopped_reason = "step_limit"
        return result

    async def _call_tool(self, client: Client, name: str, raw_args: str) -> tuple[str, bool]:
        """Execute one tool call; any failure is returned to the model as text so it can recover."""
        try:
            args = json.loads(raw_args or "{}")
        except json.JSONDecodeError:
            return "Arguments were not valid JSON. Try again.", False
        r = await client.call_tool(name, args, raise_on_error=False)
        text = r.content[0].text if r.content else ""
        return text[: self.cfg.max_tool_result_chars], not r.is_error
