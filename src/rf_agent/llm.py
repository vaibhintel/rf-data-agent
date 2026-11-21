"""LLM access layer: a small provider-agnostic interface with ordered fallback.

The agent depends only on `LLMClient.complete()`, never on a vendor SDK. Any
OpenAI-compatible endpoint works (Ollama, vLLM, hosted APIs), and tests plug in a
scripted fake. `RoutedLLM` tries providers in order and falls back on errors.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

from .config import ProviderConfig

log = logging.getLogger(__name__)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str   # raw JSON string, exactly as the model produced it


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    provider: str = ""


class LLMClient(Protocol):
    def complete(self, messages: list[dict], tools: list[dict]) -> LLMResponse: ...


class OpenAICompatibleLLM:
    """Wraps any OpenAI-compatible chat-completions endpoint."""

    def __init__(self, cfg: ProviderConfig, temperature: float = 0.0):
        from openai import OpenAI  # imported lazily so tests don't need a server
        self.cfg = cfg
        self.temperature = temperature
        self.client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key, timeout=cfg.timeout_s)

    def complete(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        msg = self.client.chat.completions.create(
            model=self.cfg.model, messages=messages, tools=tools, temperature=self.temperature,
        ).choices[0].message
        calls = [ToolCall(tc.id, tc.function.name, tc.function.arguments or "{}")
                 for tc in (msg.tool_calls or [])]
        return LLMResponse(content=msg.content, tool_calls=calls, provider=self.cfg.name)


class RoutedLLM:
    """Try each provider in order; fall back to the next one on any error (timeout, 5xx, bad output)."""

    def __init__(self, providers: list[LLMClient]):
        if not providers:
            raise ValueError("At least one provider is required")
        self.providers = providers

    def complete(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        errors = []
        for p in self.providers:
            try:
                return p.complete(messages, tools)
            except Exception as e:  # noqa: BLE001 - any provider failure triggers fallback
                name = getattr(getattr(p, "cfg", None), "name", type(p).__name__)
                log.warning("provider %s failed: %s; falling back", name, e)
                errors.append(f"{name}: {e}")
        raise RuntimeError("All LLM providers failed: " + "; ".join(errors))
