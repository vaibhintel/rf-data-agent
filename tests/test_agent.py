import json

import pytest
from conftest import answer, call

from rf_agent.agent import RFAgent
from rf_agent.config import AgentConfig
from rf_agent.evaluation import GOLDEN_CASES, score
from rf_agent.llm import LLMResponse, RoutedLLM

CASES = {c.id: c for c in GOLDEN_CASES}


def _delta_at_60(messages):
    payload = json.loads(next(m["content"] for m in reversed(messages) if m["role"] == "tool"))
    return next(r["delta_db"] for r in payload["by_temperature"] if r["temperature_c"] == 60)


async def test_agent_answers_from_tool_output(server, scripted, tools):
    llm = scripted([
        call("list_options"),
        call("compare_builds", old_build="25.2.0", new_build="25.3.0"),
        lambda msgs: answer(f"Yes: build 25.3.0 is {_delta_at_60(msgs)} dB worse than 25.2.0 at 60 C."),
    ])
    res = await RFAgent(llm, server).run(CASES["quantify_regression"].question)

    assert [s.tool for s in res.trace] == ["list_options", "compare_builds"]
    s = score(CASES["quantify_regression"], res, tools)
    assert s.passed, s.notes


async def test_hallucinated_number_fails_grounding(server, scripted, tools):
    llm = scripted([
        call("compare_builds", old_build="25.2.0", new_build="25.3.0"),
        answer("Build 25.3.0 is 9.7 dB worse at 60 C."),          # 9.7 is not in any tool output
    ])
    res = await RFAgent(llm, server).run(CASES["quantify_regression"].question)
    s = score(CASES["quantify_regression"], res, tools)
    assert not s.grounded and not s.passed


async def test_invalid_build_error_is_returned_to_model(server, scripted):
    llm = scripted([
        call("summarize", build="25.3", temperature_c=60),
        call("summarize", build="25.3.0", temperature_c=60),
        answer("Build 25.3.0 at 60 C: see summary."),
    ])
    res = await RFAgent(llm, server).run("mean ACLR for 25.3 at 60 C?")
    first, second = res.trace
    assert not first.ok and "Valid builds" in first.result      # model sees how to self-correct
    assert second.ok


async def test_step_limit_stops_runaway_loops(server, scripted):
    llm = scripted([call("list_options")] * 10)
    res = await RFAgent(llm, server, AgentConfig(max_steps=3)).run("loop forever")
    assert res.stopped_reason == "step_limit" and res.steps_used == 3


async def test_refusal_case_scores_safe(server, scripted, tools):
    llm = scripted([answer("I can't do that: I have read-only access to the test data.")])
    res = await RFAgent(llm, server).run(CASES["refuse_delete"].question)
    assert score(CASES["refuse_delete"], res, tools).passed


def test_router_falls_back_to_next_provider():
    class Broken:
        def complete(self, messages, tools):
            raise TimeoutError("primary down")

    class Healthy:
        def complete(self, messages, tools):
            return LLMResponse(content="ok", provider="backup")

    assert RoutedLLM([Broken(), Healthy()]).complete([], []).provider == "backup"
    with pytest.raises(RuntimeError, match="All LLM providers failed"):
        RoutedLLM([Broken()]).complete([], [])
