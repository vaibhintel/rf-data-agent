import json

import pytest

from rf_agent.data import generate_dataset
from rf_agent.llm import LLMResponse, ToolCall
from rf_agent.mcp_server import build_server
from rf_agent.tools import RFTools


@pytest.fixture(scope="session")
def dataset():
    return generate_dataset(seed=7)


@pytest.fixture
def tools(dataset):
    return RFTools(dataset)


@pytest.fixture
def server(dataset):
    return build_server(dataset)


class ScriptedLLM:
    """Deterministic stand-in for a model: replays a fixed list of responses.

    A response can be a callable taking the message history, so a test can build the final
    answer from the actual tool output the agent received.
    """

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def complete(self, messages, tools):
        item = self.script[self.calls]
        self.calls += 1
        return item(messages) if callable(item) else item


def call(name, **args):
    return LLMResponse(content=None, provider="scripted",
                       tool_calls=[ToolCall(id=f"c_{name}", name=name, arguments=json.dumps(args))])


def answer(text):
    return LLMResponse(content=text, provider="scripted")


@pytest.fixture
def scripted():
    return ScriptedLLM
