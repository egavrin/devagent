import pytest

from ai_dev_agent.cli.router import IntentDecision, IntentRouter, IntentRoutingError
from ai_dev_agent.core.utils.config import Settings
from ai_dev_agent.core.utils.constants import LLM_DEFAULT_TEMPERATURE
from ai_dev_agent.providers.llm.base import ToolCall, ToolCallResult
from ai_dev_agent.tools import RUN


class DummyClient:
    def __init__(self, result: ToolCallResult) -> None:
        self.result = result
        self.captured_messages = None
        self.captured_tools = None

    def invoke_tools(
        self,
        messages,
        tools,
        temperature=LLM_DEFAULT_TEMPERATURE,
        max_tokens=None,
        tool_choice="auto",
        extra_headers=None,
        **kwargs,
    ):
        self.captured_messages = messages
        self.captured_tools = tools
        return self.result


def test_intent_router_returns_tool_decision(tmp_path):
    settings = Settings()
    settings.workspace_root = tmp_path
    result = ToolCallResult(
        calls=[ToolCall(name=RUN, arguments={"cmd": "ls"})],
        message_content="Enumerating workspace",
    )
    client = DummyClient(result)

    router = IntentRouter(client, settings)
    decision = router.route("покажи содержимое директории")

    assert isinstance(decision, IntentDecision)
    assert decision.tool == RUN
    assert decision.arguments["cmd"] == "ls"
    assert decision.rationale == "Enumerating workspace"
    assert client.captured_messages is not None
    assert client.captured_tools is not None


def test_intent_router_direct_response(tmp_path):
    settings = Settings()
    settings.workspace_root = tmp_path
    result = ToolCallResult(calls=[], message_content="Просто ответ")
    client = DummyClient(result)

    router = IntentRouter(client, settings)
    decision = router.route("скажи привет")

    assert decision.tool is None
    assert decision.arguments["text"] == "Просто ответ"


def test_intent_router_empty_prompt_raises(tmp_path):
    settings = Settings()
    settings.workspace_root = tmp_path
    client = DummyClient(ToolCallResult(calls=[], message_content=""))

    router = IntentRouter(client, settings)
    with pytest.raises(IntentRoutingError):
        router.route("   ")
