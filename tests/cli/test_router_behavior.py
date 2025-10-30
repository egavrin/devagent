from types import SimpleNamespace

import pytest

from ai_dev_agent.cli.router import IntentRouter, IntentRoutingError
from ai_dev_agent.core.utils.config import Settings
from ai_dev_agent.providers.llm.base import ToolCall, ToolCallResult
from ai_dev_agent.session.manager import SessionManager


class DummyLLM:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def invoke_tools(self, messages, tools, temperature=0.1):
        self.calls.append(("invoke_tools", messages, tools))
        return self.response

    def generate_with_tools(self, messages, tools, temperature=0.1):
        self.calls.append(("generate_with_tools", messages, tools))
        return self.response


def _cleanup_router_session(router: IntentRouter):
    manager = SessionManager.get_instance()
    with manager._lock:  # type: ignore[attr-defined]
        manager._sessions.pop(router._session_id, None)  # type: ignore[attr-defined]


def test_intent_router_returns_decision(tmp_path):
    tool_call = ToolCall(name="read", arguments={"paths": ["foo.py"]}, call_id="1")
    response = ToolCallResult(
        calls=[tool_call],
        message_content="Reading foo.py",
        raw_tool_calls=[
            {
                "id": "1",
                "type": "function",
                "function": {"name": "read", "arguments": '{"paths": ["foo.py"]}'},
            }
        ],
    )
    router = IntentRouter(
        DummyLLM(response),
        settings=Settings(),
        project_profile={"primary_language": "python"},
    )
    router.settings.workspace_root = tmp_path

    decision = router.route("Open foo.py")

    try:
        assert decision.tool == "read"
        assert decision.arguments == {"paths": ["foo.py"]}
        assert decision.rationale
    finally:
        _cleanup_router_session(router)


def test_intent_router_tuple_response_and_argument_parsing(tmp_path):
    tuple_response = (
        "Execute tests",
        [
            {
                "function": {
                    "name": "run",
                    "arguments": '{"cmd": "pytest", "path": "tests/"}',
                },
                "id": "call-42",
            }
        ],
    )
    router = IntentRouter(DummyLLM(tuple_response), settings=Settings())
    router.settings.workspace_root = tmp_path

    decision = router.route_prompt("Run the unit tests")

    try:
        assert decision.tool == "run"
        assert decision.arguments == {"cmd": "pytest", "path": "tests/"}
    finally:
        _cleanup_router_session(router)


def test_intent_router_raises_when_client_missing_methods(tmp_path):
    class IncompleteClient:
        pass

    router = IntentRouter(IncompleteClient(), settings=Settings())
    router.settings.workspace_root = tmp_path

    try:
        with pytest.raises(IntentRoutingError):
            router.route("Anything")
    finally:
        _cleanup_router_session(router)


def test_system_context_normalization(tmp_path):
    router = IntentRouter(DummyLLM(None), settings=Settings())
    router._system_context = router._normalise_system_context(
        {"available_tools": ["read", None, 5], "command_separator": None}
    )
    assert router._system_context["available_tools"] == ["read", "5"]
    assert router._system_context["command_separator"] == "&&"
