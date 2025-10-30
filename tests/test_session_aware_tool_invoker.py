import uuid
from pathlib import Path

from ai_dev_agent.core.utils.config import Settings
from ai_dev_agent.engine.react.tool_invoker import SessionAwareToolInvoker
from ai_dev_agent.engine.react.types import ActionRequest
from ai_dev_agent.session import SessionManager
from ai_dev_agent.tools import READ, RUN


def test_session_aware_tool_invoker_formats_and_records(monkeypatch, tmp_path):
    session_manager = SessionManager.get_instance()
    session_id = f"session-{uuid.uuid4()}"
    session_manager.ensure_session(session_id, system_messages=[])

    invoker = SessionAwareToolInvoker(
        workspace=Path(tmp_path),
        settings=Settings(workspace_root=Path(tmp_path)),
        session_manager=session_manager,
        session_id=session_id,
    )

    monkeypatch.setattr(
        SessionAwareToolInvoker,
        "_invoke_registry",
        lambda self, tool_name, payload: {"summary": "ok"},
    )

    action = ActionRequest(
        step_id="S1",
        thought="test",
        tool="custom.tool",
        args={"value": 1},
    )

    observation = invoker(action)

    assert "custom.tool" in observation.display_message
    assert observation.formatted_output is not None
    assert observation.artifact_path is None

    session = session_manager.get_session(session_id)
    with session.lock:
        assert session.history
        tool_message = session.history[-1]
    assert tool_message.role == "tool"
    assert "custom.tool" in (tool_message.content or "")


def test_read_line_count_matches_wc(monkeypatch, tmp_path):
    session_manager = SessionManager.get_instance()
    session_id = f"session-read-{uuid.uuid4()}"
    session_manager.ensure_session(session_id, system_messages=[])

    invoker = SessionAwareToolInvoker(
        workspace=Path(tmp_path),
        settings=Settings(workspace_root=Path(tmp_path)),
        session_manager=session_manager,
        session_id=session_id,
    )

    def fake_invoke(self, tool_name, payload):
        assert tool_name == READ
        return {
            "files": [
                {
                    "path": "demo.py",
                    "content": "line1\nline2\nline3\n",
                }
            ]
        }

    monkeypatch.setattr(SessionAwareToolInvoker, "_invoke_registry", fake_invoke)

    action = ActionRequest(
        step_id="S1",
        thought="Inspect file",
        tool=READ,
        args={"path": "demo.py"},
    )

    observation = invoker(action)

    assert observation.metrics["lines_read"] == 3
    assert "3 lines read" in (observation.display_message or "")


def test_run_display_message_includes_stdout_preview(monkeypatch, tmp_path):
    session_manager = SessionManager.get_instance()
    session_id = f"session-run-{uuid.uuid4()}"
    session_manager.ensure_session(session_id, system_messages=[])

    invoker = SessionAwareToolInvoker(
        workspace=Path(tmp_path),
        settings=Settings(workspace_root=Path(tmp_path)),
        session_manager=session_manager,
        session_id=session_id,
    )

    def fake_invoke(self, tool_name, payload):
        assert tool_name == RUN
        return {
            "exit_code": 0,
            "stdout_tail": "277 ai_dev_agent/cli/commands.py",
            "stderr_tail": "",
        }

    monkeypatch.setattr(SessionAwareToolInvoker, "_invoke_registry", fake_invoke)

    action = ActionRequest(
        step_id="S1",
        thought="Count lines",
        tool=RUN,
        args={"cmd": "wc -l ai_dev_agent/cli/commands.py"},
    )

    observation = invoker(action)

    display = observation.display_message or ""
    assert "277 ai_dev_agent/cli/commands.py" in display
    assert "(stdout:" in display
    assert (
        observation.formatted_output
        and "277 ai_dev_agent/cli/commands.py" in observation.formatted_output
    )
    session = session_manager.get_session(session_id)
    with session.lock:
        tool_messages = [msg for msg in session.history if msg.role == "tool"]
    assert any("277 ai_dev_agent/cli/commands.py" in (msg.content or "") for msg in tool_messages)
