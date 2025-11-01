"""Tests covering command execution error scenarios for ShellSession."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from ai_dev_agent.tools.execution import shell_session
from ai_dev_agent.tools.execution.shell_session import (
    ShellSession,
    ShellSessionError,
    ShellSessionManager,
    ShellSessionTimeout,
)

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="ShellSession relies on POSIX shell semantics"
)


class _BrokenPipeStream:
    def __init__(self) -> None:
        self.closed = False

    def write(self, _: str) -> None:
        raise BrokenPipeError("simulated broken pipe")

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


def _echo_command(text: str) -> list[str]:
    return [sys.executable, "-c", f"print({text!r})"]


def test_run_handles_broken_pipe_and_closes_session(tmp_path: Path) -> None:
    """Broken stdin should surface a ShellSessionError and close the session."""

    session = ShellSession(shell=["/bin/sh"], cwd=tmp_path)
    session._stdin = _BrokenPipeStream()  # type: ignore[assignment]

    try:
        with pytest.raises(ShellSessionError) as excinfo:
            session.run("echo unreachable")

        assert "pipe closed" in str(excinfo.value).lower()
        with pytest.raises(ShellSessionError):
            session.run("echo still-unreachable")
    finally:
        session.close()


def test_manager_clears_session_on_failure(tmp_path: Path) -> None:
    """ShellSessionManager should remove sessions that crash mid-command."""

    manager = ShellSessionManager(shell=["/bin/sh"])
    session_id = manager.create_session(cwd=tmp_path)

    original_session = manager.get_session(session_id)
    original_session._process.terminate()  # Force the shell to exit unexpectedly

    with pytest.raises(ShellSessionError):
        manager.execute(session_id, _echo_command("will fail"))

    assert not manager.is_session_active(session_id)

    try:
        new_session = manager.create_session(cwd=tmp_path)
        result = manager.execute(new_session, _echo_command("recovered"))
        assert "recovered" in result.stdout
    finally:
        manager.close_all()


def test_run_supports_positive_timeout(tmp_path: Path) -> None:
    """Quick commands with a timeout should still succeed."""

    session = ShellSession(shell=["/bin/sh"], cwd=tmp_path)
    try:
        result = session.run("printf 'done\\n'", timeout=1.0)
    finally:
        session.close()

    assert result.exit_code == 0
    assert "done" in result.stdout


def test_run_timeout_without_select(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Disabling select should still trigger timeout handling."""

    session = ShellSession(shell=["/bin/sh"], cwd=tmp_path)
    monkeypatch.setattr(shell_session, "_HAS_SELECT", False)

    with pytest.raises(ShellSessionTimeout):
        session.run([sys.executable, "-c", "import time; time.sleep(1)"], timeout=0.1)

    session.close()
