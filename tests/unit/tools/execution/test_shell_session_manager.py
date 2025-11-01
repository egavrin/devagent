"""Unit tests for ai_dev_agent.tools.execution.shell_session."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from ai_dev_agent.tools.execution.shell_session import (
    ShellCommandHistoryEntry,
    ShellSessionError,
    ShellSessionManager,
    ShellSessionTimeout,
)


def _python_command(code: str) -> list[str]:
    """Helper to build a python command executed in the shell session."""

    return [sys.executable, "-c", code]


def test_shell_session_management(tmp_path: Path) -> None:
    manager = ShellSessionManager()
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    session_id = manager.create_session(cwd=workspace)
    assert manager.is_session_active(session_id)
    assert session_id in set(manager.active_sessions())

    with pytest.raises(ShellSessionError):
        manager.create_session(session_id=session_id, cwd=workspace)

    manager.close_session(session_id)
    assert not manager.is_session_active(session_id)

    secondary = manager.create_session(cwd=workspace)
    tertiary = manager.create_session(cwd=workspace)
    assert {secondary, tertiary} == set(manager.active_sessions())

    manager.close_all()
    assert not list(manager.active_sessions())

    with pytest.raises(FileNotFoundError):
        manager.create_session(cwd=workspace / "missing")


def test_shell_command_execution(tmp_path: Path) -> None:
    manager = ShellSessionManager()
    session_id = manager.create_session(cwd=tmp_path)

    command = _python_command(
        'import sys; print("stdout content"); print("stderr content", file=sys.stderr); sys.exit(3)'
    )
    result = manager.execute(session_id, command)
    assert result.exit_code == 3
    assert result.stdout.strip() == "stdout content"
    assert result.stderr.strip() == "stderr content"
    assert result.duration_ms >= 0

    if os.name != "nt":
        with pytest.raises(ShellSessionTimeout):
            manager.execute(
                session_id,
                _python_command("import time; time.sleep(0.5)"),
                timeout=0.1,
            )
        assert not manager.is_session_active(session_id)
    else:  # pragma: no cover - Windows timeout semantics rely on different mechanisms
        manager.execute(
            session_id,
            _python_command("import time; time.sleep(0.05)"),
            timeout=0.1,
        )
        manager.close_session(session_id)

    new_session = manager.create_session(cwd=tmp_path)
    manager.close_session(new_session)
    with pytest.raises(ShellSessionError):
        manager.execute(new_session, "echo should not run")


def test_shell_history_tracking(tmp_path: Path) -> None:
    manager = ShellSessionManager()
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()

    session_a = manager.create_session(cwd=dir_a)
    session_b = manager.create_session(cwd=dir_b)

    first = manager.execute(session_a, 'printf "first\\n"')
    assert first.exit_code == 0

    manager.execute(session_a, 'export SAMPLE_VAR="alpha"')
    second = manager.execute(session_a, 'printf "%s" "$SAMPLE_VAR"')
    assert second.stdout.strip() == "alpha"

    isolated = manager.execute(session_b, 'printf "%s" "${SAMPLE_VAR:-missing}"')
    assert isolated.stdout.strip() == "missing"

    history_a = manager.get_history(session_a)
    assert all(isinstance(entry, ShellCommandHistoryEntry) for entry in history_a)
    assert [entry.command for entry in history_a][-1] == 'printf "%s" "$SAMPLE_VAR"'
    assert history_a[0].result.stdout.strip() == "first"

    replay = manager.replay_history(session_a, 0)
    assert replay.stdout.strip() == "first"

    history_a_updated = manager.get_history(session_a)
    assert len(history_a_updated) == len(history_a) + 1

    history_b = manager.get_history(session_b)
    assert len(history_b) == 1
    assert history_b[0].result.stdout.strip() == "missing"

    replay_b = manager.replay_history(session_b)
    assert replay_b.stdout.strip() == "missing"

    manager.close_all()


def test_history_retention_limit(tmp_path: Path) -> None:
    manager = ShellSessionManager(max_history_entries=2)
    session_id = manager.create_session(cwd=tmp_path)

    manager.execute(session_id, 'printf "first\\n"')
    manager.execute(session_id, 'printf "second\\n"')
    manager.execute(session_id, 'printf "third\\n"')

    history = manager.get_history(session_id)
    assert len(history) == 2
    assert [entry.result.stdout.strip() for entry in history] == ["second", "third"]

    manager.execute(session_id, 'printf "fourth\\n"')
    history = manager.get_history(session_id)
    assert [entry.result.stdout.strip() for entry in history] == ["third", "fourth"]


def test_history_clearing_api(tmp_path: Path) -> None:
    manager = ShellSessionManager()
    session_a = manager.create_session(cwd=tmp_path)
    session_b = manager.create_session(cwd=tmp_path)

    manager.execute(session_a, 'printf "one\\n"')
    manager.execute(session_a, 'printf "two\\n"')
    manager.execute(session_a, 'printf "three\\n"')
    assert len(manager.get_history(session_a)) == 3

    manager.clear_history(session_a, keep_last=1)
    history_a = manager.get_history(session_a)
    assert len(history_a) == 1
    assert history_a[0].result.stdout.strip() == "three"

    manager.execute(session_b, 'printf "alpha\\n"')
    manager.execute(session_b, 'printf "beta\\n"')
    assert len(manager.get_history(session_b)) == 2

    manager.clear_all_history()
    assert manager.get_history(session_a) == []
    assert manager.get_history(session_b) == []
