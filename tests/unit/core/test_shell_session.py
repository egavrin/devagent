"""Tests covering shell session persistence and integration."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from ai_dev_agent.core.utils.config import Settings
from ai_dev_agent.tools import RUN, ToolContext
from ai_dev_agent.tools import registry as tool_registry
from ai_dev_agent.tools.execution.shell_session import ShellSessionManager


def _select_shell() -> str:
    candidates = [os.environ.get("SHELL"), "bash", "zsh", "sh"]
    for candidate in candidates:
        if not candidate:
            continue
        path = shutil.which(candidate) if os.path.sep not in candidate else candidate
        if path and Path(path).exists():
            return path
    pytest.skip("No suitable shell executable available for shell session tests")
    return ""


def _make_context(root: Path, manager: ShellSessionManager, session_id: str) -> ToolContext:
    settings = Settings(workspace_root=root)
    return ToolContext(
        repo_root=root,
        settings=settings,
        sandbox=None,
        devagent_config=None,
        metrics_collector=None,
        extra={
            "shell_session_manager": manager,
            "shell_session_id": session_id,
        },
    )


def test_shell_session_persists_state(tmp_path: Path) -> None:
    shell_path = _select_shell()
    manager = ShellSessionManager(shell=shell_path)
    session_id = manager.create_session(cwd=tmp_path)

    try:
        result = manager.execute(session_id, "pwd")
        assert str(tmp_path) in result.stdout.strip()

        manager.execute(session_id, "mkdir nested_dir")
        manager.execute(session_id, "cd nested_dir")
        nested_result = manager.execute(session_id, "pwd")
        assert nested_result.stdout.strip().endswith("nested_dir")
    finally:
        manager.close_all()


def test_exec_tool_uses_persistent_shell(tmp_path: Path) -> None:
    shell_path = _select_shell()
    manager = ShellSessionManager(shell=shell_path)
    session_id = manager.create_session(cwd=tmp_path)
    context = _make_context(tmp_path, manager, session_id)

    try:
        initial = tool_registry.invoke(RUN, {"cmd": "pwd"}, context)
        assert str(tmp_path) in initial["stdout_tail"].strip()

        nested = tmp_path / "persist"
        nested.mkdir()
        tool_registry.invoke(RUN, {"cmd": "cd", "args": ["persist"]}, context)
        follow_up = tool_registry.invoke(RUN, {"cmd": "pwd"}, context)
        assert str(nested) in follow_up["stdout_tail"].strip()
    finally:
        manager.close_all()


def test_shell_session_manager_timeout() -> None:
    """Test session manager with timeout."""
    shell_path = _select_shell()
    manager = ShellSessionManager(
        shell=shell_path,
        default_timeout=1,  # 1 second timeout
    )
    session_id = manager.create_session()

    try:
        # Command that would take too long
        result = manager.execute(session_id, "sleep 10", timeout=0.5)
        # Should timeout
        assert result.exit_code != 0
    except Exception:
        # Timeout is expected
        pass
    finally:
        manager.close_all()


def test_shell_session_manager_multiple_sessions(tmp_path: Path) -> None:
    """Test managing multiple shell sessions."""
    shell_path = _select_shell()
    manager = ShellSessionManager(shell=shell_path)

    session1 = manager.create_session(cwd=tmp_path)
    session2 = manager.create_session(cwd=tmp_path)

    try:
        # Get sessions and execute commands
        sess1 = manager.get_session(session1)
        sess2 = manager.get_session(session2)

        # Set different environment variables in each session
        sess1.run("export TEST_VAR=session1")
        sess2.run("export TEST_VAR=session2")

        # Verify each session maintains its own state
        result1 = sess1.run("echo $TEST_VAR")
        result2 = sess2.run("echo $TEST_VAR")

        assert "session1" in result1.stdout
        assert "session2" in result2.stdout

        # Check sessions are active
        assert manager.is_session_active(session1)
        assert manager.is_session_active(session2)
    finally:
        manager.close_all()


def test_shell_session_error_handling(tmp_path: Path) -> None:
    """Test error handling in shell sessions."""
    shell_path = _select_shell()
    manager = ShellSessionManager(shell=shell_path)
    session_id = manager.create_session(cwd=tmp_path)

    try:
        # Execute command that fails
        result = manager.execute(session_id, "false")
        assert result.exit_code != 0

        # Execute command with stderr output
        result = manager.execute(session_id, "echo error >&2")
        assert "error" in result.stderr or "error" in result.stdout

        # Invalid command
        result = manager.execute(session_id, "nonexistent_command_xyz")
        assert result.exit_code != 0
    finally:
        manager.close_all()


def test_shell_session_close_and_cleanup(tmp_path: Path) -> None:
    """Test closing sessions and cleanup."""
    shell_path = _select_shell()
    manager = ShellSessionManager(shell=shell_path)

    session1 = manager.create_session(cwd=tmp_path)
    session2 = manager.create_session(cwd=tmp_path)

    # Check both sessions are active
    assert manager.is_session_active(session1)
    assert manager.is_session_active(session2)

    # Close one session
    manager.close_session(session1)
    assert not manager.is_session_active(session1)
    assert manager.is_session_active(session2)

    # Close non-existent session (should not raise)
    manager.close_session("nonexistent")

    # Close all remaining
    manager.close_all()
    active = list(manager.active_sessions())
    assert len(active) == 0


def test_shell_session_with_environment(tmp_path: Path) -> None:
    """Test shell session with custom environment."""
    shell_path = _select_shell()
    manager = ShellSessionManager(shell=shell_path)
    session_id = manager.create_session(cwd=tmp_path, env={"CUSTOM_VAR": "custom_value"})

    try:
        result = manager.execute(session_id, "echo $CUSTOM_VAR")
        assert "custom_value" in result.stdout
    finally:
        manager.close_all()


def test_shell_session_directory_changes(tmp_path: Path) -> None:
    """Test directory navigation in shell sessions."""
    shell_path = _select_shell()
    manager = ShellSessionManager(shell=shell_path)
    session_id = manager.create_session(cwd=tmp_path)

    try:
        # Create nested directories
        (tmp_path / "dir1" / "dir2").mkdir(parents=True)

        # Navigate down
        manager.execute(session_id, "cd dir1")
        result = manager.execute(session_id, "pwd")
        assert "dir1" in result.stdout

        manager.execute(session_id, "cd dir2")
        result = manager.execute(session_id, "pwd")
        assert "dir2" in result.stdout

        # Navigate back up
        manager.execute(session_id, "cd ../..")
        result = manager.execute(session_id, "pwd")
        assert str(tmp_path) in result.stdout
    finally:
        manager.close_all()


# Removed test_shell_session_get_info as ShellSessionManager
# doesn't have a get_session_info method


def test_shell_session_command_with_args(tmp_path: Path) -> None:
    """Test executing commands with arguments."""
    shell_path = _select_shell()
    manager = ShellSessionManager(shell=shell_path)
    session_id = manager.create_session(cwd=tmp_path)

    try:
        # Simple echo with args
        result = manager.execute(session_id, 'echo "hello world"')
        assert "hello world" in result.stdout

        # Create file with args
        test_file = tmp_path / "test.txt"
        manager.execute(session_id, f'echo "content" > {test_file}')
        assert test_file.exists()
        assert "content" in test_file.read_text()
    finally:
        manager.close_all()
