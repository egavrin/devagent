"""Focused tests for core ShellSession behaviors and error handling."""

from __future__ import annotations

import os
import stat
import sys
import threading
from pathlib import Path
from typing import Callable

import pytest

from ai_dev_agent.tools.execution import shell_session
from ai_dev_agent.tools.execution.shell_session import (
    ShellSession,
    ShellSessionError,
    ShellSessionTimeout,
)

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="ShellSession relies on POSIX shell semantics"
)


def test_resolve_shell_env_and_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure shell resolution honours $SHELL and falls back cleanly."""

    monkeypatch.delenv("SHELL", raising=False)

    calls: list[str] = []

    def _fake_which(candidate: str) -> str | None:
        calls.append(candidate)
        return None

    monkeypatch.setattr(shell_session.shutil, "which", _fake_which)

    resolved = shell_session._resolve_shell_executable(None)  # type: ignore[attr-defined]
    assert resolved == ["/bin/sh"]
    assert calls == ["bash", "zsh", "sh"]


def test_shell_session_rejects_non_executable_shell(tmp_path: Path) -> None:
    """ShellSession should convert launch failures into ShellSessionError."""

    fake_shell = tmp_path / "fake-shell"
    fake_shell.write_text("#!/bin/sh\necho 'should not run'\n", encoding="utf-8")
    fake_shell.chmod(stat.S_IRUSR | stat.S_IWUSR)  # Ensure execute bit is missing

    with pytest.raises(ShellSessionError) as excinfo:
        ShellSession(shell=str(fake_shell), cwd=tmp_path)

    assert str(fake_shell) in str(excinfo.value)
    assert "launch" in str(excinfo.value).lower()


def test_shell_session_consumes_stderr_with_injected_consumer(tmp_path: Path) -> None:
    """Inject a synchronous stderr consumer to exercise error-path coverage."""

    class ImmediateRunner(threading.Thread):
        def __init__(self, target: Callable[[], None]) -> None:
            super().__init__(target=target, daemon=True)
            self._target = target

        def start(self) -> None:
            self._target()

    session = ShellSession(
        shell=["/bin/sh"],
        cwd=tmp_path,
        stderr_consumer_factory=lambda target: ImmediateRunner(target),
    )
    try:
        result = session.run(
            [
                sys.executable,
                "-c",
                "import sys; print('stdout-line'); print('stderr-line', file=sys.stderr); sys.exit(7)",
            ]
        )
    finally:
        session.close()

    assert result.exit_code == 7
    assert "stdout-line" in result.stdout
    assert "stderr-line" in result.stderr
    assert result.duration_ms >= 0


def test_shell_session_timeout_closes_session(tmp_path: Path) -> None:
    """A timed-out command should close the session and prevent further use."""

    session = ShellSession(shell=["/bin/sh"], cwd=tmp_path)
    with pytest.raises(ShellSessionTimeout):
        session.run([sys.executable, "-c", "import time; time.sleep(1)"], timeout=0.1)

    with pytest.raises(ShellSessionError):
        session.run("echo still-closed")


def test_resolve_shell_handles_non_standard_type() -> None:
    """Objects with __str__ should be coerced into a command list."""

    class CustomShell:
        def __str__(self) -> str:
            return "/custom/shell"

    resolved = shell_session._resolve_shell_executable(CustomShell())  # type: ignore[attr-defined]
    assert resolved == ["/custom/shell"]


def test_shell_session_applies_custom_environment(tmp_path: Path) -> None:
    """Environment mappings with non-string values should be coerced."""

    session = ShellSession(shell=["/bin/sh"], cwd=tmp_path, env={"CUSTOM_VAR": 123})
    try:
        result = session.run('printf "%s" "$CUSTOM_VAR"')
    finally:
        session.close()

    assert result.stdout.strip() == "123"


def test_shell_session_validates_process_pipes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing pipes from Popen should raise an immediate ShellSessionError."""

    class DummyProcess:
        stdin = None
        stdout = None
        stderr = None

        def __init__(self, *_, **__):
            self.returncode = 0

        def poll(self):
            return 0

    monkeypatch.setattr(shell_session.subprocess, "Popen", lambda *args, **kwargs: DummyProcess())

    with pytest.raises(ShellSessionError):
        ShellSession(shell=["/bin/sh"], cwd=tmp_path)
