"""Direct command execution tool without sandbox restrictions."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ai_dev_agent.core.utils.constants import RUN_STDERR_TAIL_CHARS, RUN_STDOUT_TAIL_CHARS

from ..names import RUN
from ..registry import ToolContext, ToolSpec, registry
from .shell_session import ShellSessionError, ShellSessionManager, ShellSessionTimeout

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schemas" / "tools"


_SHELL_CONTROL_TOKENS = {"|", "||", "&&", ";", ";;", "(", ")"}
_SHELL_REDIRECTION_PATTERN = re.compile(r"^\d*(?:>>|>|<<|<<<|<|<>|>&|<&|&>|>\||\|&)")
_ENV_ASSIGNMENT_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\+?=.*")
_SUBSTITUTION_MARKERS: tuple[str, ...] = ("$(", "${", "`")


def _summarize_output(output: str | None, limit: int) -> str:
    """Return the trailing portion of ``output`` capped at ``limit`` with a truncation marker."""
    if not output:
        return ""
    if len(output) <= limit:
        return output
    tail = output[-limit:]
    omitted = len(output) - len(tail)
    return f"{tail}\n\n[... truncated: {omitted} characters omitted ...]"


def _wrap_with_shell(command: str) -> list[str]:
    if os.name == "nt":
        comspec = os.environ.get("COMSPEC", "cmd.exe")
        return [comspec, "/S", "/C", command]

    shell_path = os.environ.get("SHELL")
    if not shell_path:
        return ["/bin/sh", "-c", command]

    shell_name = Path(shell_path).name.lower()
    if shell_name in {"bash", "zsh", "fish", "ksh"}:
        return [shell_path, "-lc", command]
    return [shell_path, "-c", command]


def _contains_shell_controls(tokens: Sequence[str]) -> bool:
    for token in tokens:
        if not token:
            continue
        if token in _SHELL_CONTROL_TOKENS:
            return True
        if _SHELL_REDIRECTION_PATTERN.match(token):
            return True
        if _ENV_ASSIGNMENT_PATTERN.match(token):
            return True
        if any(marker in token for marker in _SUBSTITUTION_MARKERS):
            return True
    return False


def _build_command(cmd: str, args: Sequence[str] | None) -> list[str]:
    if args:
        tokens = [cmd, *args]
        if _contains_shell_controls(tokens):
            command = shlex.join(tokens)
            return _wrap_with_shell(command)
        return tokens

    tokens = shlex.split(cmd)
    if _contains_shell_controls(tokens):
        return _wrap_with_shell(cmd)
    return tokens


def _exec_command(payload: Mapping[str, Any], context: ToolContext) -> Mapping[str, Any]:
    repo_root = context.repo_root

    cmd = payload["cmd"]
    args = payload.get("args")
    cwd_value = payload.get("cwd")
    cwd = repo_root if cwd_value is None else (repo_root / cwd_value).resolve()
    if repo_root not in cwd.parents and cwd != repo_root:
        raise ValueError("cwd must be within repository root")
    timeout = payload.get("timeout_sec")
    env_overrides = payload.get("env")
    env = None
    if env_overrides:
        env = os.environ.copy()
        env.update({str(key): str(value) for key, value in env_overrides.items()})

    command = _build_command(cmd, args)

    session_command = shlex.join([cmd, *args]) if args else cmd

    extra = context.extra or {}
    manager = extra.get("shell_session_manager") if isinstance(extra, dict) else None
    session_id = extra.get("shell_session_id") if isinstance(extra, dict) else None

    if isinstance(manager, ShellSessionManager) and isinstance(session_id, str):
        command_str = session_command
        if cwd_value is not None:
            command_str = f"cd {shlex.quote(str(cwd))} && {command_str}"
        try:
            result = manager.execute(
                session_id,
                command_str,
                timeout=float(timeout) if timeout is not None else None,
            )
        except ShellSessionTimeout as exc:
            raise TimeoutError(str(exc)) from exc
        except ShellSessionError:
            raise

        stdout_tail = _summarize_output(result.stdout, RUN_STDOUT_TAIL_CHARS)
        stderr_tail = _summarize_output(result.stderr, RUN_STDERR_TAIL_CHARS)

        return {
            "exit_code": result.exit_code,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
            "log_path": None,
            "duration_ms": result.duration_ms,
        }

    start = time.perf_counter()
    process = subprocess.run(
        command,
        cwd=str(cwd),
        timeout=float(timeout) if timeout is not None else None,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    duration_ms = int((time.perf_counter() - start) * 1000)

    stdout_truncated = _summarize_output(process.stdout, RUN_STDOUT_TAIL_CHARS)
    stderr_truncated = _summarize_output(process.stderr, RUN_STDERR_TAIL_CHARS)

    return {
        "exit_code": process.returncode,
        "stdout_tail": stdout_truncated,
        "stderr_tail": stderr_truncated,
        "log_path": None,
        "duration_ms": duration_ms,
    }


registry.register(
    ToolSpec(
        name=RUN,
        handler=_exec_command,
        request_schema_path=SCHEMA_DIR / "run.request.json",
        response_schema_path=SCHEMA_DIR / "run.response.json",
        description=(
            "Execute a command directly. Provide 'cmd' (string) and optionally 'args' (list), "
            "'cwd' (string within the repo), and 'timeout_sec' (int)."
        ),
        category="command",
    )
)


__all__ = ["_exec_command"]
