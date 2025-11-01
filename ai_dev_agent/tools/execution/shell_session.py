"""Persistent shell session management for command execution."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from ai_dev_agent.core.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

LOGGER = get_logger(__name__)

try:  # pragma: no cover - platform dependent
    import resource
except ImportError:  # pragma: no cover - Windows
    resource = None  # type: ignore

try:  # pragma: no cover - platform dependent
    import select
except ImportError:  # pragma: no cover - minimal python builds
    select = None  # type: ignore

_HAS_SELECT = select is not None and os.name != "nt"


class ShellSessionError(RuntimeError):
    """Raised when a shell session encounters an unrecoverable error."""


class ShellSessionTimeout(TimeoutError):
    """Raised when a command exceeds the allowed execution time."""


@dataclass
class ShellCommandResult:
    """Structured result from executing a shell command inside a session."""

    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int


@dataclass
class ShellCommandHistoryEntry:
    """Record of a command executed inside a shell session."""

    command: str
    result: ShellCommandResult


def _resolve_shell_executable(shell: str | Sequence[str] | None) -> list[str]:
    """Return the command list used to spawn the interactive shell."""

    if not shell:
        candidate = os.environ.get("SHELL")
        if not candidate:
            candidate = shutil.which("bash") or shutil.which("zsh") or shutil.which("sh")
        shell = candidate or "/bin/sh"

    if isinstance(shell, (list, tuple)):
        return list(shell)
    if isinstance(shell, str):
        return shlex.split(shell)
    # This shouldn't happen given the type signature, but handle it defensively
    return [str(shell)]


def _stringify_command(command: Sequence[str] | str) -> str:
    """Return a shell-friendly string representation of the command."""

    if isinstance(command, str):
        return command
    return " ".join(shlex.quote(str(part)) for part in command)


def _make_preexec_fn(
    cpu_limit: int | None, memory_limit_mb: int | None
):  # pragma: no cover - posix specific
    if resource is None:
        return None

    if cpu_limit is None and memory_limit_mb is None:
        return None

    cpu_limit = cpu_limit if cpu_limit and cpu_limit > 0 else None
    memory_limit_mb = memory_limit_mb if memory_limit_mb and memory_limit_mb > 0 else None

    if cpu_limit is None and memory_limit_mb is None:
        return None

    def apply_limits() -> None:
        if cpu_limit is not None:
            try:
                resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
            except (ValueError, OSError) as exc:
                LOGGER.warning("Failed to set CPU limit for shell session: %s", exc)
        if memory_limit_mb is not None:
            byte_limit = memory_limit_mb * 1024 * 1024
            try:
                resource.setrlimit(resource.RLIMIT_AS, (byte_limit, byte_limit))
            except (ValueError, OSError) as exc:
                LOGGER.warning("Failed to set memory limit for shell session: %s", exc)

    return apply_limits


class ShellSession:
    """Encapsulates a running shell process that maintains state between commands."""

    def __init__(
        self,
        shell: Sequence[str] | str | None = None,
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        cpu_time_limit: int | None = None,
        memory_limit_mb: int | None = None,
        stderr_consumer_factory: Callable[[Callable[[], None]], threading.Thread] | None = None,
    ) -> None:
        command = _resolve_shell_executable(shell)
        run_cwd = Path(cwd).resolve() if cwd else Path.cwd()
        if not run_cwd.exists():
            raise FileNotFoundError(f"Shell session cwd does not exist: {run_cwd}")

        shell_env = os.environ.copy()
        if env:
            shell_env.update({str(key): str(value) for key, value in env.items()})
        shell_env.setdefault("PYTHONUNBUFFERED", "1")

        preexec_fn = _make_preexec_fn(cpu_time_limit, memory_limit_mb)

        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(run_cwd),
                env=shell_env,
                text=True,
                bufsize=1,
                universal_newlines=True,
                preexec_fn=preexec_fn,
            )
        except OSError as exc:
            executable = command[0] if command else "<unknown>"
            raise ShellSessionError(f"Failed to launch shell '{executable}': {exc}") from exc

        if not self._process.stdin or not self._process.stdout or not self._process.stderr:
            raise ShellSessionError("Failed to initialise shell session pipes.")

        self._stdin = self._process.stdin
        self._stdout = self._process.stdout
        self._stderr = self._process.stderr
        self._lock = threading.Lock()
        self._closed = False
        self._shell_description = " ".join(command)
        self._stderr_consumer_factory: Callable[[Callable[[], None]], threading.Thread]
        if stderr_consumer_factory is None:
            self._stderr_consumer_factory = lambda target: threading.Thread(
                target=target, daemon=True
            )
        else:
            self._stderr_consumer_factory = stderr_consumer_factory

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self, command: Sequence[str] | str, *, timeout: float | None = None
    ) -> ShellCommandResult:
        """Execute a command in the persistent session."""

        if self._closed:
            raise ShellSessionError("Shell session is closed.")

        with self._lock:
            return self._run_locked(command, timeout=timeout)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._process.poll() is None:
                try:
                    self._stdin.write("exit\n")
                    self._stdin.flush()
                except Exception:
                    pass
                self._process.terminate()
        finally:
            try:
                self._stdin.close()
            except Exception:
                pass
            try:
                self._stdout.close()
            except Exception:
                pass
            try:
                self._stderr.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_locked(
        self, command: Sequence[str] | str, *, timeout: float | None
    ) -> ShellCommandResult:
        cmd_string = self._coerce_command(command)
        sentinel = f"__DEVAGENT_DONE__{uuid.uuid4().hex}__"
        exit_marker = f"{sentinel}:"

        try:
            self._stdin.write(f"{cmd_string}\n")
            self._stdin.write("__DEVAGENT_STATUS=$?\n")
            self._stdin.write(f'printf "%s%s\\n" "{exit_marker}" "$__DEVAGENT_STATUS"\n')
            self._stdin.write(f'printf "%s%s\\n" "{exit_marker}" "$__DEVAGENT_STATUS" >&2\n')
            self._stdin.write("unset __DEVAGENT_STATUS\n")
            self._stdin.flush()
        except BrokenPipeError as exc:
            self.close()
            raise ShellSessionError("Shell session pipe closed unexpectedly") from exc

        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []
        stderr_exit_holder: dict[str, int | None] = {"code": None}
        stderr_done = threading.Event()

        def consume_stderr() -> None:
            try:
                while True:
                    line = self._stderr.readline()
                    if line == "":
                        break
                    marker_index = line.find(exit_marker)
                    if marker_index != -1:
                        prefix = line[:marker_index]
                        if prefix:
                            stderr_chunks.append(prefix)
                        try:
                            remainder = line[marker_index + len(exit_marker) :]
                            stderr_exit_holder["code"] = int(remainder.strip())
                        except ValueError:
                            stderr_exit_holder["code"] = None
                        break
                    stderr_chunks.append(line)
            finally:
                stderr_done.set()

        try:
            stderr_thread = self._stderr_consumer_factory(consume_stderr)
        except Exception as exc:  # pragma: no cover - defensive
            raise ShellSessionError("Failed to prepare stderr consumer.") from exc

        if not hasattr(stderr_thread, "start"):  # pragma: no cover - defensive guard
            raise ShellSessionError("Stderr consumer factory must return an object with start().")

        stderr_thread.start()

        exit_code: int | None = None
        start_time = time.perf_counter()

        try:
            while True:
                wait_timeout = None
                if timeout is not None and timeout > 0:
                    elapsed = time.perf_counter() - start_time
                    remaining = timeout - elapsed
                    if remaining <= 0:
                        raise ShellSessionTimeout(f"Command exceeded timeout of {timeout} seconds")
                    wait_timeout = remaining

                if wait_timeout is not None and _HAS_SELECT:
                    ready, _, _ = select.select([self._stdout], [], [], wait_timeout)
                    if not ready:
                        raise ShellSessionTimeout(f"Command exceeded timeout of {timeout} seconds")

                line = self._stdout.readline()
                if wait_timeout is not None and not _HAS_SELECT:
                    elapsed = time.perf_counter() - start_time
                    if elapsed > timeout:
                        raise ShellSessionTimeout(f"Command exceeded timeout of {timeout} seconds")
                if line == "":
                    if self._process.poll() is not None:
                        raise ShellSessionError("Shell session terminated unexpectedly.")
                    continue
                marker_index = line.find(exit_marker)
                if marker_index != -1:
                    prefix = line[:marker_index]
                    if prefix:
                        stdout_chunks.append(prefix)
                    try:
                        remainder = line[marker_index + len(exit_marker) :]
                        exit_code = int(remainder.strip())
                    except ValueError:
                        exit_code = stderr_exit_holder.get("code") or 1
                    break
                stdout_chunks.append(line)
        except Exception:
            self._handle_command_failure()
            raise
        finally:
            remaining_timeout = None
            if timeout is not None and timeout > 0:
                elapsed = time.perf_counter() - start_time
                remaining_timeout = max(0.0, timeout - elapsed)
            stderr_done.wait(timeout=remaining_timeout)

        duration_ms = int((time.perf_counter() - start_time) * 1000)
        if exit_code is None:
            exit_code = stderr_exit_holder.get("code")
        if exit_code is None:
            raise ShellSessionError("Failed to capture exit code from shell session command.")

        stdout = "".join(stdout_chunks)
        stderr = "".join(stderr_chunks)
        return ShellCommandResult(
            exit_code=exit_code, stdout=stdout, stderr=stderr, duration_ms=duration_ms
        )

    def _coerce_command(self, command: Sequence[str] | str) -> str:
        return _stringify_command(command)

    def _handle_command_failure(self) -> None:
        try:
            self.close()
        except Exception:
            pass
        LOGGER.error("Shell session '%s' closed due to command failure.", self._shell_description)


class ShellSessionManager:
    """Manage one or more persistent shell sessions identified by unique IDs."""

    def __init__(
        self,
        *,
        shell: Sequence[str] | str | None = None,
        default_timeout: float | None = None,
        cpu_time_limit: int | None = None,
        memory_limit_mb: int | None = None,
        max_history_entries: int | None = None,
    ) -> None:
        self._shell = shell
        self._default_timeout = default_timeout
        self._cpu_limit = cpu_time_limit
        self._memory_limit = memory_limit_mb
        self._max_history_entries = (
            max_history_entries if max_history_entries and max_history_entries > 0 else None
        )
        self._sessions: dict[str, ShellSession] = {}
        self._history: dict[str, list[ShellCommandHistoryEntry]] = {}
        self._lock = threading.Lock()

    def create_session(
        self,
        *,
        session_id: str | None = None,
        shell: Sequence[str] | str | None = None,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> str:
        """Create a new shell session and return its ID."""

        sid = session_id or uuid.uuid4().hex
        with self._lock:
            if sid in self._sessions:
                raise ShellSessionError(f"Shell session '{sid}' already exists")
            session = ShellSession(
                shell=shell or self._shell,
                cwd=cwd,
                env=env,
                cpu_time_limit=self._cpu_limit,
                memory_limit_mb=self._memory_limit,
            )
            self._sessions[sid] = session
            self._history[sid] = []
        return sid

    def start_session(
        self,
        *,
        session_id: str | None = None,
        shell: Sequence[str] | str | None = None,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> str:
        """
        Backwards-compatible alias for create_session.

        Some CLI code paths (and tests) expect a start_session method. This
        delegates to create_session so existing behavior is preserved.
        """
        return self.create_session(
            session_id=session_id,
            shell=shell,
            cwd=cwd,
            env=env,
        )

    def get_session(self, session_id: str) -> ShellSession:
        with self._lock:
            if session_id not in self._sessions:
                raise ShellSessionError(f"Unknown shell session '{session_id}'")
            return self._sessions[session_id]

    def execute(
        self,
        session_id: str,
        command: Sequence[str] | str,
        *,
        timeout: float | None = None,
    ) -> ShellCommandResult:
        session = self.get_session(session_id)
        effective_timeout = timeout if timeout is not None else self._default_timeout
        command_text = _stringify_command(command)
        try:
            result = session.run(command, timeout=effective_timeout)
        except (ShellSessionError, ShellSessionTimeout):
            self.close_session(session_id)
            raise
        else:
            with self._lock:
                history = self._history.get(session_id)
                if history is not None:
                    history.append(
                        ShellCommandHistoryEntry(
                            command=command_text,
                            result=result,
                        )
                    )
                    self._enforce_history_limits(session_id)
            return result

    def close_session(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions.pop(session_id, None)
            self._history.pop(session_id, None)
        if session:
            session.close()

    def close_all(self) -> None:
        with self._lock:
            sessions = list(self._sessions.items())
            self._sessions.clear()
            self._history.clear()
        for _, session in sessions:
            session.close()

    def active_sessions(self) -> Iterable[str]:
        with self._lock:
            return list(self._sessions.keys())

    def is_session_active(self, session_id: str) -> bool:
        """Return True if the session exists and is still managed."""
        with self._lock:
            return session_id in self._sessions

    def get_history(self, session_id: str) -> list[ShellCommandHistoryEntry]:
        """Return a copy of the command history for the specified session."""

        with self._lock:
            if session_id not in self._sessions:
                raise ShellSessionError(f"Unknown shell session '{session_id}'")
            entries = self._history.get(session_id, [])
            return [
                ShellCommandHistoryEntry(
                    command=entry.command,
                    result=ShellCommandResult(
                        exit_code=entry.result.exit_code,
                        stdout=entry.result.stdout,
                        stderr=entry.result.stderr,
                        duration_ms=entry.result.duration_ms,
                    ),
                )
                for entry in entries
            ]

    def replay_history(
        self,
        session_id: str,
        index: int = -1,
        *,
        timeout: float | None = None,
    ) -> ShellCommandResult:
        """Re-execute a previously recorded command by index."""

        with self._lock:
            if session_id not in self._sessions:
                raise ShellSessionError(f"Unknown shell session '{session_id}'")
            entries = self._history.get(session_id, [])
            if not entries:
                raise ShellSessionError(f"No recorded history for shell session '{session_id}'")
            try:
                command = entries[index].command
            except IndexError as exc:  # pragma: no cover - defensive
                raise ShellSessionError(
                    f"History index {index} out of range for shell session '{session_id}'"
                ) from exc
        return self.execute(session_id, command, timeout=timeout)

    def clear_history(self, session_id: str, *, keep_last: int = 0) -> None:
        """Remove history for a session, optionally retaining latest entries."""

        if keep_last < 0:
            raise ValueError("keep_last must be >= 0")

        with self._lock:
            if session_id not in self._sessions:
                raise ShellSessionError(f"Unknown shell session '{session_id}'")
            entries = self._history.get(session_id, [])
            if not entries:
                self._history[session_id] = []
                return
            if keep_last == 0:
                self._history[session_id] = []
            else:
                self._history[session_id] = entries[-keep_last:]

    def clear_all_history(self) -> None:
        """Remove history for all active sessions."""

        with self._lock:
            for session_id in list(self._history.keys()):
                self._history[session_id] = []

    def _enforce_history_limits(self, session_id: str) -> None:
        if self._max_history_entries is None:
            return
        entries = self._history.get(session_id)
        if not entries:
            return
        excess = len(entries) - self._max_history_entries
        if excess > 0:
            del entries[:excess]


__all__ = [
    "ShellCommandHistoryEntry",
    "ShellCommandResult",
    "ShellSession",
    "ShellSessionError",
    "ShellSessionManager",
    "ShellSessionTimeout",
]
