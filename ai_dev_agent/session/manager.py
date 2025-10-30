"""Session lifecycle management for DevAgent."""

from __future__ import annotations

from datetime import datetime, timedelta
from threading import RLock, Timer
from typing import TYPE_CHECKING, Any, Callable
from uuid import uuid4

from ai_dev_agent.providers.llm.base import Message

from .context_service import ContextPruningConfig, ContextPruningService
from .models import Session

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .summarizer import ConversationSummarizer

_UNSET = object()


class SessionManager:
    """Singleton responsible for creating and tracking conversational sessions."""

    _instance: SessionManager | None = None
    DEFAULT_SESSION_TTL = timedelta(minutes=30)  # 30 minutes default TTL
    DEFAULT_MAX_SESSIONS = 100  # Maximum number of concurrent sessions
    CLEANUP_INTERVAL = 60  # Run cleanup every 60 seconds

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = RLock()
        self._context_service = ContextPruningService()
        self._session_ttl = self.DEFAULT_SESSION_TTL
        self._max_sessions = self.DEFAULT_MAX_SESSIONS
        self._cleanup_timer: Timer | None = None
        self._start_cleanup_timer()

    def configure_context_service(
        self,
        config: ContextPruningConfig | None = None,
        *,
        summarizer: ConversationSummarizer | None | object = _UNSET,
    ) -> None:
        """Replace the active context pruning configuration or summarizer."""

        with self._lock:
            if config is None and summarizer is _UNSET:
                self._context_service = ContextPruningService()
                return

            if config is not None and summarizer is _UNSET:
                self._context_service = ContextPruningService(config)
                return

            base_config = (
                config or getattr(self._context_service, "config", None) or ContextPruningConfig()
            )
            if summarizer is _UNSET:
                summarizer_to_use = getattr(self._context_service, "summarizer", None)
            else:
                summarizer_to_use = summarizer

            self._context_service = ContextPruningService(base_config, summarizer=summarizer_to_use)

    @classmethod
    def get_instance(cls) -> SessionManager:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def ensure_session(
        self,
        session_id: str | None = None,
        *,
        system_messages: Iterable[Message] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Session:
        """Return an existing session or create a new one with optional metadata."""
        if session_id is None:
            session_id = f"session-{uuid4()}"
        with self._lock:
            # Check if we're at max sessions and need to evict
            if session_id not in self._sessions and len(self._sessions) >= self._max_sessions:
                self._evict_oldest_session()

            session = self._sessions.get(session_id)
            if session is None:
                session = Session(id=session_id)
                self._sessions[session_id] = session
            else:
                # Update last accessed time for existing session
                session.last_accessed = datetime.now()
        if system_messages is not None:
            with session.lock:
                session.system_messages = list(system_messages)
        if metadata:
            with session.lock:
                session.metadata.update(metadata)
        return session

    def has_session(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._sessions

    def get_session(self, session_id: str) -> Session:
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session '{session_id}' does not exist")
            session = self._sessions[session_id]
            session.last_accessed = datetime.now()
            return session

    def list_sessions(self) -> list[str]:
        with self._lock:
            return list(self._sessions.keys())

    def compose(self, session_id: str) -> list[Message]:
        session = self.get_session(session_id)
        with session.lock:
            return session.compose()

    def extend_history(self, session_id: str, messages: Iterable[Message]) -> None:
        session = self.get_session(session_id)
        with session.lock:
            session.history.extend(messages)
            self._context_service.update_session(session)

    def add_user_message(self, session_id: str, content: str) -> Message:
        message = Message(role="user", content=content)
        self._append_history(session_id, message)
        return message

    def add_assistant_message(
        self,
        session_id: str,
        content: str | None,
        *,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> Message:
        message = Message(role="assistant", content=content, tool_calls=tool_calls)
        self._append_history(session_id, message)
        return message

    def add_tool_message(self, session_id: str, tool_call_id: str | None, content: str) -> Message:
        session = self.get_session(session_id)
        normalized_id = self._normalize_tool_call_id(session, tool_call_id)
        message = Message(role="tool", content=content, tool_call_id=normalized_id)
        self._append_history(session_id, message)
        return message

    def add_system_message(
        self,
        session_id: str,
        content: str,
        *,
        location: str = "history",
    ) -> Message:
        message = Message(role="system", content=content)
        session = self.get_session(session_id)
        with session.lock:
            if location == "system":
                session.system_messages.append(message)
            else:
                session.history.append(message)
                self._context_service.update_session(session)
        return message

    def remove_system_messages(self, session_id: str, predicate: Callable[[Message], bool]) -> None:
        session = self.get_session(session_id)
        with session.lock:
            session.system_messages = [msg for msg in session.system_messages if not predicate(msg)]
            session.history = [
                msg for msg in session.history if not (msg.role == "system" and predicate(msg))
            ]

    def _append_history(self, session_id: str, message: Message) -> None:
        session = self.get_session(session_id)
        with session.lock:
            session.history.append(message)
            self._context_service.update_session(session)

    def _normalize_tool_call_id(self, session: Session, tool_call_id: str | None) -> str:
        if isinstance(tool_call_id, str):
            stripped = tool_call_id.strip()
            if stripped:
                return stripped

        return self._generate_tool_call_id(session)

    def _generate_tool_call_id(self, session: Session) -> str:
        with session.lock:
            used: set[str] = {
                msg.tool_call_id
                for msg in session.history
                if msg.role == "tool" and msg.tool_call_id
            }

        while True:
            candidate = f"tool-{uuid4().hex[:8]}"
            if candidate not in used:
                return candidate

    def _start_cleanup_timer(self) -> None:
        """Start the background cleanup timer."""
        if self._cleanup_timer:
            self._cleanup_timer.cancel()

        self._cleanup_timer = Timer(self.CLEANUP_INTERVAL, self._run_cleanup)
        self._cleanup_timer.daemon = True
        self._cleanup_timer.start()

    def _run_cleanup(self) -> None:
        """Run periodic cleanup of expired sessions."""
        try:
            self.cleanup_expired_sessions()
        finally:
            # Schedule next cleanup
            self._start_cleanup_timer()

    def cleanup_expired_sessions(self) -> int:
        """Remove expired sessions based on TTL. Returns number of sessions removed."""
        now = datetime.now()
        expired_ids = []

        with self._lock:
            for session_id, session in self._sessions.items():
                if now - session.last_accessed > self._session_ttl:
                    expired_ids.append(session_id)

            for session_id in expired_ids:
                del self._sessions[session_id]

        return len(expired_ids)

    def _evict_oldest_session(self) -> None:
        """Evict the oldest session when at max capacity."""
        with self._lock:
            if not self._sessions:
                return

            # Find the oldest session by last_accessed time
            oldest_id = min(
                self._sessions.keys(), key=lambda sid: self._sessions[sid].last_accessed
            )
            del self._sessions[oldest_id]

    def set_session_ttl(self, ttl_minutes: int) -> None:
        """Configure the session TTL in minutes."""
        self._session_ttl = timedelta(minutes=ttl_minutes)

    def set_max_sessions(self, max_sessions: int) -> None:
        """Configure the maximum number of concurrent sessions."""
        self._max_sessions = max_sessions

    def get_session_stats(self) -> dict[str, Any]:
        """Get statistics about current sessions."""
        with self._lock:
            now = datetime.now()
            stats = {
                "total_sessions": len(self._sessions),
                "max_sessions": self._max_sessions,
                "ttl_minutes": self._session_ttl.total_seconds() / 60,
                "sessions": [],
            }

            for session_id, session in self._sessions.items():
                age = now - session.created_at
                idle_time = now - session.last_accessed
                stats["sessions"].append(
                    {
                        "id": session_id,
                        "age_seconds": age.total_seconds(),
                        "idle_seconds": idle_time.total_seconds(),
                        "message_count": len(session.history),
                    }
                )

            return stats

    def shutdown(self) -> None:
        """Shutdown the session manager and cleanup resources."""
        if self._cleanup_timer:
            self._cleanup_timer.cancel()
            self._cleanup_timer = None
