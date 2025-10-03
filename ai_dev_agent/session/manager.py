"""Session lifecycle management for DevAgent."""
from __future__ import annotations

from threading import RLock
from typing import Any, Callable, Dict, Iterable, List, Optional
from uuid import uuid4

from ai_dev_agent.providers.llm.base import Message

from .models import Session
from .context_service import ContextPruningConfig, ContextPruningService
from .summarizer import ConversationSummarizer

_UNSET = object()


class SessionManager:
    """Singleton responsible for creating and tracking conversational sessions."""

    _instance: "SessionManager" | None = None

    def __init__(self) -> None:
        self._sessions: Dict[str, Session] = {}
        self._lock = RLock()
        self._context_service = ContextPruningService()

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

            base_config = config or getattr(self._context_service, "config", None) or ContextPruningConfig()
            if summarizer is _UNSET:
                summarizer_to_use = getattr(self._context_service, "summarizer", None)
            else:
                summarizer_to_use = summarizer

            self._context_service = ContextPruningService(base_config, summarizer=summarizer_to_use)

    @classmethod
    def get_instance(cls) -> "SessionManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def ensure_session(
        self,
        session_id: Optional[str] = None,
        *,
        system_messages: Optional[Iterable[Message]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Session:
        """Return an existing session or create a new one with optional metadata."""
        if session_id is None:
            session_id = f"session-{uuid4()}"
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = Session(id=session_id)
                self._sessions[session_id] = session
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
            return self._sessions[session_id]

    def list_sessions(self) -> List[str]:
        with self._lock:
            return list(self._sessions.keys())

    def compose(self, session_id: str) -> List[Message]:
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
        content: Optional[str],
        *,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> Message:
        message = Message(role="assistant", content=content, tool_calls=tool_calls)
        self._append_history(session_id, message)
        return message

    def add_tool_message(self, session_id: str, tool_call_id: str | None, content: str) -> Message:
        message = Message(role="tool", content=content, tool_call_id=tool_call_id)
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
            session.history = [msg for msg in session.history if not (msg.role == "system" and predicate(msg))]

    def _append_history(self, session_id: str, message: Message) -> None:
        session = self.get_session(session_id)
        with session.lock:
            session.history.append(message)
            self._context_service.update_session(session)
