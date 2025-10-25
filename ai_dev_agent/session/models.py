"""Core data structures for DevAgent session management."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from threading import RLock
from typing import Any, Dict, List

from ai_dev_agent.providers.llm.base import Message


@dataclass
class Session:
    """Represents a conversational session scoped to a workspace invocation."""

    id: str
    system_messages: List[Message] = field(default_factory=list)
    history: List[Message] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    lock: RLock = field(default_factory=RLock)
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)

    def compose(self) -> List[Message]:
        """Return the flattened message list for LLM consumption."""
        return [*self.system_messages, *self.history]

