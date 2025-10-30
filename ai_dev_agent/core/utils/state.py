"""State helpers used to share context within a CLI process."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .constants import MAX_HISTORY_ENTRIES, MAX_METRICS_ENTRIES
from .logger import get_logger

LOGGER = get_logger(__name__)


@dataclass
class PlanSession:
    """Represents a plan execution session that lives for the current process."""

    session_id: str
    goal: str
    status: str
    current_task_id: str | None = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "goal": self.goal,
            "status": self.status,
            "current_task_id": self.current_task_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlanSession:
        return cls(
            session_id=data["session_id"],
            goal=data["goal"],
            status=data["status"],
            current_task_id=data.get("current_task_id"),
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            updated_at=data.get("updated_at", datetime.utcnow().isoformat()),
        )


@dataclass
class InMemoryStateStore:
    """Process-local state container that deliberately avoids persistence."""

    state_file: Path | None = None
    _cache: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.state_file is not None and type(self) is InMemoryStateStore:
            # Provide a debug hint when a state file is configured but ignored.
            LOGGER.debug(
                "Configured state file %s will be ignored by InMemoryStateStore.",
                self.state_file,
            )

    def load(self) -> dict[str, Any]:
        """Return the current state data, creating defaults when empty."""
        if not self._cache:
            self._cache = self._create_default_state()
        return self._cache

    def save(self, data: dict[str, Any]) -> None:
        """Replace the in-memory state after validation."""
        self._validate_state(data)
        self._cache = data
        LOGGER.debug("State stored in memory (not persisted)")

    def update(self, **updates: Any) -> dict[str, Any]:
        """Update state with automatic timestamping."""
        data = self.load().copy()
        data.update(updates)
        data["last_updated"] = datetime.utcnow().isoformat()
        self.save(data)
        return data

    def get_current_session(self) -> PlanSession | None:
        """Get the current active plan session."""
        data = self.load()
        session_data = data.get("current_session")
        if session_data:
            return PlanSession.from_dict(session_data)
        return None

    def start_session(self, goal: str, session_id: str | None = None) -> PlanSession:
        """Start a new plan session."""
        if session_id is None:
            session_id = f"session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        session = PlanSession(session_id=session_id, goal=goal, status="active")

        history = list(self._get_session_history())
        history.append(session.to_dict())

        self.update(
            current_session=session.to_dict(),
            session_history=history,
        )

        LOGGER.info("Started new in-memory session: %s", session_id)
        return session

    def update_session(self, **updates: Any) -> PlanSession | None:
        """Update the current session."""
        session = self.get_current_session()
        if not session:
            LOGGER.warning("No active session to update")
            return None

        for key, value in updates.items():
            if hasattr(session, key):
                setattr(session, key, value)

        session.updated_at = datetime.utcnow().isoformat()

        self.update(current_session=session.to_dict())
        return session

    def end_session(self, status: str = "completed") -> None:
        """End the current session."""
        session = self.get_current_session()
        if session:
            session.status = status
            session.updated_at = datetime.utcnow().isoformat()

            history = []
            replaced = False
            for entry in self._get_session_history():
                if not replaced and entry.get("session_id") == session.session_id:
                    history.append(session.to_dict())
                    replaced = True
                else:
                    history.append(entry)
            if not replaced:
                history.append(session.to_dict())

            self.update(
                current_session=None,
                session_history=history,
            )
            LOGGER.info("Ended in-memory session: %s with status: %s", session.session_id, status)

    def can_resume(self) -> bool:
        """Check if there's a session that can be resumed."""
        session = self.get_current_session()
        return session is not None and session.status in ["active", "paused", "interrupted"]

    def get_resumable_tasks(self) -> list[dict[str, Any]]:
        """Get tasks that can be resumed."""
        data = self.load()
        plan = data.get("last_plan", {})
        tasks = plan.get("tasks", [])

        return [
            task
            for task in tasks
            if task.get("status") in ["pending", "in_progress", "needs_attention"]
        ]

    def _create_default_state(self) -> dict[str, Any]:
        """Create default state structure."""
        now = datetime.utcnow().isoformat()
        return {
            "version": "1.0",
            "created_at": now,
            "last_updated": now,
            "current_session": None,
            "session_history": [],
            "last_plan": None,
            "last_plan_raw": None,
            "command_history": [],
            "metrics": [],
        }

    def _validate_state(self, data: dict[str, Any]) -> None:
        """Validate state structure."""
        if not isinstance(data, dict):
            raise ValueError("State must be a dictionary")

        if "last_updated" not in data:
            data["last_updated"] = datetime.utcnow().isoformat()

    def _get_session_history(self) -> list[dict[str, Any]]:
        """Get session history list."""
        data = self.load()
        return data.get("session_history", [])

    def append_history(self, entry: dict[str, Any], limit: int = MAX_HISTORY_ENTRIES) -> None:
        """Append an entry to the in-memory command history."""
        data = self.load()
        history = list(data.get("command_history", []))
        history.append(entry)
        if limit and len(history) > limit:
            history = history[-limit:]
        data["command_history"] = history
        data["last_updated"] = datetime.utcnow().isoformat()
        self.save(data)

    def record_metric(self, entry: dict[str, Any], limit: int = MAX_METRICS_ENTRIES) -> None:
        """Record a metrics entry while bounding total storage."""
        data = self.load()
        metrics = list(data.get("metrics", []))
        metrics.append(entry)
        if limit and len(metrics) > limit:
            metrics = metrics[-limit:]
        data["metrics"] = metrics
        data["last_updated"] = datetime.utcnow().isoformat()
        self.save(data)


class StateStore(InMemoryStateStore):
    """State store that persists to disk when a state file path is provided."""

    def __post_init__(self) -> None:
        if self.state_file is not None:
            self.state_file = Path(self.state_file)
        super().__post_init__()

    def load(self) -> dict[str, Any]:
        if self.state_file is None:
            return super().load()

        if self._cache:
            return self._cache

        if self.state_file.exists():
            try:
                with self.state_file.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
                self._validate_state(data)
                self._cache = data
                return self._cache
            except Exception as exc:  # pragma: no cover - defensive logging
                LOGGER.warning(
                    "Failed to load state from %s: %s — using defaults.",
                    self.state_file,
                    exc,
                )

        self._cache = self._create_default_state()
        self._write_cache_to_disk()
        return self._cache

    def save(self, data: dict[str, Any]) -> None:
        if self.state_file is None:
            super().save(data)
            return

        self._validate_state(data)
        self._cache = data
        self._write_cache_to_disk()

    def _write_cache_to_disk(self) -> None:
        if self.state_file is None:
            LOGGER.debug("State stored in memory (no persistence path configured)")
            return

        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with self.state_file.open("w", encoding="utf-8") as handle:
                json.dump(self._cache, handle, indent=2, sort_keys=True)
            LOGGER.debug("State written to %s", self.state_file)
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.error("Failed to write state file %s: %s", self.state_file, exc)


__all__ = ["InMemoryStateStore", "PlanSession", "StateStore"]
