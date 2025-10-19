"""Dynamic Instruction Manager - ILWS (Instruction-Level Weight Shaping) Pattern.

This module implements real-time instruction updates based on execution feedback,
allowing the system to adapt its behavior during task execution rather than only
learning after completion.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)


class UpdateConfidence(str, Enum):
    """Confidence level for instruction updates."""
    VERY_LOW = "very_low"      # <0.3 - don't apply
    LOW = "low"                # 0.3-0.5 - apply with caution
    MEDIUM = "medium"          # 0.5-0.7 - apply normally
    HIGH = "high"              # 0.7-0.9 - apply eagerly
    VERY_HIGH = "very_high"    # >0.9 - apply immediately


class UpdateType(str, Enum):
    """Type of instruction update."""
    ADD = "add"                # Add new instruction
    MODIFY = "modify"          # Modify existing instruction
    REMOVE = "remove"          # Remove instruction
    WEIGHT_INCREASE = "weight_increase"  # Increase priority/weight
    WEIGHT_DECREASE = "weight_decrease"  # Decrease priority/weight
    ENABLE = "enable"          # Re-enable disabled instruction
    DISABLE = "disable"        # Temporarily disable instruction


class UpdateSource(str, Enum):
    """Source of the instruction update."""
    EXECUTION_FEEDBACK = "execution_feedback"  # From execution monitoring
    ERROR_RECOVERY = "error_recovery"          # From error handling
    SUCCESS_PATTERN = "success_pattern"        # From successful execution
    USER_FEEDBACK = "user_feedback"            # From explicit user feedback
    AB_TEST = "ab_test"                        # From A/B test results
    AUTOMATIC = "automatic"                    # From automatic analysis


@dataclass
class InstructionUpdate:
    """Represents a proposed or applied instruction update."""

    update_id: str = field(default_factory=lambda: str(uuid4()))
    instruction_id: str = ""                    # Target instruction ID
    update_type: UpdateType = UpdateType.MODIFY
    update_source: UpdateSource = UpdateSource.EXECUTION_FEEDBACK

    # Update content
    old_content: Optional[str] = None
    new_content: Optional[str] = None
    old_priority: Optional[int] = None
    new_priority: Optional[int] = None

    # Confidence and metadata
    confidence: float = 0.5                     # 0.0 - 1.0
    confidence_level: UpdateConfidence = UpdateConfidence.MEDIUM
    reasoning: str = ""                         # Why this update is proposed

    # Execution context
    session_id: Optional[str] = None
    task_context: Optional[str] = None

    # Status tracking
    applied: bool = False
    applied_at: Optional[str] = None
    rolled_back: bool = False
    rolled_back_at: Optional[str] = None

    # Outcome tracking
    success_after_update: Optional[bool] = None
    error_after_update: Optional[str] = None

    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data["update_type"] = self.update_type.value
        data["update_source"] = self.update_source.value
        data["confidence_level"] = self.confidence_level.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> InstructionUpdate:
        """Create from dictionary."""
        if "update_type" in data and isinstance(data["update_type"], str):
            data["update_type"] = UpdateType(data["update_type"])
        if "update_source" in data and isinstance(data["update_source"], str):
            data["update_source"] = UpdateSource(data["update_source"])
        if "confidence_level" in data and isinstance(data["confidence_level"], str):
            data["confidence_level"] = UpdateConfidence(data["confidence_level"])
        return cls(**data)

    @staticmethod
    def confidence_to_level(confidence: float) -> UpdateConfidence:
        """Convert confidence score to confidence level."""
        if confidence < 0.3:
            return UpdateConfidence.VERY_LOW
        elif confidence < 0.5:
            return UpdateConfidence.LOW
        elif confidence < 0.7:
            return UpdateConfidence.MEDIUM
        elif confidence < 0.9:
            return UpdateConfidence.HIGH
        else:
            return UpdateConfidence.VERY_HIGH


@dataclass
class InstructionSnapshot:
    """Snapshot of instruction state for rollback."""

    snapshot_id: str = field(default_factory=lambda: str(uuid4()))
    instruction_id: str = ""
    content: str = ""
    priority: int = 5
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> InstructionSnapshot:
        """Create from dictionary."""
        return cls(**data)


class DynamicInstructionManager:
    """Manages dynamic instruction updates during execution.

    Implements the ILWS (Instruction-Level Weight Shaping) pattern for real-time
    instruction adaptation based on execution feedback.
    """

    DEFAULT_HISTORY_PATH = Path.home() / ".devagent" / "dynamic_instructions" / "update_history.json"
    DEFAULT_SNAPSHOTS_PATH = Path.home() / ".devagent" / "dynamic_instructions" / "snapshots.json"

    def __init__(
        self,
        history_path: Optional[Path] = None,
        snapshots_path: Optional[Path] = None,
        confidence_threshold: float = 0.5,
        auto_rollback_on_error: bool = True,
        max_history: int = 1000
    ):
        """Initialize the dynamic instruction manager.

        Args:
            history_path: Path to update history storage
            snapshots_path: Path to snapshots storage
            confidence_threshold: Minimum confidence to apply updates automatically
            auto_rollback_on_error: Whether to auto-rollback on errors
            max_history: Maximum update history entries to keep
        """
        self.history_path = history_path or self.DEFAULT_HISTORY_PATH
        self.snapshots_path = snapshots_path or self.DEFAULT_SNAPSHOTS_PATH
        self.confidence_threshold = confidence_threshold
        self.auto_rollback_on_error = auto_rollback_on_error
        self.max_history = max_history

        # Thread safety
        self._lock = threading.RLock()

        # In-memory storage
        self._update_history: List[InstructionUpdate] = []
        self._snapshots: Dict[str, List[InstructionSnapshot]] = {}  # instruction_id -> snapshots
        self._pending_updates: Dict[str, InstructionUpdate] = {}    # update_id -> update

        # Active session tracking
        self._active_session_id: Optional[str] = None
        self._session_updates: Dict[str, List[str]] = {}  # session_id -> update_ids

        # Load existing data
        self._load_data()

    def _load_data(self) -> None:
        """Load update history and snapshots from storage."""
        # Load update history
        if self.history_path.exists():
            try:
                with open(self.history_path, "r") as f:
                    data = json.load(f)
                    self._update_history = [
                        InstructionUpdate.from_dict(item) for item in data.get("updates", [])
                    ]
                logger.debug(f"Loaded {len(self._update_history)} update history entries")
            except Exception as e:
                logger.error(f"Failed to load update history: {e}")

        # Load snapshots
        if self.snapshots_path.exists():
            try:
                with open(self.snapshots_path, "r") as f:
                    data = json.load(f)
                    for inst_id, snapshots_data in data.get("snapshots", {}).items():
                        self._snapshots[inst_id] = [
                            InstructionSnapshot.from_dict(s) for s in snapshots_data
                        ]
                logger.debug(f"Loaded snapshots for {len(self._snapshots)} instructions")
            except Exception as e:
                logger.error(f"Failed to load snapshots: {e}")

    def _save_data(self) -> None:
        """Save update history and snapshots to storage."""
        # Save update history
        try:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.history_path, "w") as f:
                json.dump({
                    "updates": [u.to_dict() for u in self._update_history[-self.max_history:]],
                    "saved_at": datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save update history: {e}")

        # Save snapshots
        try:
            self.snapshots_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.snapshots_path, "w") as f:
                json.dump({
                    "snapshots": {
                        inst_id: [s.to_dict() for s in snapshots[-10:]]  # Keep last 10 snapshots
                        for inst_id, snapshots in self._snapshots.items()
                    },
                    "saved_at": datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save snapshots: {e}")

    def start_session(self, session_id: str) -> None:
        """Start tracking updates for a session.

        Args:
            session_id: Unique session identifier
        """
        with self._lock:
            self._active_session_id = session_id
            self._session_updates[session_id] = []
            logger.debug(f"Started tracking session: {session_id}")

    def end_session(self, session_id: str, success: bool = True) -> List[InstructionUpdate]:
        """End a session and return updates made during it.

        Args:
            session_id: Session identifier
            success: Whether the session was successful

        Returns:
            List of updates made during the session
        """
        with self._lock:
            if session_id not in self._session_updates:
                return []

            update_ids = self._session_updates[session_id]
            updates = [
                self._get_update_by_id(uid) for uid in update_ids
                if self._get_update_by_id(uid) is not None
            ]

            # Update success tracking
            for update in updates:
                if update.applied and not update.rolled_back:
                    update.success_after_update = success

            # Auto-rollback on failure if enabled
            if not success and self.auto_rollback_on_error:
                for update in updates:
                    if update.applied and not update.rolled_back:
                        self.rollback_update(update.update_id)
                logger.info(f"Auto-rolled back {len(updates)} updates for failed session")

            if self._active_session_id == session_id:
                self._active_session_id = None

            self._save_data()
            return updates

    def propose_update(
        self,
        instruction_id: str,
        update_type: UpdateType,
        update_source: UpdateSource,
        confidence: float,
        reasoning: str,
        new_content: Optional[str] = None,
        new_priority: Optional[int] = None,
        old_content: Optional[str] = None,
        old_priority: Optional[int] = None,
        task_context: Optional[str] = None
    ) -> InstructionUpdate:
        """Propose an instruction update.

        Args:
            instruction_id: ID of instruction to update
            update_type: Type of update
            update_source: Source of update
            confidence: Confidence score (0-1)
            reasoning: Why this update is proposed
            new_content: New instruction content (if applicable)
            new_priority: New priority (if applicable)
            old_content: Current instruction content
            old_priority: Current priority
            task_context: Task context for the update

        Returns:
            Created InstructionUpdate object
        """
        with self._lock:
            confidence_level = InstructionUpdate.confidence_to_level(confidence)

            update = InstructionUpdate(
                instruction_id=instruction_id,
                update_type=update_type,
                update_source=update_source,
                old_content=old_content,
                new_content=new_content,
                old_priority=old_priority,
                new_priority=new_priority,
                confidence=confidence,
                confidence_level=confidence_level,
                reasoning=reasoning,
                session_id=self._active_session_id,
                task_context=task_context
            )

            # Add to pending updates
            self._pending_updates[update.update_id] = update

            # Auto-apply if confidence is above threshold
            if confidence >= self.confidence_threshold:
                logger.info(f"Auto-applying update {update.update_id} (confidence: {confidence:.2f})")
                # Note: Actual application would happen in apply_update()
            else:
                logger.debug(f"Update {update.update_id} pending approval (confidence: {confidence:.2f})")

            return update

    def apply_update(
        self,
        update_id: str,
        snapshot_before: Optional[InstructionSnapshot] = None
    ) -> bool:
        """Apply a pending update.

        Args:
            update_id: ID of update to apply
            snapshot_before: Snapshot of instruction before update (for rollback)

        Returns:
            True if applied successfully
        """
        with self._lock:
            if update_id not in self._pending_updates:
                logger.warning(f"Update {update_id} not found in pending updates")
                return False

            update = self._pending_updates[update_id]

            # Create snapshot if provided
            if snapshot_before:
                if update.instruction_id not in self._snapshots:
                    self._snapshots[update.instruction_id] = []
                self._snapshots[update.instruction_id].append(snapshot_before)

            # Mark as applied
            update.applied = True
            update.applied_at = datetime.now().isoformat()

            # Move to history
            self._update_history.append(update)
            del self._pending_updates[update_id]

            # Track in session
            if self._active_session_id:
                if self._active_session_id not in self._session_updates:
                    self._session_updates[self._active_session_id] = []
                self._session_updates[self._active_session_id].append(update_id)

            logger.info(f"Applied update {update_id}: {update.update_type.value} to {update.instruction_id}")
            self._save_data()
            return True

    def rollback_update(self, update_id: str) -> bool:
        """Rollback a previously applied update.

        Args:
            update_id: ID of update to rollback

        Returns:
            True if rolled back successfully
        """
        with self._lock:
            # Find the update in history
            update = self._get_update_by_id(update_id)
            if not update:
                logger.warning(f"Update {update_id} not found")
                return False

            if not update.applied:
                logger.warning(f"Update {update_id} was not applied")
                return False

            if update.rolled_back:
                logger.warning(f"Update {update_id} already rolled back")
                return False

            # Mark as rolled back
            update.rolled_back = True
            update.rolled_back_at = datetime.now().isoformat()

            logger.info(f"Rolled back update {update_id}")
            self._save_data()
            return True

    def get_snapshot(self, instruction_id: str, snapshot_id: Optional[str] = None) -> Optional[InstructionSnapshot]:
        """Get a snapshot for an instruction.

        Args:
            instruction_id: Instruction ID
            snapshot_id: Specific snapshot ID (if None, returns latest)

        Returns:
            InstructionSnapshot or None
        """
        with self._lock:
            if instruction_id not in self._snapshots:
                return None

            snapshots = self._snapshots[instruction_id]
            if not snapshots:
                return None

            if snapshot_id:
                for snapshot in snapshots:
                    if snapshot.snapshot_id == snapshot_id:
                        return snapshot
                return None
            else:
                return snapshots[-1]  # Return latest

    def get_update_history(
        self,
        instruction_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 100
    ) -> List[InstructionUpdate]:
        """Get update history with optional filters.

        Args:
            instruction_id: Filter by instruction ID
            session_id: Filter by session ID
            limit: Maximum number of updates to return

        Returns:
            List of InstructionUpdate objects
        """
        with self._lock:
            updates = list(self._update_history)

            if instruction_id:
                updates = [u for u in updates if u.instruction_id == instruction_id]

            if session_id:
                updates = [u for u in updates if u.session_id == session_id]

            # Return most recent first
            return updates[-limit:][::-1]

    def get_pending_updates(
        self,
        min_confidence: Optional[float] = None
    ) -> List[InstructionUpdate]:
        """Get pending updates.

        Args:
            min_confidence: Filter by minimum confidence

        Returns:
            List of pending updates
        """
        with self._lock:
            updates = list(self._pending_updates.values())

            if min_confidence is not None:
                updates = [u for u in updates if u.confidence >= min_confidence]

            # Sort by confidence descending
            return sorted(updates, key=lambda u: u.confidence, reverse=True)

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about dynamic updates.

        Returns:
            Dictionary with statistics
        """
        with self._lock:
            total_updates = len(self._update_history)
            applied_updates = sum(1 for u in self._update_history if u.applied)
            rolled_back = sum(1 for u in self._update_history if u.rolled_back)
            successful = sum(1 for u in self._update_history if u.success_after_update is True)
            failed = sum(1 for u in self._update_history if u.success_after_update is False)

            by_type = {}
            for update in self._update_history:
                update_type = update.update_type.value
                by_type[update_type] = by_type.get(update_type, 0) + 1

            by_source = {}
            for update in self._update_history:
                source = update.update_source.value
                by_source[source] = by_source.get(source, 0) + 1

            return {
                "total_updates": total_updates,
                "applied_updates": applied_updates,
                "rolled_back": rolled_back,
                "successful_updates": successful,
                "failed_updates": failed,
                "pending_updates": len(self._pending_updates),
                "snapshots_stored": sum(len(s) for s in self._snapshots.values()),
                "by_type": by_type,
                "by_source": by_source,
                "success_rate": successful / applied_updates if applied_updates > 0 else 0.0,
                "rollback_rate": rolled_back / applied_updates if applied_updates > 0 else 0.0
            }

    def _get_update_by_id(self, update_id: str) -> Optional[InstructionUpdate]:
        """Get an update by ID from history or pending.

        Args:
            update_id: Update ID

        Returns:
            InstructionUpdate or None
        """
        # Check pending first
        if update_id in self._pending_updates:
            return self._pending_updates[update_id]

        # Check history
        for update in self._update_history:
            if update.update_id == update_id:
                return update

        return None

    def save_state(self) -> None:
        """Save current state to disk (public API).

        This method exposes the internal _save_data() for external use,
        such as saving state when the CLI session ends.
        """
        with self._lock:
            self._save_data()
            logger.debug("Dynamic instruction state saved to disk")

    def load_state(self) -> None:
        """Reload state from disk (public API).

        This method allows reloading the state after it has been saved,
        useful for testing or manual state management.
        """
        with self._lock:
            self._load_data()
            logger.debug("Dynamic instruction state reloaded from disk")
