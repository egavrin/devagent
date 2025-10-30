"""Dynamic Instruction Manager - ILWS (Instruction-Level Weight Shaping) Pattern.

This module implements real-time instruction updates based on execution feedback,
allowing the system to adapt its behavior during task execution rather than only
learning after completion.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


class UpdateConfidence(str, Enum):
    """Confidence level for instruction updates."""

    VERY_LOW = "very_low"  # <0.3 - don't apply
    LOW = "low"  # 0.3-0.5 - apply with caution
    MEDIUM = "medium"  # 0.5-0.7 - apply normally
    HIGH = "high"  # 0.7-0.9 - apply eagerly
    VERY_HIGH = "very_high"  # >0.9 - apply immediately


class UpdateType(str, Enum):
    """Type of instruction update."""

    ADD = "add"  # Add new instruction
    MODIFY = "modify"  # Modify existing instruction
    REMOVE = "remove"  # Remove instruction
    WEIGHT_INCREASE = "weight_increase"  # Increase priority/weight
    WEIGHT_DECREASE = "weight_decrease"  # Decrease priority/weight
    ENABLE = "enable"  # Re-enable disabled instruction
    DISABLE = "disable"  # Temporarily disable instruction


class UpdateSource(str, Enum):
    """Source of the instruction update."""

    EXECUTION_FEEDBACK = "execution_feedback"  # From execution monitoring
    ERROR_RECOVERY = "error_recovery"  # From error handling
    SUCCESS_PATTERN = "success_pattern"  # From successful execution
    USER_FEEDBACK = "user_feedback"  # From explicit user feedback
    AB_TEST = "ab_test"  # From A/B test results
    AUTOMATIC = "automatic"  # From automatic analysis


@dataclass
class InstructionUpdate:
    """Represents a proposed or applied instruction update."""

    update_id: str = field(default_factory=lambda: str(uuid4()))
    instruction_id: str = ""  # Target instruction ID
    update_type: UpdateType = UpdateType.MODIFY
    update_source: UpdateSource = UpdateSource.EXECUTION_FEEDBACK

    # Update content
    old_content: str | None = None
    new_content: str | None = None
    old_priority: int | None = None
    new_priority: int | None = None

    # Confidence and metadata
    confidence: float = 0.5  # 0.0 - 1.0
    confidence_level: UpdateConfidence = UpdateConfidence.MEDIUM
    reasoning: str = ""  # Why this update is proposed

    # Execution context
    session_id: str | None = None
    task_context: str | None = None

    # Status tracking
    applied: bool = False
    applied_at: str | None = None
    rolled_back: bool = False
    rolled_back_at: str | None = None

    # Outcome tracking
    success_after_update: bool | None = None
    error_after_update: str | None = None

    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data["update_type"] = self.update_type.value
        data["update_source"] = self.update_source.value
        data["confidence_level"] = self.confidence_level.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InstructionUpdate:
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
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InstructionSnapshot:
        """Create from dictionary."""
        return cls(**data)


class DynamicInstructionManager:
    """Manages dynamic instruction updates during execution.

    Implements the ILWS (Instruction-Level Weight Shaping) pattern for real-time
    instruction adaptation based on execution feedback.
    """

    DEFAULT_HISTORY_PATH = (
        Path.home() / ".devagent" / "dynamic_instructions" / "update_history.json"
    )
    DEFAULT_SNAPSHOTS_PATH = Path.home() / ".devagent" / "dynamic_instructions" / "snapshots.json"

    def __init__(
        self,
        history_path: Path | None = None,
        snapshots_path: Path | None = None,
        confidence_threshold: float = 0.5,
        auto_rollback_on_error: bool = True,
        max_history: int = 1000,
        analysis_interval: int = 15,
        auto_apply_threshold: float = 0.8,
        proposal_min_queries: int = 10,
        max_auto_apply_per_cycle: int = 3,
    ):
        """Initialize the dynamic instruction manager.

        Args:
            history_path: Path to update history storage
            snapshots_path: Path to snapshots storage
            confidence_threshold: Minimum confidence to apply updates automatically
            auto_rollback_on_error: Whether to auto-rollback on errors
            max_history: Maximum update history entries to keep
            analysis_interval: Number of queries between pattern analysis
            auto_apply_threshold: Confidence threshold for auto-applying proposals
            proposal_min_queries: Minimum queries before proposing instructions
            max_auto_apply_per_cycle: Maximum auto-applied instructions per analysis
        """
        self.history_path = history_path or self.DEFAULT_HISTORY_PATH
        self.snapshots_path = snapshots_path or self.DEFAULT_SNAPSHOTS_PATH
        self.confidence_threshold = confidence_threshold
        self.auto_rollback_on_error = auto_rollback_on_error
        self.max_history = max_history
        self.analysis_interval = analysis_interval
        self.auto_apply_threshold = auto_apply_threshold
        self.proposal_min_queries = proposal_min_queries
        self.max_auto_apply_per_cycle = max_auto_apply_per_cycle

        # Thread safety
        self._lock = threading.RLock()

        # In-memory storage
        self._update_history: list[InstructionUpdate] = []
        self._snapshots: dict[str, list[InstructionSnapshot]] = {}  # instruction_id -> snapshots
        self._pending_updates: dict[str, InstructionUpdate] = {}  # update_id -> update

        # Active session tracking
        self._active_session_id: str | None = None
        self._session_updates: dict[str, list[str]] = {}  # session_id -> update_ids

        # Pattern tracking for automatic proposals
        from .pattern_tracker import PatternTracker

        patterns_path = self.history_path.parent / "patterns.json"
        self._pattern_tracker = PatternTracker(storage_path=patterns_path)

        # Rollback tracking for safety
        self._recent_rollbacks: list[str] = []  # Recent rollback timestamps

        # Load existing data
        self._load_data()

    def _load_data(self) -> None:
        """Load update history and snapshots from storage."""
        # Load update history
        if self.history_path.exists():
            try:
                with self.history_path.open() as f:
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
                with self.snapshots_path.open() as f:
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
            with self.history_path.open("w") as f:
                json.dump(
                    {
                        "updates": [u.to_dict() for u in self._update_history[-self.max_history :]],
                        "saved_at": datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            logger.error(f"Failed to save update history: {e}")

        # Save snapshots
        try:
            self.snapshots_path.parent.mkdir(parents=True, exist_ok=True)
            with self.snapshots_path.open("w") as f:
                json.dump(
                    {
                        "snapshots": {
                            inst_id: [
                                s.to_dict() for s in snapshots[-10:]
                            ]  # Keep last 10 snapshots
                            for inst_id, snapshots in self._snapshots.items()
                        },
                        "saved_at": datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                )
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

    def end_session(self, session_id: str, success: bool = True) -> list[InstructionUpdate]:
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
                self._get_update_by_id(uid)
                for uid in update_ids
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
        new_content: str | None = None,
        new_priority: int | None = None,
        old_content: str | None = None,
        old_priority: int | None = None,
        task_context: str | None = None,
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
                task_context=task_context,
            )

            # Add to pending updates
            self._pending_updates[update.update_id] = update

            # Auto-apply if confidence is above threshold
            if confidence >= self.confidence_threshold:
                logger.info(
                    f"Auto-applying update {update.update_id} (confidence: {confidence:.2f})"
                )
                # Note: Actual application would happen in apply_update()
            else:
                logger.debug(
                    f"Update {update.update_id} pending approval (confidence: {confidence:.2f})"
                )

            return update

    def apply_update(
        self, update_id: str, snapshot_before: InstructionSnapshot | None = None
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

            logger.info(
                f"Applied update {update_id}: {update.update_type.value} to {update.instruction_id}"
            )
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

            # Record rollback for safety tracking
            self._record_rollback()

            logger.info(f"Rolled back update {update_id}")
            self._save_data()
            return True

    def get_snapshot(
        self, instruction_id: str, snapshot_id: str | None = None
    ) -> InstructionSnapshot | None:
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
        instruction_id: str | None = None,
        session_id: str | None = None,
        limit: int = 100,
    ) -> list[InstructionUpdate]:
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

    def get_pending_updates(self, min_confidence: float | None = None) -> list[InstructionUpdate]:
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

    def get_statistics(self) -> dict[str, Any]:
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
                "rollback_rate": rolled_back / applied_updates if applied_updates > 0 else 0.0,
            }

    def _get_update_by_id(self, update_id: str) -> InstructionUpdate | None:
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

    # =============================================================================
    # Pattern Tracking & Automatic Proposals
    # =============================================================================

    def record_query_outcome(
        self,
        session_id: str,
        success: bool,
        tools_used: list[str],
        task_type: str = "general",
        error_type: str | None = None,
        duration_seconds: float | None = None,
    ) -> None:
        """Record a query outcome for pattern analysis.

        Args:
            session_id: Unique session identifier
            success: Whether query was successful
            tools_used: List of tools used in order
            task_type: Type of task (e.g., "debugging", "feature")
            error_type: Type of error if failed
            duration_seconds: Query duration in seconds
        """
        self._pattern_tracker.record_query(
            session_id=session_id,
            success=success,
            tools_used=tools_used,
            task_type=task_type,
            error_type=error_type,
            duration_seconds=duration_seconds,
        )

    def should_analyze_patterns(self) -> bool:
        """Check if pattern analysis should be triggered.

        Returns:
            True if analysis interval is reached and enough data exists
        """
        query_count = self._pattern_tracker.get_query_count()

        # Check if analysis interval is reached
        if query_count > 0 and query_count % self.analysis_interval == 0:
            # Check if enough data exists
            return self._pattern_tracker.has_significant_patterns(
                min_queries=self.proposal_min_queries
            )

        return False

    def get_pattern_statistics(self) -> dict[str, Any]:
        """Get statistics about recorded patterns.

        Returns:
            Dictionary of pattern statistics
        """
        return self._pattern_tracker.get_statistics()

    def _check_rollback_safety(self) -> bool:
        """Check if it's safe to auto-apply instructions based on rollback history.

        Returns:
            True if safe to auto-apply (< 3 rollbacks in last 50 queries)
        """
        # Keep only recent rollbacks (last 50 queries worth)
        query_count = self._pattern_tracker.get_query_count()
        max(0, query_count - 50)

        # Count recent rollbacks
        recent_count = len(
            [ts for ts in self._recent_rollbacks if query_count - int(ts.split("_")[-1]) <= 50]
        )

        # Safe if < 3 recent rollbacks
        return recent_count < 3

    def _record_rollback(self) -> None:
        """Record a rollback event for safety tracking."""
        query_count = self._pattern_tracker.get_query_count()
        self._recent_rollbacks.append(f"{datetime.now().isoformat()}_{query_count}")

        # Keep only last 100 rollback records
        if len(self._recent_rollbacks) > 100:
            self._recent_rollbacks = self._recent_rollbacks[-100:]

    def analyze_and_propose_instructions(
        self,
        playbook_manager: Any,  # PlaybookManager instance
        settings: Any = None,  # Settings instance
    ) -> dict[str, Any]:
        """Analyze patterns and generate/apply instruction proposals.

        Args:
            playbook_manager: PlaybookManager instance to apply instructions to
            settings: Settings instance for LLM configuration

        Returns:
            Dictionary with analysis results:
            - patterns_detected: Number of patterns found
            - proposals_generated: Number of proposals created
            - auto_applied: Number of proposals auto-applied
            - pending_review: Number of proposals pending review
            - proposals: List of proposal details
        """
        from .proposal_generator import ProposalGenerator

        logger.info("Analyzing query patterns for instruction proposals")

        # Detect patterns
        patterns = self._pattern_tracker.detect_patterns(min_sample_size=5, min_success_rate=0.7)

        if not patterns:
            logger.info("No significant patterns detected")
            return {
                "patterns_detected": 0,
                "proposals_generated": 0,
                "auto_applied": 0,
                "pending_review": 0,
                "proposals": [],
            }

        logger.info(f"Detected {len(patterns)} significant patterns")

        # Generate proposals using LLM
        generator = ProposalGenerator(settings=settings)
        proposals = generator.generate_proposals(patterns, max_proposals=5)

        if not proposals:
            logger.info("No proposals generated from patterns")
            return {
                "patterns_detected": len(patterns),
                "proposals_generated": 0,
                "auto_applied": 0,
                "pending_review": 0,
                "proposals": [],
            }

        # Validate proposal quality
        valid_proposals = [p for p in proposals if generator.validate_proposal_quality(p)]

        logger.info(f"Generated {len(valid_proposals)} valid proposals")

        # Check safety (rollback history)
        safety_ok = self._check_rollback_safety()

        # Process proposals
        auto_applied = 0
        pending_review = 0
        proposal_details = []

        for proposal in valid_proposals[: self.max_auto_apply_per_cycle]:
            detail = {
                "type": proposal.update_type.value,
                "content": proposal.new_content,
                "reasoning": proposal.reasoning,
                "confidence": proposal.confidence,
                "action": "none",
            }

            # Auto-apply if high confidence and safe
            if (
                proposal.confidence >= self.auto_apply_threshold
                and safety_ok
                and auto_applied < self.max_auto_apply_per_cycle
            ):
                # Add instruction to playbook
                try:
                    from ..playbook.manager import PlaybookInstruction

                    instruction = PlaybookInstruction(
                        content=proposal.new_content or "",
                        category="auto_generated",
                        priority=5,  # Medium priority
                        tags=["auto_proposal"],
                        enabled=True,
                    )

                    playbook_manager.add_instruction(instruction)

                    # Mark as applied
                    proposal.instruction_id = instruction.instruction_id
                    proposal.applied = True
                    proposal.applied_at = datetime.now().isoformat()

                    # Add to history
                    self._update_history.append(proposal)

                    auto_applied += 1
                    detail["action"] = "auto_applied"

                    logger.info(
                        f"Auto-applied instruction (confidence={proposal.confidence:.2f}): "
                        f"{proposal.new_content[:60]}..."
                    )

                except Exception as e:
                    logger.error(f"Failed to auto-apply instruction: {e}")
                    detail["action"] = "failed"

            elif proposal.confidence >= 0.5:
                # Queue for review
                self._pending_updates[proposal.update_id] = proposal
                pending_review += 1
                detail["action"] = "pending_review"

                logger.info(
                    f"Queued instruction for review (confidence={proposal.confidence:.2f}): "
                    f"{proposal.new_content[:60]}..."
                )
            else:
                # Confidence too low
                detail["action"] = "discarded"
                logger.debug(f"Discarded low-confidence proposal ({proposal.confidence:.2f})")

            proposal_details.append(detail)

        # Save updated state
        self._save_data()

        result = {
            "patterns_detected": len(patterns),
            "proposals_generated": len(valid_proposals),
            "auto_applied": auto_applied,
            "pending_review": pending_review,
            "proposals": proposal_details,
        }

        logger.info(
            f"Proposal analysis complete: {auto_applied} auto-applied, "
            f"{pending_review} pending review"
        )

        return result
