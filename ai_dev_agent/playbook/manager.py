"""Playbook Manager - Maintains evolving instructions for DevAgent.

Based on ACE (Agentic Context Engineering) pattern:
- Incremental updates (deltas) instead of full rewrites
- Structured organization by category
- Effectiveness tracking per instruction
- Version control for rollback
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

logger = logging.getLogger(__name__)


class InstructionCategory(str, Enum):
    """Categories for organizing instructions."""

    DEBUGGING = "debugging"
    TESTING = "testing"
    REFACTORING = "refactoring"
    OPTIMIZATION = "optimization"
    SECURITY = "security"
    DOCUMENTATION = "documentation"
    CODE_REVIEW = "code_review"
    ERROR_HANDLING = "error_handling"
    API_DESIGN = "api_design"
    DATABASE = "database"
    GENERAL = "general"


@dataclass
class Instruction:
    """A single instruction in the playbook."""

    instruction_id: str = field(default_factory=lambda: str(uuid4()))
    category: InstructionCategory = InstructionCategory.GENERAL
    content: str = ""  # The actual instruction text
    priority: int = 5  # 1 (low) to 10 (high)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    usage_count: int = 0
    success_count: int = 0  # Times this instruction led to success
    effectiveness_score: float = 0.0  # success_count / usage_count
    tags: Set[str] = field(default_factory=set)
    examples: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data["category"] = self.category.value
        data["tags"] = list(self.tags)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Instruction:
        """Create from dictionary."""
        # Convert category string to enum
        if "category" in data and isinstance(data["category"], str):
            data["category"] = InstructionCategory(data["category"])

        # Convert tags list to set
        if "tags" in data and isinstance(data["tags"], list):
            data["tags"] = set(data["tags"])

        return cls(**data)

    def update_effectiveness(self, success: bool) -> None:
        """Update effectiveness based on usage outcome."""
        self.usage_count += 1
        if success:
            self.success_count += 1

        # Calculate effectiveness score
        if self.usage_count > 0:
            self.effectiveness_score = self.success_count / self.usage_count

        self.updated_at = datetime.now().isoformat()


class PlaybookManager:
    """Manages the evolving playbook of instructions."""

    DEFAULT_PLAYBOOK_PATH = Path.home() / ".devagent" / "playbook" / "instructions.json"
    BACKUP_SUFFIX = ".backup"

    def __init__(
        self,
        playbook_path: Optional[Path] = None,
        max_instructions: int = 50,
        auto_save: bool = True
    ):
        """Initialize the playbook manager.

        Args:
            playbook_path: Path to playbook storage file
            max_instructions: Maximum instructions to maintain
            auto_save: Whether to auto-save after modifications
        """
        self.playbook_path = playbook_path or self.DEFAULT_PLAYBOOK_PATH
        self.max_instructions = max_instructions
        self.auto_save = auto_save

        # Thread safety
        self._lock = threading.RLock()

        # In-memory storage
        self._instructions: Dict[str, Instruction] = {}
        self._instructions_by_category: Dict[InstructionCategory, List[str]] = {
            category: [] for category in InstructionCategory
        }

        # Version tracking
        self._version: int = 1
        self._change_history: List[Dict[str, Any]] = []

        # Load existing playbook
        self._load_playbook()

    def _load_playbook(self) -> None:
        """Load playbook from persistent storage."""
        if not self.playbook_path.exists():
            # Create directory if needed
            self.playbook_path.parent.mkdir(parents=True, exist_ok=True)
            # Initialize with default instructions
            self._initialize_default_playbook()
            logger.debug(f"Created new playbook at {self.playbook_path}")
            return

        try:
            with open(self.playbook_path, "r") as f:
                data = json.load(f)

            # Load version
            self._version = data.get("version", 1)

            # Load instructions
            for instruction_data in data.get("instructions", []):
                instruction = Instruction.from_dict(instruction_data)
                self._instructions[instruction.instruction_id] = instruction
                self._instructions_by_category[instruction.category].append(
                    instruction.instruction_id
                )

            # Load change history
            self._change_history = data.get("change_history", [])

            logger.debug(f"Loaded {len(self._instructions)} instructions from {self.playbook_path}")

        except Exception as e:
            logger.error(f"Failed to load playbook: {e}")
            # Initialize with defaults
            self._initialize_default_playbook()

    def _initialize_default_playbook(self) -> None:
        """Initialize with default instructions."""
        default_instructions = [
            # Debugging
            Instruction(
                category=InstructionCategory.DEBUGGING,
                content="Always check logs and error messages first before making assumptions",
                priority=10,
                tags={"debugging", "errors", "logs"}
            ),
            Instruction(
                category=InstructionCategory.DEBUGGING,
                content="Reproduce the issue reliably before attempting a fix",
                priority=9,
                tags={"debugging", "reproduction"}
            ),
            Instruction(
                category=InstructionCategory.DEBUGGING,
                content="Use a debugger or print statements to understand program state",
                priority=8,
                tags={"debugging", "inspection"}
            ),

            # Testing
            Instruction(
                category=InstructionCategory.TESTING,
                content="Write tests before implementing fixes (TDD approach)",
                priority=10,
                tags={"testing", "tdd"}
            ),
            Instruction(
                category=InstructionCategory.TESTING,
                content="Always run existing tests after making changes",
                priority=10,
                tags={"testing", "regression"}
            ),
            Instruction(
                category=InstructionCategory.TESTING,
                content="Test edge cases: null values, empty inputs, boundary conditions",
                priority=8,
                tags={"testing", "edge-cases"}
            ),

            # Code Quality
            Instruction(
                category=InstructionCategory.REFACTORING,
                content="Keep functions small and focused on a single responsibility",
                priority=7,
                tags={"refactoring", "clean-code"}
            ),
            Instruction(
                category=InstructionCategory.REFACTORING,
                content="Extract duplicated code into reusable functions",
                priority=7,
                tags={"refactoring", "dry"}
            ),

            # Error Handling
            Instruction(
                category=InstructionCategory.ERROR_HANDLING,
                content="Always validate user inputs and handle potential errors",
                priority=9,
                tags={"error-handling", "validation"}
            ),
            Instruction(
                category=InstructionCategory.ERROR_HANDLING,
                content="Provide meaningful error messages that help diagnose issues",
                priority=8,
                tags={"error-handling", "messages"}
            ),

            # Security
            Instruction(
                category=InstructionCategory.SECURITY,
                content="Never log or store sensitive data like passwords or API keys",
                priority=10,
                tags={"security", "credentials"}
            ),
            Instruction(
                category=InstructionCategory.SECURITY,
                content="Validate and sanitize all user inputs to prevent injection attacks",
                priority=10,
                tags={"security", "validation", "injection"}
            ),

            # General
            Instruction(
                category=InstructionCategory.GENERAL,
                content="Read existing code and understand the patterns before making changes",
                priority=8,
                tags={"general", "code-reading"}
            ),
            Instruction(
                category=InstructionCategory.GENERAL,
                content="Commit changes incrementally with clear messages",
                priority=7,
                tags={"general", "git"}
            ),
        ]

        for instruction in default_instructions:
            self.add_instruction(instruction)

        logger.debug(f"Initialized playbook with {len(default_instructions)} default instructions")

    def save_playbook(self) -> None:
        """Save playbook to persistent storage."""
        with self._lock:
            # Create backup if requested
            if self.playbook_path.exists():
                backup_path = self.playbook_path.with_suffix(self.BACKUP_SUFFIX)
                try:
                    import shutil
                    shutil.copy2(self.playbook_path, backup_path)
                except Exception as e:
                    logger.warning(f"Failed to create backup: {e}")

            # Prepare data
            data = {
                "version": self._version,
                "updated_at": datetime.now().isoformat(),
                "instructions": [
                    instruction.to_dict()
                    for instruction in self._instructions.values()
                ],
                "change_history": self._change_history[-100:],  # Keep last 100 changes
                "metadata": {
                    "total_instructions": len(self._instructions),
                    "instructions_by_category": {
                        category.value: len(ids)
                        for category, ids in self._instructions_by_category.items()
                    }
                }
            }

            # Save to file
            try:
                temp_path = self.playbook_path.with_suffix(".tmp")
                with open(temp_path, "w") as f:
                    json.dump(data, f, indent=2, default=str)

                # Atomic rename
                temp_path.replace(self.playbook_path)
                logger.debug(f"Saved {len(self._instructions)} instructions to {self.playbook_path}")

            except Exception as e:
                logger.error(f"Failed to save playbook: {e}")
                if temp_path.exists():
                    temp_path.unlink()

    def add_instruction(
        self,
        instruction: Instruction,
        source: str = "manual"
    ) -> str:
        """Add a new instruction to the playbook.

        Args:
            instruction: Instruction to add
            source: Source of the instruction (manual, learned, suggested)

        Returns:
            Instruction ID
        """
        with self._lock:
            # Check if we're at max capacity
            if len(self._instructions) >= self.max_instructions:
                # Remove least effective instruction in same category
                self._prune_category(instruction.category, count=1)

            # Add to storage
            self._instructions[instruction.instruction_id] = instruction
            self._instructions_by_category[instruction.category].append(
                instruction.instruction_id
            )

            # Record change
            self._record_change("add", instruction, source)

            # Save if auto-save enabled
            if self.auto_save:
                self.save_playbook()

            logger.info(f"Added instruction: {instruction.content[:50]}...")
            return instruction.instruction_id

    def update_instruction(
        self,
        instruction_id: str,
        content: Optional[str] = None,
        priority: Optional[int] = None,
        tags: Optional[Set[str]] = None,
        examples: Optional[List[str]] = None
    ) -> bool:
        """Update an existing instruction.

        Args:
            instruction_id: ID of instruction to update
            content: New content (if provided)
            priority: New priority (if provided)
            tags: New tags (if provided)
            examples: New examples (if provided)

        Returns:
            True if updated successfully
        """
        with self._lock:
            instruction = self._instructions.get(instruction_id)
            if not instruction:
                return False

            # Track what changed
            changes = {}

            if content is not None and content != instruction.content:
                changes["content"] = {"old": instruction.content, "new": content}
                instruction.content = content

            if priority is not None and priority != instruction.priority:
                changes["priority"] = {"old": instruction.priority, "new": priority}
                instruction.priority = priority

            if tags is not None:
                changes["tags"] = {"old": list(instruction.tags), "new": list(tags)}
                instruction.tags = tags

            if examples is not None:
                instruction.examples = examples

            if changes:
                instruction.updated_at = datetime.now().isoformat()
                self._record_change("update", instruction, "manual", changes)

                if self.auto_save:
                    self.save_playbook()

                logger.info(f"Updated instruction: {instruction_id}")
                return True

            return False

    def remove_instruction(self, instruction_id: str) -> bool:
        """Remove an instruction from the playbook.

        Args:
            instruction_id: ID of instruction to remove

        Returns:
            True if removed successfully
        """
        with self._lock:
            instruction = self._instructions.get(instruction_id)
            if not instruction:
                return False

            # Remove from storage
            del self._instructions[instruction_id]
            self._instructions_by_category[instruction.category].remove(instruction_id)

            # Record change
            self._record_change("remove", instruction, "manual")

            if self.auto_save:
                self.save_playbook()

            logger.info(f"Removed instruction: {instruction_id}")
            return True

    def get_instruction(self, instruction_id: str) -> Optional[Instruction]:
        """Get a specific instruction by ID.

        Args:
            instruction_id: Instruction identifier

        Returns:
            Instruction if found
        """
        with self._lock:
            return self._instructions.get(instruction_id)

    def get_instructions_by_category(
        self,
        category: InstructionCategory,
        min_priority: int = 1,
        limit: Optional[int] = None
    ) -> List[Instruction]:
        """Get instructions for a specific category.

        Args:
            category: Category to retrieve
            min_priority: Minimum priority threshold
            limit: Maximum number to return

        Returns:
            List of instructions
        """
        with self._lock:
            instruction_ids = self._instructions_by_category.get(category, [])
            instructions = [
                self._instructions[iid]
                for iid in instruction_ids
                if self._instructions[iid].priority >= min_priority
            ]

            # Sort by priority (descending) then effectiveness
            instructions.sort(
                key=lambda i: (i.priority, i.effectiveness_score),
                reverse=True
            )

            if limit:
                instructions = instructions[:limit]

            return instructions

    def get_all_instructions(
        self,
        min_priority: int = 1,
        min_effectiveness: float = 0.0
    ) -> List[Instruction]:
        """Get all instructions meeting criteria.

        Args:
            min_priority: Minimum priority threshold
            min_effectiveness: Minimum effectiveness threshold

        Returns:
            List of instructions
        """
        with self._lock:
            instructions = [
                instruction
                for instruction in self._instructions.values()
                if instruction.priority >= min_priority
                and instruction.effectiveness_score >= min_effectiveness
            ]

            # Sort by priority and effectiveness
            instructions.sort(
                key=lambda i: (i.priority, i.effectiveness_score),
                reverse=True
            )

            return instructions

    def get_instructions_by_tags(
        self,
        tags: Set[str],
        match_all: bool = False
    ) -> List[Instruction]:
        """Get instructions matching tags.

        Args:
            tags: Tags to match
            match_all: If True, require all tags; if False, match any

        Returns:
            List of matching instructions
        """
        with self._lock:
            instructions = []

            for instruction in self._instructions.values():
                if match_all:
                    if tags.issubset(instruction.tags):
                        instructions.append(instruction)
                else:
                    if tags & instruction.tags:  # Any overlap
                        instructions.append(instruction)

            # Sort by relevance (number of matching tags) then priority
            instructions.sort(
                key=lambda i: (len(tags & i.tags), i.priority),
                reverse=True
            )

            return instructions

    def format_for_context(
        self,
        categories: Optional[List[InstructionCategory]] = None,
        max_instructions: int = 10,
        min_priority: int = 5
    ) -> str:
        """Format instructions for injection into LLM context.

        Args:
            categories: Categories to include (all if None)
            max_instructions: Maximum instructions to include
            min_priority: Minimum priority threshold

        Returns:
            Formatted instruction text
        """
        with self._lock:
            # Get relevant instructions
            if categories:
                instructions = []
                for category in categories:
                    instructions.extend(
                        self.get_instructions_by_category(category, min_priority)
                    )
            else:
                instructions = self.get_all_instructions(min_priority)

            # Deduplicate and sort
            seen = set()
            unique_instructions = []
            for instruction in instructions:
                if instruction.instruction_id not in seen:
                    seen.add(instruction.instruction_id)
                    unique_instructions.append(instruction)

            # Limit count
            unique_instructions = unique_instructions[:max_instructions]

            if not unique_instructions:
                return ""

            # Format as bullet list organized by category
            lines = ["[DevAgent Playbook - Best Practices]"]

            current_category = None
            for instruction in unique_instructions:
                if instruction.category != current_category:
                    current_category = instruction.category
                    lines.append(f"\n{current_category.value.replace('_', ' ').title()}:")

                priority_indicator = "⚡" if instruction.priority >= 9 else "•"
                lines.append(f"  {priority_indicator} {instruction.content}")

            return "\n".join(lines)

    def track_usage(
        self,
        instruction_id: str,
        success: bool,
        feedback: Optional[str] = None
    ) -> None:
        """Track instruction usage and effectiveness.

        Args:
            instruction_id: Instruction that was used
            success: Whether the instruction led to success
            feedback: Optional feedback about the instruction
        """
        with self._lock:
            instruction = self._instructions.get(instruction_id)
            if not instruction:
                return

            instruction.update_effectiveness(success)

            if feedback:
                if "feedback" not in instruction.metadata:
                    instruction.metadata["feedback"] = []
                instruction.metadata["feedback"].append({
                    "timestamp": datetime.now().isoformat(),
                    "success": success,
                    "feedback": feedback
                })

            if self.auto_save:
                self.save_playbook()

    def _prune_category(self, category: InstructionCategory, count: int = 1) -> int:
        """Remove least effective instructions from a category.

        Args:
            category: Category to prune
            count: Number of instructions to remove

        Returns:
            Number of instructions removed
        """
        instruction_ids = self._instructions_by_category.get(category, [])
        if not instruction_ids:
            return 0

        # Get instructions sorted by effectiveness (ascending)
        instructions = [
            (iid, self._instructions[iid])
            for iid in instruction_ids
        ]
        instructions.sort(key=lambda x: x[1].effectiveness_score)

        # Remove least effective
        removed = 0
        for iid, instruction in instructions[:count]:
            if self.remove_instruction(iid):
                removed += 1

        return removed

    def _record_change(
        self,
        action: str,
        instruction: Instruction,
        source: str,
        details: Optional[Dict] = None
    ) -> None:
        """Record a change to the playbook.

        Args:
            action: Type of change (add, update, remove)
            instruction: Instruction that changed
            source: Source of the change
            details: Additional details about the change
        """
        change_record = {
            "version": self._version,
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "instruction_id": instruction.instruction_id,
            "category": instruction.category.value,
            "source": source,
            "details": details or {}
        }

        self._change_history.append(change_record)
        self._version += 1

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the playbook.

        Returns:
            Dictionary of statistics
        """
        with self._lock:
            if not self._instructions:
                return {
                    "total_instructions": 0,
                    "instructions_by_category": {},
                    "avg_effectiveness": 0,
                    "version": self._version
                }

            effectiveness_scores = [
                i.effectiveness_score
                for i in self._instructions.values()
                if i.usage_count > 0
            ]

            return {
                "total_instructions": len(self._instructions),
                "instructions_by_category": {
                    category.value: len(ids)
                    for category, ids in self._instructions_by_category.items()
                    if ids
                },
                "avg_effectiveness": (
                    sum(effectiveness_scores) / len(effectiveness_scores)
                    if effectiveness_scores else 0
                ),
                "total_usage": sum(i.usage_count for i in self._instructions.values()),
                "version": self._version,
                "change_count": len(self._change_history)
            }
