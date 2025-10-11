"""
Work Planning Data Models

Defines Task and WorkPlan models with metadata for intelligent planning.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
import uuid


class TaskStatus(str, Enum):
    """Task lifecycle states"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class Priority(str, Enum):
    """Task priority levels"""

    CRITICAL = "critical"  # Blocks everything else
    HIGH = "high"  # Important, do soon
    MEDIUM = "medium"  # Normal priority
    LOW = "low"  # Nice to have


@dataclass
class Task:
    """Individual work item with metadata"""

    # Identity
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""

    # Content
    description: str = ""  # Detailed explanation (Aider style)
    acceptance_criteria: List[str] = field(default_factory=list)

    # Metadata
    status: TaskStatus = TaskStatus.PENDING
    priority: Priority = Priority.MEDIUM
    effort_estimate: str = "unknown"  # "15m", "1h", "2h", "1d", etc.

    # Relationships
    dependencies: List[str] = field(default_factory=list)  # Task IDs
    parent_id: Optional[str] = None  # For subtasks
    tags: List[str] = field(default_factory=list)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Context
    notes: List[str] = field(default_factory=list)
    files_involved: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "acceptance_criteria": self.acceptance_criteria,
            "status": self.status.value,
            "priority": self.priority.value,
            "effort_estimate": self.effort_estimate,
            "dependencies": self.dependencies,
            "parent_id": self.parent_id,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "notes": self.notes,
            "files_involved": self.files_involved,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """Create from dictionary"""
        task = cls(
            id=data["id"],
            title=data["title"],
            description=data["description"],
            acceptance_criteria=data.get("acceptance_criteria", []),
            status=TaskStatus(data["status"]),
            priority=Priority(data["priority"]),
            effort_estimate=data.get("effort_estimate", "unknown"),
            dependencies=data.get("dependencies", []),
            parent_id=data.get("parent_id"),
            tags=data.get("tags", []),
            notes=data.get("notes", []),
            files_involved=data.get("files_involved", []),
        )
        task.created_at = datetime.fromisoformat(data["created_at"])
        task.updated_at = datetime.fromisoformat(data["updated_at"])
        if data.get("started_at"):
            task.started_at = datetime.fromisoformat(data["started_at"])
        if data.get("completed_at"):
            task.completed_at = datetime.fromisoformat(data["completed_at"])
        return task


@dataclass
class WorkPlan:
    """Collection of related tasks for a goal"""

    # Identity
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""

    # Content
    goal: str = ""  # High-level objective
    context: str = ""  # Background, constraints, requirements
    tasks: List[Task] = field(default_factory=list)

    # Metadata
    version: int = 1
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    # Status
    is_active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "name": self.name,
            "goal": self.goal,
            "context": self.context,
            "tasks": [task.to_dict() for task in self.tasks],
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "is_active": self.is_active,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkPlan":
        """Create from dictionary"""
        plan = cls(
            id=data["id"],
            name=data["name"],
            goal=data["goal"],
            context=data["context"],
            tasks=[Task.from_dict(t) for t in data.get("tasks", [])],
            version=data.get("version", 1),
            is_active=data.get("is_active", True),
        )
        plan.created_at = datetime.fromisoformat(data["created_at"])
        plan.updated_at = datetime.fromisoformat(data["updated_at"])
        return plan

    def get_task(self, task_id: str) -> Optional[Task]:
        """Find task by ID"""
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None

    def get_next_task(self) -> Optional[Task]:
        """Get next task respecting dependencies and priorities"""
        # Find tasks that are pending and have no incomplete dependencies
        available_tasks = []
        for task in self.tasks:
            if task.status != TaskStatus.PENDING:
                continue

            # Check if all dependencies are completed
            deps_met = all(
                self.get_task(dep_id) and self.get_task(dep_id).status == TaskStatus.COMPLETED
                for dep_id in task.dependencies
            )

            if deps_met:
                available_tasks.append(task)

        if not available_tasks:
            return None

        # Sort by priority (critical > high > medium > low)
        priority_order = {
            Priority.CRITICAL: 0,
            Priority.HIGH: 1,
            Priority.MEDIUM: 2,
            Priority.LOW: 3,
        }
        available_tasks.sort(key=lambda t: priority_order[t.priority])

        return available_tasks[0]

    def get_completion_percentage(self) -> float:
        """Calculate completion percentage"""
        if not self.tasks:
            return 0.0
        completed = sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED)
        return (completed / len(self.tasks)) * 100
