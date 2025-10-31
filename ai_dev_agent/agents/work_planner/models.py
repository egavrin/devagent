"""Work Planning Agent Models"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """Task status enum"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class Priority(str, Enum):
    """Task priority enum"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Task(BaseModel):
    """Task model for work planning."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    description: str = ""
    acceptance_criteria: List[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    priority: Priority = Priority.MEDIUM
    effort_estimate: str = "unknown"
    dependencies: List[str] = Field(default_factory=list)
    parent_id: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
    files_involved: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    class Config:
        """Pydantic config"""

        use_enum_values = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """Create task from dictionary."""
        return cls(**data)


class WorkPlan(BaseModel):
    """Work plan model containing multiple tasks."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    goal: str
    context: str = ""
    tasks: List[Task] = Field(default_factory=list)
    version: int = 1
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    is_active: bool = True

    class Config:
        """Pydantic config"""

        use_enum_values = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert work plan to dictionary."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkPlan":
        """Create work plan from dictionary."""
        tasks_data = data.pop("tasks", [])
        return cls(tasks=[Task.from_dict(task) for task in tasks_data], **data)

    def get_task(self, task_id: str) -> Optional[Task]:
        """Find task by ID."""
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None

    def get_next_task(self) -> Optional[Task]:
        """Get the next available task based on priority and dependencies."""
        available_tasks = []

        for task in self.tasks:
            # Skip tasks that are finished or unavailable
            if task.status in {TaskStatus.COMPLETED, TaskStatus.CANCELLED, TaskStatus.BLOCKED}:
                continue

            # Check if task has unmet dependencies
            has_unmet_dependencies = False
            for dep_id in task.dependencies:
                dep_task = self.get_task(dep_id)
                if dep_task is None or dep_task.status != TaskStatus.COMPLETED:
                    has_unmet_dependencies = True
                    break

            if not has_unmet_dependencies:
                available_tasks.append(task)

        if not available_tasks:
            return None

        # Sort by priority (critical first, then high, medium, low)
        priority_order = {
            Priority.CRITICAL: 0,
            Priority.HIGH: 1,
            Priority.MEDIUM: 2,
            Priority.LOW: 3,
        }

        available_tasks.sort(key=lambda task: priority_order[task.priority])
        return available_tasks[0]

    def get_completion_percentage(self) -> float:
        """Calculate completion percentage of all tasks."""
        if not self.tasks:
            return 0.0

        completed_count = sum(1 for task in self.tasks if task.status == TaskStatus.COMPLETED)
        return (completed_count / len(self.tasks)) * 100.0

    def update_task_status(self, task_id: str, status: TaskStatus) -> bool:
        """Update task status and timestamps."""
        task = self.get_task(task_id)
        if not task:
            return False

        task.status = status
        task.updated_at = datetime.now()

        if status == TaskStatus.IN_PROGRESS and task.started_at is None:
            task.started_at = datetime.now()
        elif status == TaskStatus.COMPLETED and task.completed_at is None:
            task.completed_at = datetime.now()

        return True
