"""
Work Planning Storage

Handles persistence of work plans to disk.
"""

import json
from pathlib import Path
from typing import Optional, List
from .models import WorkPlan


class WorkPlanStorage:
    """Handles persistence of work plans"""

    def __init__(self, storage_dir: Optional[Path] = None):
        """
        Initialize storage.

        Args:
            storage_dir: Directory for storing plans. Defaults to ~/.devagent/plans
        """
        if storage_dir is None:
            storage_dir = Path.home() / ".devagent" / "plans"

        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save_plan(self, plan: WorkPlan) -> None:
        """
        Save plan to disk.

        Args:
            plan: WorkPlan to save
        """
        file_path = self.storage_dir / f"{plan.id}.json"
        with open(file_path, "w") as f:
            json.dump(plan.to_dict(), f, indent=2)

    def load_plan(self, plan_id: str) -> Optional[WorkPlan]:
        """
        Load plan from disk.

        Args:
            plan_id: ID of plan to load

        Returns:
            WorkPlan if found, None otherwise
        """
        file_path = self.storage_dir / f"{plan_id}.json"
        if not file_path.exists():
            return None

        with open(file_path, "r") as f:
            data = json.load(f)
            return WorkPlan.from_dict(data)

    def list_plans(self) -> List[WorkPlan]:
        """
        List all plans.

        Returns:
            List of all WorkPlan objects
        """
        plans = []
        for file_path in self.storage_dir.glob("*.json"):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    plans.append(WorkPlan.from_dict(data))
            except (json.JSONDecodeError, KeyError):
                # Skip invalid files
                continue
        return plans

    def delete_plan(self, plan_id: str) -> bool:
        """
        Delete plan.

        Args:
            plan_id: ID of plan to delete

        Returns:
            True if deleted, False if not found
        """
        file_path = self.storage_dir / f"{plan_id}.json"
        if file_path.exists():
            file_path.unlink()
            return True
        return False
