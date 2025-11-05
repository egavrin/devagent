"""Workflow orchestration tools for agent delegation and planning."""

from __future__ import annotations

from pathlib import Path

from ..registry import ToolSpec, registry
from .delegate import delegate
from .get_task_status import get_task_status
from .plan import plan

# Schema directory is two levels up: tools/schemas/tools/
SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schemas" / "tools"

# Register delegate tool
registry.register(
    ToolSpec(
        name="delegate",
        handler=delegate,
        request_schema_path=SCHEMA_DIR / "delegate.request.json",
        response_schema_path=SCHEMA_DIR / "delegate.response.json",
        description="Delegate task to specialized agent for async execution (design_agent, test_agent, review_agent, implementation_agent)",
        display_name="Delegate",
        category="workflow",
    )
)

# Register get_task_status tool
registry.register(
    ToolSpec(
        name="get_task_status",
        handler=get_task_status,
        request_schema_path=SCHEMA_DIR / "get_task_status.request.json",
        response_schema_path=SCHEMA_DIR / "get_task_status.response.json",
        description="Check status of a delegated task (queued, running, completed, failed)",
        display_name="Get Task Status",
        category="workflow",
    )
)

# Register plan tool
registry.register(
    ToolSpec(
        name="plan",
        handler=plan,
        request_schema_path=SCHEMA_DIR / "plan.request.json",
        response_schema_path=SCHEMA_DIR / "plan.response.json",
        description="Create structured work plan with tasks and dependencies for complex goals",
        display_name="Plan",
        category="workflow",
    )
)

__all__ = ["delegate", "get_task_status", "plan"]
