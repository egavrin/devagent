"""Initialize built-in tool implementations."""

from __future__ import annotations

# Import simple tools
# Trigger tool registration by importing subpackages for their side effects.
from . import analysis as _analysis
from . import code as _code
from . import execution as _execution
from . import filesystem as _filesystem
from . import find as _find
from . import grep as _grep
from . import symbols as _symbols
from . import workflow as _workflow
from .names import WRITE  # Kept for backward compatibility but not in ALL_TOOLS
from .names import ALL_TOOLS, DELEGATE, EDIT, FIND, GET_TASK_STATUS, GREP, PLAN, READ, RUN, SYMBOLS
from .registry import ToolContext, ToolSpec, registry

__all__ = [
    "ALL_TOOLS",
    "DELEGATE",
    "EDIT",
    "FIND",
    "GREP",
    "GET_TASK_STATUS",
    "PLAN",
    "READ",
    "RUN",
    "SYMBOLS",
    "WRITE",  # Exported for backward compat, but NOT registered as a tool
    "ToolContext",
    "ToolSpec",
    "registry",
]
