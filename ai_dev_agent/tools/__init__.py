"""Initialize built-in tool implementations."""
from __future__ import annotations

from .registry import ToolContext, ToolSpec, registry
from .names import (
    READ,
    WRITE,
    RUN,
    FIND,
    GREP,
    SYMBOLS,
    ALL_TOOLS,
)

# Trigger tool registration by importing subpackages for their side effects.
from . import filesystem as _filesystem  # noqa: F401
from . import code as _code  # noqa: F401
from . import execution as _execution  # noqa: F401
from . import analysis as _analysis  # noqa: F401

# Import simple tools
from . import find as _find  # noqa: F401
from . import grep as _grep  # noqa: F401
from . import symbols as _symbols  # noqa: F401

__all__ = [
    "ToolContext",
    "ToolSpec",
    "registry",
    "READ",
    "WRITE",
    "RUN",
    "FIND",
    "GREP",
    "SYMBOLS",
    "ALL_TOOLS",
]
