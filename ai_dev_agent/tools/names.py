"""Central definitions for canonical tool identifiers."""
from __future__ import annotations

READ = "read"
WRITE = "write"
RUN = "run"
FIND = "find"
GREP = "grep"
SYMBOLS = "symbols"

ALL_TOOLS = (
    READ,
    WRITE,
    RUN,
    FIND,
    GREP,
    SYMBOLS,
)

__all__ = [
    "READ",
    "WRITE",
    "RUN",
    "FIND",
    "GREP",
    "SYMBOLS",
    "ALL_TOOLS",
]
