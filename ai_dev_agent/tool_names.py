"""Central definitions for canonical tool identifiers that avoid heavy imports."""

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
    "ALL_TOOLS",
    "FIND",
    "GREP",
    "READ",
    "RUN",
    "SYMBOLS",
    "WRITE",
]
