"""Central definitions for canonical tool identifiers that avoid heavy imports."""

from __future__ import annotations

READ = "read"
WRITE = "write"
RUN = "run"
FIND = "find"
GREP = "grep"
SYMBOLS = "symbols"
DELEGATE = "delegate"
GET_TASK_STATUS = "get_task_status"
PLAN = "plan"

ALL_TOOLS = (
    READ,
    WRITE,
    RUN,
    FIND,
    GREP,
    SYMBOLS,
    DELEGATE,
    GET_TASK_STATUS,
    PLAN,
)

__all__ = [
    "ALL_TOOLS",
    "DELEGATE",
    "FIND",
    "GREP",
    "GET_TASK_STATUS",
    "PLAN",
    "READ",
    "RUN",
    "SYMBOLS",
    "WRITE",
]
