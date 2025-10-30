"""CLI package exposing the DevAgent command entry points."""

from __future__ import annotations

from ai_dev_agent.core.utils.config import Settings, load_settings

from .commands import (
    NaturalLanguageGroup,
    chat,
    cli,
    create_design,
    generate_tests,
    main,
    query,
    review,
    write_code,
)
from .router import IntentDecision, IntentRouter, IntentRoutingError
from .utils import get_llm_client, infer_task_files, update_task_state

__all__ = [
    "IntentDecision",
    "IntentRouter",
    "IntentRoutingError",
    "NaturalLanguageGroup",
    "Settings",
    "chat",
    "cli",
    "create_design",
    "generate_tests",
    "get_llm_client",
    "infer_task_files",
    "load_settings",
    "main",
    "query",
    "review",
    "update_task_state",
    "write_code",
]
