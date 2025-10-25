"""CLI package exposing the DevAgent command entry points."""
from __future__ import annotations

from .commands import (
    NaturalLanguageGroup,
    cli,
    main,
    query,
    chat,
    review,
    create_design,
    generate_tests,
    write_code,
)
from .router import IntentDecision, IntentRouter, IntentRoutingError
from .utils import get_llm_client, infer_task_files, update_task_state
from ai_dev_agent.core.utils.config import Settings, load_settings

__all__ = [
    "NaturalLanguageGroup",
    "IntentDecision",
    "IntentRouter",
    "IntentRoutingError",
    "cli",
    "main",
    "query",
    "chat",
    "review",
    "create_design",
    "generate_tests",
    "write_code",
    "get_llm_client",
    "infer_task_files",
    "update_task_state",
    "load_settings",
    "Settings",
]
