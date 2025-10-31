"""CLI package exposing the DevAgent command entry points."""

from __future__ import annotations

from ai_dev_agent.core.utils.config import Settings, load_settings

from .router import IntentDecision, IntentRouter, IntentRoutingError
from .runtime.commands import (
    chat_command,
    create_design_command,
    execute_chat,
    execute_create_design,
    execute_generate_tests,
    execute_query,
    execute_review,
    execute_write_code,
    generate_tests_command,
    query_command,
    review_command,
    write_code_command,
)
from .runtime.main import NaturalLanguageGroup, cli
from .utils import get_llm_client, infer_task_files, update_task_state

chat = chat_command
create_design = create_design_command
generate_tests = generate_tests_command
query = query_command
review = review_command
write_code = write_code_command

# Expose internal helpers for backward compatibility with existing tests.
try:  # pragma: no cover - defensive guard
    import importlib

    review_module = importlib.import_module("ai_dev_agent.cli.review")
    review.run_review = review_module.run_review
except Exception:
    pass


def main() -> None:
    """Invoke the CLI entry point."""
    cli(prog_name="devagent")


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
    "execute_chat",
    "execute_create_design",
    "execute_generate_tests",
    "execute_query",
    "execute_review",
    "execute_write_code",
]
