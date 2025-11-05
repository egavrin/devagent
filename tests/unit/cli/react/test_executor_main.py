"""Main execution flow tests for the CLI ReAct executor."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import click
import pytest

from ai_dev_agent.cli.react.executor import _execute_react_assistant
from ai_dev_agent.core.utils.config import Settings
from ai_dev_agent.engine.react.types import TaskSpec
from tests.unit.cli.react._executor_test_support import (
    DummyRouter,
    StubContextSynthesizer,
    StubDynamicContextTracker,
    StubExecutor,
    StubSessionManager,
)


@pytest.fixture
def base_settings() -> Settings:
    settings = Settings()
    settings.enable_cost_tracking = False
    settings.enable_retry = False
    settings.enable_summarization = False
    settings.enable_memory_bank = False
    settings.enable_reflection = False
    settings.adaptive_budget_scaling = False
    settings.keep_last_assistant_messages = 3
    settings.global_system_message = None
    settings.global_context_message = None
    return settings


def _build_click_context() -> click.Context:
    ctx = click.Context(click.Command("devagent"))
    ctx.obj = {}
    return ctx


def test_execute_react_assistant_exposes_search_queries(monkeypatch, base_settings):
    """Returned payload should surface deduplicated search queries for callers."""

    stub_manager = StubSessionManager()
    monkeypatch.setattr(
        "ai_dev_agent.cli.react.executor.SessionManager.get_instance",
        lambda: stub_manager,
    )
    monkeypatch.setattr(
        "ai_dev_agent.cli.react.executor.ContextSynthesizer",
        lambda: StubContextSynthesizer(),
    )
    monkeypatch.setattr(
        "ai_dev_agent.cli.react.executor.DynamicContextTracker",
        StubDynamicContextTracker,
    )
    monkeypatch.setattr(
        "ai_dev_agent.cli.react.executor.BudgetAwareExecutor",
        StubExecutor,
    )
    monkeypatch.setattr(
        "ai_dev_agent.cli.react.executor._resolve_intent_router",
        lambda: DummyRouter,
    )
    monkeypatch.setattr(
        "ai_dev_agent.cli.react.executor._collect_project_structure_outline",
        lambda _root: {},
    )
    monkeypatch.setattr(
        "ai_dev_agent.cli.react.executor._detect_repository_language",
        lambda *_args, **_kwargs: ("python", 10),
    )
    monkeypatch.setattr(
        "ai_dev_agent.cli.react.executor.resolve_prompt_input",
        lambda value: value,
    )
    monkeypatch.setattr(
        "ai_dev_agent.cli.react.executor.create_budget_integration",
        lambda _settings: MagicMock(),
    )

    ctx = _build_click_context()
    client = SimpleNamespace(invoke_tools=MagicMock(), complete=MagicMock())

    result_payload = _execute_react_assistant(
        ctx,
        client,
        base_settings,
        user_prompt="Run analysis",
        use_planning=False,
        system_extension=None,
        format_schema=None,
    )

    assert "search_queries" in result_payload, "Search activity should be returned to callers"
    assert result_payload["search_queries"] == [
        "pytest"
    ], "Queries should be deduplicated and ordered"
    assert stub_manager.get_session("irrelevant").metadata.get("search_queries") == ["pytest"]
