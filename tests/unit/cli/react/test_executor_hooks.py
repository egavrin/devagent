"""Hook and integration coverage for the CLI ReAct executor."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import click
import pytest

from ai_dev_agent.cli.react.executor import _execute_react_assistant
from ai_dev_agent.core.utils.config import Settings
from tests.unit.cli.react._executor_test_support import (
    DummyRouter,
    StubContextSynthesizer,
    StubDynamicContextTracker,
    StubExecutor,
    StubSessionManager,
)


@pytest.fixture
def lightweight_settings() -> Settings:
    settings = Settings()
    settings.enable_cost_tracking = False
    settings.enable_retry = False
    settings.enable_summarization = False
    settings.enable_memory_bank = False
    settings.enable_reflection = False
    settings.adaptive_budget_scaling = False
    settings.keep_last_assistant_messages = 2
    return settings


def _prepare_context() -> click.Context:
    ctx = click.Context(click.Command("devagent"))
    ctx.obj = {}
    return ctx


def _apply_common_patches(monkeypatch, session_manager):
    monkeypatch.setattr(
        "ai_dev_agent.cli.react.executor.SessionManager.get_instance",
        lambda: session_manager,
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
        lambda *_args, **_kwargs: ("python", 12),
    )
    monkeypatch.setattr(
        "ai_dev_agent.cli.react.executor.resolve_prompt_input",
        lambda value: value,
    )
    monkeypatch.setattr(
        "ai_dev_agent.cli.react.executor.create_budget_integration",
        lambda _settings: MagicMock(),
    )
    monkeypatch.setattr(
        "ai_dev_agent.cli.react.executor.sanitize_conversation",
        lambda messages: messages,
    )


def test_step_hook_persists_records_in_context(monkeypatch, lightweight_settings):
    """Step hook data should be captured in the Click context for later inspection."""

    session_manager = StubSessionManager()
    _apply_common_patches(monkeypatch, session_manager)

    ctx = _prepare_context()
    client = SimpleNamespace(invoke_tools=MagicMock(), complete=MagicMock())

    payload = _execute_react_assistant(
        ctx,
        client,
        lightweight_settings,
        user_prompt="Inspect hooks",
    )

    assert "step_records" in ctx.obj, "Expected step records to be stored on ctx.obj"
    assert len(ctx.obj["step_records"]) == 2, "Both steps should be captured"
    assert (
        payload["result"].steps == ctx.obj["step_records"]
    ), "Stored steps should match run result"


def test_iteration_hook_tracks_phase_history(monkeypatch, lightweight_settings):
    """Iteration hook should update the context with observed phases."""

    session_manager = StubSessionManager()
    _apply_common_patches(monkeypatch, session_manager)

    ctx = _prepare_context()
    client = SimpleNamespace(invoke_tools=MagicMock(), complete=MagicMock())

    _execute_react_assistant(
        ctx,
        client,
        lightweight_settings,
        user_prompt="Track phases",
    )

    assert ctx.obj.get("phase_history") == [
        "exploration",
        "synthesis",
    ], "Phase transitions observed by the iteration hook should be recorded"
