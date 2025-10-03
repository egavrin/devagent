"""Tests for translating CLI settings into context pruning configuration."""

from ai_dev_agent.cli.utils import _build_context_pruning_config_from_settings
from ai_dev_agent.core.utils.config import Settings


def test_build_context_pruning_config_defaults() -> None:
    settings = Settings()

    config = _build_context_pruning_config_from_settings(settings)

    expected_trigger = int(
        settings.context_pruner_max_total_tokens * settings.context_pruner_trigger_ratio
    )
    assert config.max_total_tokens == settings.context_pruner_max_total_tokens
    assert config.trigger_tokens == expected_trigger
    assert config.keep_recent_messages == settings.context_pruner_keep_recent_messages
    assert config.summary_max_chars == settings.context_pruner_summary_max_chars
    assert config.max_event_history == settings.context_pruner_max_event_history


def test_build_context_pruning_config_clamps_values() -> None:
    settings = Settings(
        context_pruner_max_total_tokens=5_000,
        context_pruner_trigger_ratio=5.0,
        context_pruner_keep_recent_messages=1,
        context_pruner_summary_max_chars=0,
        context_pruner_max_event_history=0,
    )

    config = _build_context_pruning_config_from_settings(settings)

    assert config.max_total_tokens == 5_000
    assert config.trigger_tokens == 5_000
    assert config.keep_recent_messages == 2
    assert config.summary_max_chars == 1
    assert config.max_event_history == 1


def test_build_context_pruning_config_respects_explicit_trigger_tokens() -> None:
    settings = Settings(
        context_pruner_max_total_tokens=2_000,
        context_pruner_trigger_tokens=1_200,
        context_pruner_trigger_ratio=0.2,
    )

    config = _build_context_pruning_config_from_settings(settings)

    assert config.trigger_tokens == 1_200
    assert config.max_total_tokens == 2_000
