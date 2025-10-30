from types import SimpleNamespace

import pytest

from ai_dev_agent.core.utils.budget_integration import BudgetIntegration, create_budget_integration
from ai_dev_agent.core.utils.config import Settings
from ai_dev_agent.core.utils.cost_tracker import (
    CostTracker,
    TokenUsage,
    create_token_usage_from_response,
)


def test_cost_tracker_records_and_formats():
    tracker = CostTracker()
    usage = TokenUsage(
        prompt_tokens=2_000,
        completion_tokens=1_000,
        cache_read_tokens=500,
        cache_write_tokens=250,
        reasoning_tokens=100,
    )
    tracker.track_request("claude-3-sonnet-20240229", usage, iteration=1, phase="exploration")
    tracker.track_request("claude-3-sonnet-20240229", usage, iteration=2, phase="exploration")
    tracker.track_request("deepseek-coder", usage, iteration=2, phase="build")

    inline = tracker.format_inline()
    assert "Cost:" in inline

    summary = tracker.format_summary(detailed=True)
    assert "Total Cost" in summary
    assert "Per Phase" in summary
    assert "Per Model" in summary

    forecast = tracker.get_forecast(remaining_iterations=3)
    assert forecast > 0
    assert tracker.should_warn(0.0001)

    avg_in, avg_out = tracker.get_average_tokens_per_iteration()
    assert avg_in > 0 and avg_out > 0


def test_create_token_usage_from_response_parses_usage():
    payload = {
        "usage": {
            "prompt_tokens": 123,
            "completion_tokens": 45,
            "total_tokens": 168,
            "cache_creation_input_tokens": 10,
            "cache_read_input_tokens": 5,
            "reasoning_tokens": 2,
        }
    }
    usage = create_token_usage_from_response(payload)
    assert usage.prompt_tokens == 123
    assert usage.cache_read_tokens == 5
    assert usage.reasoning_tokens == 2


def test_budget_integration_tracks_cost_and_retry(tmp_path, monkeypatch):
    settings = Settings()
    settings.enable_cost_tracking = True
    settings.enable_retry = True
    settings.enable_summarization = True
    settings.enable_two_tier_pruning = True

    integration = create_budget_integration(settings)
    assert isinstance(integration, BudgetIntegration)

    llm_stub = SimpleNamespace(complete=lambda *args, **kwargs: "summary")
    integration.initialize_summarizer(llm_stub)
    assert integration.summarizer is not None

    # Inject a deterministic retry handler
    retried = {"count": 0}

    def fake_execute(func, *args, on_retry=None, **kwargs):
        retried["count"] += 1
        return func(*args, **kwargs)

    integration.retry_handler = SimpleNamespace(
        execute_with_retry=fake_execute, reset=lambda: None, get_retry_stats=lambda: retried
    )

    def sample_func(x):
        return x + 1

    result = integration.execute_with_retry(sample_func, 1)
    assert result == 2
    assert retried["count"] == 1

    llm_response = {
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 25,
            "total_tokens": 75,
        }
    }
    integration.track_llm_call("default", llm_response, iteration=1, phase="build")

    summary = integration.get_cost_summary(detailed=True)
    assert "Total Cost" in summary

    forecast = integration.get_cost_forecast(remaining_iterations=2)
    assert forecast >= 0

    status = integration.get_integration_status()
    assert status["cost_tracking"]["enabled"]
    assert status["retry_handling"]["enabled"]

    integration.reset_retry_stats()

    # Summarizer path exercising optimize_context branch
    integration.summarizer = SimpleNamespace(optimize_context=lambda messages, budget: ["ok"])
    summarized = integration.summarize_if_needed(["a"], 10)
    assert summarized == ["ok"]

    # Summarizer with summarize_if_needed fallback
    integration.summarizer = SimpleNamespace(
        summarize_if_needed=lambda messages, budget: ["fallback"]
    )
    summarized = integration.summarize_if_needed(["a"], 10)
    assert summarized == ["fallback"]

    # Disabled summarizer returns original
    integration.summarizer = None
    assert integration.summarize_if_needed(["orig"], 5) == ["orig"]
