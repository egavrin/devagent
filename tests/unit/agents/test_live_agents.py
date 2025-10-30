"""Tests for live agent wrappers to ensure code paths stay covered."""

from ai_dev_agent.agents.base import AgentContext
from ai_dev_agent.agents.specialized import live_agents
from ai_dev_agent.core.utils import config as config_module


def test_get_live_agent_returns_expected_instances():
    assert isinstance(live_agents.get_live_agent("design"), live_agents.LiveDesignAgent)
    assert isinstance(live_agents.get_live_agent("review"), live_agents.LiveReviewAgent)
    assert live_agents.get_live_agent("unknown") is None


def test_live_design_agent_returns_guidance(monkeypatch):
    agent = live_agents.LiveDesignAgent()
    context = AgentContext(session_id="design-test")

    result = agent.execute("Draft API service architecture", context)

    assert result.success
    assert "Design task received" in result.output
    assert "API specifications" in result.metadata["prompt"]
    assert result.metadata["type"] == "design"


def test_live_review_agent_requires_api_key(monkeypatch):
    class StubSettings:
        def __init__(self):
            self.api_key = None

    monkeypatch.setattr(config_module, "Settings", lambda: StubSettings())
    monkeypatch.delenv("DEVAGENT_API_KEY", raising=False)

    agent = live_agents.LiveReviewAgent()
    context = AgentContext(session_id="review-test")

    result = agent.execute("Inspect security posture", context)

    assert not result.success
    assert result.error == "Missing API key"
    assert "No API key configured" in result.output


def test_live_review_agent_success_path(monkeypatch):
    class StubSettings:
        def __init__(self):
            self.api_key = "live-token"

    monkeypatch.setattr(config_module, "Settings", lambda: StubSettings())

    agent = live_agents.LiveReviewAgent()
    context = AgentContext(session_id="review-live")

    result = agent.execute("Audit user flows", context)

    assert result.success
    assert "Review task received" in result.output
    assert result.metadata["type"] == "review"
    assert "security vulnerabilities" in result.metadata["prompt"].lower()
