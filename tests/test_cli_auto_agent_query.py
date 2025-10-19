"""Tests for auto agent CLI orchestration to boost coverage."""
from types import SimpleNamespace
from click.testing import CliRunner

from ai_dev_agent.cli import auto_agent_query


def test_should_use_multi_agent_detects_complexity():
    assert auto_agent_query.should_use_multi_agent("Build API with tests and review")
    assert not auto_agent_query.should_use_multi_agent("List files")


def test_execute_with_auto_agents_single_mode(monkeypatch):
    result = auto_agent_query.execute_with_auto_agents("List files quickly")
    assert result["mode"] == "single"
    assert result["recommendation"].startswith("devagent query")


def test_execute_with_auto_agents_multi_agent_verbose(monkeypatch, capsys):
    class DummyPlan:
        def __init__(self):
            self.id = "plan987654"
            self.tasks = [
                SimpleNamespace(title="Design module"),
                SimpleNamespace(title="Implement module"),
            ]

        def get_completion_percentage(self):
            return 75.0

    class DummyPlanner:
        def create_plan(self, goal, context):
            assert goal == "Build analytics service"
            assert context["auto_generated"]
            return DummyPlan()

    class DummyRegistry:
        def __init__(self):
            self.registered = []

        def register_agent(self, agent):
            self.registered.append(agent)

    class DummyOrchestrator:
        def __init__(self):
            self.subagents = {}

        def register_subagent(self, name, agent):
            self.subagents[name] = agent

    class DummyWorkflow:
        def __init__(self):
            self.calls = []

        def execute_plan_automatically(self, plan, context, stop_on_failure, progress_callback):
            if progress_callback:
                progress_callback("t1", "started", "Begin design")
                progress_callback("t1", "completed", "Finish design")
            self.calls.append((plan.id, context.session_id, stop_on_failure))
            return {
                "success": True,
                "tasks_completed": 2,
                "total_tasks": 2,
            }

    monkeypatch.setattr(auto_agent_query, "WorkPlanningAgent", lambda: DummyPlanner())
    monkeypatch.setattr(auto_agent_query, "EnhancedAgentRegistry", lambda: DummyRegistry())
    monkeypatch.setattr(auto_agent_query, "OrchestratorAgent", lambda: DummyOrchestrator())
    monkeypatch.setattr(auto_agent_query, "AutomatedWorkflow", lambda: DummyWorkflow())
    monkeypatch.setattr(auto_agent_query, "DesignAgent", lambda: SimpleNamespace(name="design"))
    monkeypatch.setattr(auto_agent_query, "TestingAgent", lambda: SimpleNamespace(name="test"))
    monkeypatch.setattr(auto_agent_query, "ImplementationAgent", lambda: SimpleNamespace(name="implement"))
    monkeypatch.setattr(auto_agent_query, "ReviewAgent", lambda: SimpleNamespace(name="review"))

    result = auto_agent_query.execute_with_auto_agents("Build analytics service", verbose=True)
    captured = capsys.readouterr().out

    assert result["mode"] == "multi-agent"
    assert result["plan_id"] == "plan987654"
    assert result["tasks_completed"] == 2
    assert "Creating work plan" in captured
    assert "Registered 4 specialized agents" in captured
    assert "Begin design" in captured and "Finish design" in captured


def test_auto_agent_command_force_single():
    runner = CliRunner()
    result = runner.invoke(
        auto_agent_query.auto_agent_command,
        ["Simple task", "--force-single"],
    )

    assert result.exit_code == 0
    assert "Forced single-agent mode" in result.output
    assert "devagent query" in result.output


def test_auto_agent_command_force_multi(monkeypatch):
    runner = CliRunner()

    monkeypatch.setattr(
        auto_agent_query,
        "execute_with_auto_agents",
        lambda query, verbose=False: {
            "mode": "multi-agent",
            "success": True,
            "completion_rate": 90.0,
            "tasks_completed": 3,
            "total_tasks": 3,
        },
    )

    result = runner.invoke(
        auto_agent_query.auto_agent_command,
        ["Build system", "--force-multi"],
    )

    assert result.exit_code == 0
    assert "Forced multi-agent mode" in result.output
    assert "Multi-agent execution complete" in result.output
    assert "Completion: 90%" in result.output
