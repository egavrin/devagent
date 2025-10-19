"""Extended coverage for `devagent agent` CLI commands."""
from types import SimpleNamespace
from click.testing import CliRunner

from ai_dev_agent.agents.base import AgentResult
from ai_dev_agent.cli.agent_commands import agent_group
import ai_dev_agent.cli.agent_commands as agent_commands


class _StubAgent:
    """Re-usable stub that records prompts and contexts."""

    def __init__(self, result: AgentResult):
        self.result = result
        self.last_prompt = None
        self.last_context = None

    def execute(self, prompt, context):  # pragma: no cover - executed in tests
        self.last_prompt = prompt
        self.last_context = context
        return self.result


def _invoke_agent(args, monkeypatch, **patches):
    runner = CliRunner()
    for name, factory in patches.items():
        monkeypatch.setattr(agent_commands, name, factory)
    return runner.invoke(agent_group, args)


def test_agent_design_success(monkeypatch):
    stub = _StubAgent(
        AgentResult(
            success=True,
            output="Design ready",
            metadata={"design_document": "docs/design.md"},
        )
    )
    result = _invoke_agent(
        ["design", "search-index", "--context", "latency budget"],
        monkeypatch,
        DesignAgent=lambda: stub,
    )

    assert result.exit_code == 0
    assert "Design Agent: Creating design" in result.output
    assert "✓ Design completed successfully" in result.output
    assert "docs/design.md" in result.output
    assert stub.last_prompt.startswith("Design search-index")
    assert stub.last_context.session_id == "design-search-index"


def test_agent_design_failure_aborts(monkeypatch):
    stub = _StubAgent(
        AgentResult(success=False, output="", error="compilation failed")
    )
    result = _invoke_agent(
        ["design", "failing-feature"],
        monkeypatch,
        DesignAgent=lambda: stub,
    )

    assert result.exit_code == 1
    assert "✗ Design failed: compilation failed" in result.output


def test_agent_test_generates_files(monkeypatch):
    created = ["tests/test_api.py", "tests/test_models.py"]
    stub = _StubAgent(
        AgentResult(
            success=True,
            output="Generated tests",
            metadata={"test_files_created": created},
        )
    )
    result = _invoke_agent(
        ["test", "auth flow", "--coverage", "85", "--type", "unit"],
        monkeypatch,
        TestingAgent=lambda: stub,
    )

    assert result.exit_code == 0
    assert "🧪 Test Agent: Generating tests for 'auth flow'..." in result.output
    for path in created:
        assert f"- {path}" in result.output
    assert stub.last_prompt.startswith("Create unit tests")
    assert "85% coverage" in stub.last_prompt


def test_agent_test_failure_aborts(monkeypatch):
    stub = _StubAgent(
        AgentResult(success=False, output="", error="no plan")
    )
    result = _invoke_agent(
        ["test", "broken feature"],
        monkeypatch,
        TestingAgent=lambda: stub,
    )

    assert result.exit_code == 1
    assert "✗ Test generation failed: no plan" in result.output


def test_agent_implement_success(monkeypatch):
    stub = _StubAgent(
        AgentResult(
            success=True,
            output="Implementation complete",
            metadata={"files_created": ["src/app.py", "tests/test_app.py"]},
        )
    )
    result = _invoke_agent(
        ["implement", "design.md", "--test-file", "tests/test_design.py"],
        monkeypatch,
        ImplementationAgent=lambda: stub,
    )

    assert result.exit_code == 0
    assert "Implementation Agent: Implementing design" in result.output
    assert "files created: 2" in result.output.lower()
    assert stub.last_prompt.endswith("with tests at tests/test_design.py")


def test_agent_implement_failure_aborts(monkeypatch):
    stub = _StubAgent(
        AgentResult(success=False, output="", error="lint failure")
    )
    result = _invoke_agent(
        ["implement", "design.md"],
        monkeypatch,
        ImplementationAgent=lambda: stub,
    )

    assert result.exit_code == 1
    assert "✗ Implementation failed: lint failure" in result.output


def test_agent_review_success(monkeypatch):
    stub = _StubAgent(
        AgentResult(
            success=True,
            output="Review summary",
            metadata={"issues_found": 2, "quality_score": 0.82},
        )
    )
    result = _invoke_agent(
        ["review", "src/app.py", "--report", "report.md"],
        monkeypatch,
        ReviewAgent=lambda: stub,
    )

    assert result.exit_code == 0
    assert "⚠ Found 2 issue(s)" in result.output
    assert "Quality score: 0.82/1.00" in result.output
    assert "Report saved to: report.md" in result.output


def test_agent_review_failure_aborts(monkeypatch):
    stub = _StubAgent(
        AgentResult(success=False, output="", error="runtime error")
    )
    result = _invoke_agent(
        ["review", "src/app.py"],
        monkeypatch,
        ReviewAgent=lambda: stub,
    )

    assert result.exit_code == 1
    assert "✗ Review failed: runtime error" in result.output


def test_agent_orchestrate_plan_not_found(monkeypatch):
    class DummyStorage:
        def load_plan(self, plan_id):
            assert plan_id == "missing"
            return None

    result = _invoke_agent(
        ["orchestrate", "missing", "--auto"],
        monkeypatch,
        WorkPlanStorage=lambda: DummyStorage(),
    )

    assert result.exit_code == 1
    assert "Plan 'missing' not found" in result.output


def test_agent_orchestrate_success(monkeypatch):
    plan = SimpleNamespace(goal="Ship feature", tasks=[1, 2, 3])

    class DummyStorage:
        def load_plan(self, plan_id):
            assert plan_id == "plan-123"
            return plan

    class DummyWorkflow:
        def __init__(self):
            self.calls = []

        def execute_plan_automatically(self, plan_obj, ctx, stop_on_failure, progress_callback):
            assert plan_obj is plan
            progress_callback("t1", "started", "Task 1")
            progress_callback("t1", "completed", "Task 1")
            self.calls.append(
                (plan_obj.goal, ctx.session_id, stop_on_failure)
            )
            return {
                "success": True,
                "tasks_completed": 3,
                "total_tasks": 3,
                "tasks_failed": 0,
                "completion_percentage": 100.0,
            }

    workflow = DummyWorkflow()
    result = _invoke_agent(
        ["orchestrate", "plan-123", "--auto"],
        monkeypatch,
        WorkPlanStorage=lambda: DummyStorage(),
        AutomatedWorkflow=lambda: workflow,
    )

    assert result.exit_code == 0
    assert "Plan: Ship feature" in result.output
    assert "Starting: Task 1" in result.output
    assert "✓ Completed: Task 1" in result.output
    assert "✓ Plan completed successfully!" in result.output
    assert workflow.calls == [("Ship feature", "orchestrate-plan-123", True)]


def test_agent_list_outputs_agents():
    runner = CliRunner()
    result = runner.invoke(agent_group, ["list"])

    assert result.exit_code == 0
    assert "Available Agents" in result.output
    assert "DesignAgent" in result.output
    assert "ReviewAgent" in result.output


def test_agent_workflow_requires_steps():
    runner = CliRunner()
    result = runner.invoke(agent_group, ["workflow", "refactor-api"])

    assert result.exit_code == 0
    assert "No workflow steps provided" in result.output


def test_agent_workflow_validates_and_runs(monkeypatch):
    class DummyOrchestrator:
        def __init__(self):
            self.subagents = {}

        def register_subagent(self, name, agent):
            self.subagents[name] = agent

        def coordinate_workflow(self, data, context):
            assert data["goal"] == "improve docs"
            assert data["steps"] == [
                {"agent": "design", "task": "draft outline"},
                {"agent": "review", "task": "quality check"},
            ]
            assert context.session_id == "workflow-improve docs"
            return {
                "success": True,
                "steps_completed": 2,
                "total_steps": 2,
            }

    stub_design = object()
    stub_test = object()
    stub_impl = object()
    stub_review = object()

    result = _invoke_agent(
        [
            "workflow",
            "improve docs",
            "--steps",
            "design:draft outline",
            "--steps",
            "invalid-step",
            "--steps",
            "review:quality check",
        ],
        monkeypatch,
        OrchestratorAgent=lambda: DummyOrchestrator(),
        DesignAgent=lambda: stub_design,
        TestingAgent=lambda: stub_test,
        ImplementationAgent=lambda: stub_impl,
        ReviewAgent=lambda: stub_review,
    )

    assert result.exit_code == 0
    assert "Invalid step format: invalid-step" in result.output
    assert "Steps completed: 2/2" in result.output
