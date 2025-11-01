"""Comprehensive tests for the ReactiveExecutor module."""

from unittest.mock import MagicMock, patch

import pytest

from ai_dev_agent.engine.react.evaluator import GateEvaluator
from ai_dev_agent.engine.react.loop import ReactiveExecutor
from ai_dev_agent.engine.react.types import (
    ActionRequest,
    EvaluationResult,
    MetricsSnapshot,
    Observation,
    StepRecord,
    TaskSpec,
)


class TestReactiveExecutor:
    """Test suite for ReactiveExecutor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.evaluator = MagicMock(spec=GateEvaluator)

        # Properly configure the mock evaluator
        config_mock = MagicMock()
        config_mock.steps_budget = 10
        self.evaluator.config = config_mock

        self.executor = ReactiveExecutor(evaluator=self.evaluator, default_max_steps=25)

        self.task = TaskSpec(
            identifier="test_task",
            goal="Test task goal",
            category="test",
            instructions="Test instructions",
        )

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        executor = ReactiveExecutor()
        assert executor.evaluator is None
        assert executor.default_max_steps == 25

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        evaluator = MagicMock(spec=GateEvaluator)
        executor = ReactiveExecutor(evaluator=evaluator, default_max_steps=50)
        assert executor.evaluator is evaluator
        assert executor.default_max_steps == 50

    def test_init_with_invalid_max_steps(self):
        """Test initialization with invalid max_steps."""
        executor = ReactiveExecutor(default_max_steps=0)
        assert executor.default_max_steps == 1  # Should default to 1

        executor = ReactiveExecutor(default_max_steps=-5)
        assert executor.default_max_steps == 1  # Should default to 1

    def test_run_simple_success(self):
        """Test successful run with single step."""
        action = ActionRequest(
            step_id="S1", thought="Test thought", tool="test_tool", args={"key": "value"}
        )
        observation = Observation(success=True, outcome="Success", tool="test_tool")

        action_provider = MagicMock(return_value=action)
        tool_invoker = MagicMock(return_value=observation)

        self.evaluator.evaluate.return_value = EvaluationResult(
            gates={"gate1": True},
            required_gates={"gate1": True},
            should_stop=True,
            stop_reason="All gates passed",
            status="success",
        )

        result = self.executor.run(self.task, action_provider, tool_invoker)

        assert result.status == "success"
        assert result.stop_reason == "All gates passed"
        assert len(result.steps) == 1
        assert result.steps[0].action == action
        assert result.steps[0].observation == observation
        action_provider.assert_called_once()
        tool_invoker.assert_called_once_with(action)

    def test_run_with_prior_steps(self):
        """Test run with prior history."""
        prior_step = StepRecord(
            action=ActionRequest(step_id="S1", thought="Test thought", tool="prior_tool", args={}),
            observation=Observation(success=True, outcome="Prior", tool="prior_tool"),
            metrics=MetricsSnapshot(),
            evaluation=EvaluationResult(
                gates={}, required_gates={}, should_stop=False, status="in_progress"
            ),
            step_index=1,
        )

        action = ActionRequest(step_id="S1", thought="Test thought", tool="new_tool", args={})
        observation = Observation(success=True, outcome="New", tool="new_tool")

        action_provider = MagicMock(return_value=action)
        tool_invoker = MagicMock(return_value=observation)

        self.evaluator.evaluate.return_value = EvaluationResult(
            gates={}, required_gates={}, should_stop=True, status="success"
        )

        result = self.executor.run(
            self.task, action_provider, tool_invoker, prior_steps=[prior_step]
        )

        assert len(result.steps) == 2
        assert result.steps[0] == prior_step
        assert result.steps[1].action == action

    def test_run_stop_iteration(self):
        """Test run stopped by StopIteration from action provider."""
        action_provider = MagicMock(side_effect=StopIteration())
        tool_invoker = MagicMock()

        result = self.executor.run(self.task, action_provider, tool_invoker)

        assert result.status == "success"
        assert result.stop_reason == "Action provider requested stop."
        assert len(result.steps) == 0
        tool_invoker.assert_not_called()

    def test_run_budget_exhausted(self):
        """Test run stopped by step budget."""
        action = ActionRequest(step_id="S1", thought="Test thought", tool="test_tool", args={})
        observation = Observation(success=True, outcome="Success", tool="test_tool")

        action_provider = MagicMock(return_value=action)
        tool_invoker = MagicMock(return_value=observation)

        self.evaluator.evaluate.return_value = EvaluationResult(
            gates={}, required_gates={}, should_stop=False, status="in_progress"
        )

        result = self.executor.run(self.task, action_provider, tool_invoker, max_steps=2)

        assert result.status == "failed"
        assert result.stop_reason == "Execution stopped before gates were satisfied."
        assert len(result.steps) == 2  # Limited by max_steps

    def test_run_exception_during_action(self):
        """Test exception raised by action provider."""
        action_provider = MagicMock(side_effect=ValueError("Action error"))
        tool_invoker = MagicMock()

        with pytest.raises(ValueError, match="Action error"):
            self.executor.run(self.task, action_provider, tool_invoker)

    def test_run_invalid_action_payload(self):
        """Test invalid action payload from provider."""
        action_provider = MagicMock(return_value="invalid")  # Not dict or ActionRequest
        tool_invoker = MagicMock()

        with pytest.raises(TypeError, match="Action provider returned unsupported type"):
            self.executor.run(self.task, action_provider, tool_invoker)

    def test_run_action_validation_error(self):
        """Test ValidationError during action validation."""
        # Return dict with invalid data
        action_provider = MagicMock(return_value={"invalid_field": "value"})
        tool_invoker = MagicMock()

        with pytest.raises(ValueError, match="Action provider returned invalid payload"):
            self.executor.run(self.task, action_provider, tool_invoker)

    def test_ensure_action_from_action_request(self):
        """Test _ensure_action with ActionRequest input."""
        action = ActionRequest(step_id="S1", thought="Test thought", tool="test", args={})
        result = self.executor._ensure_action(action, 1)
        assert result is action

    def test_ensure_action_from_dict(self):
        """Test _ensure_action with dict input."""
        payload = {"tool": "test", "thought": "Test thought", "args": {}}
        result = self.executor._ensure_action(payload, 1)
        assert isinstance(result, ActionRequest)
        assert result.tool == "test"
        assert result.step_id == "S1"  # Default added

    def test_ensure_action_from_dict_with_step_id(self):
        """Test _ensure_action preserves existing step_id."""
        payload = {"tool": "test", "thought": "Test thought", "args": {}, "step_id": "custom"}
        result = self.executor._ensure_action(payload, 1)
        assert result.step_id == "custom"

    def test_ensure_action_invalid_type(self):
        """Test _ensure_action with invalid type."""
        with pytest.raises(TypeError, match="Action provider returned unsupported type"):
            self.executor._ensure_action(123, 1)

    def test_invoke_tool_success(self):
        """Test successful tool invocation."""
        action = ActionRequest(step_id="S1", thought="Test thought", tool="test", args={})
        observation = Observation(success=True, outcome="Success")
        invoker = MagicMock(return_value=observation)

        result = self.executor._invoke_tool(invoker, action)

        assert result is observation
        assert result.tool == "test"  # Tool should be set

    def test_invoke_tool_returns_dict(self):
        """Test tool returning dict."""
        action = ActionRequest(step_id="S1", thought="Test thought", tool="test", args={})
        obs_dict = {"success": True, "outcome": "Success"}
        invoker = MagicMock(return_value=obs_dict)

        result = self.executor._invoke_tool(invoker, action)

        assert isinstance(result, Observation)
        assert result.success is True
        assert result.tool == "test"

    def test_invoke_tool_exception(self):
        """Test tool invocation exception."""
        action = ActionRequest(step_id="S1", thought="Test thought", tool="test", args={})
        invoker = MagicMock(side_effect=RuntimeError("Tool failed"))

        result = self.executor._invoke_tool(invoker, action)

        assert isinstance(result, Observation)
        assert result.success is False
        assert result.error == "Tool failed"
        assert result.tool == "test"

    def test_invoke_tool_invalid_type(self):
        """Test tool returning invalid type."""
        action = ActionRequest(step_id="S1", thought="Test thought", tool="test", args={})
        invoker = MagicMock(return_value=123)  # Invalid type

        with pytest.raises(TypeError, match="Tool invoker returned unsupported type"):
            self.executor._invoke_tool(invoker, action)

    def test_metrics_from_observation_with_metrics_snapshot(self):
        """Test _metrics_from_observation with MetricsSnapshot."""
        metrics = MetricsSnapshot(tokens_used=100)
        # Create observation with dict, then manually set metrics to MetricsSnapshot
        observation = Observation(success=True, outcome="Success", metrics={})
        observation.metrics = metrics  # Set after construction

        result = self.executor._metrics_from_observation(observation)
        assert result is metrics

    def test_metrics_from_observation_with_dict(self):
        """Test _metrics_from_observation with dict metrics."""
        observation = Observation(success=True, outcome="Success", metrics={"tokens_used": 100})

        result = self.executor._metrics_from_observation(observation)
        assert isinstance(result, MetricsSnapshot)
        assert result.tokens_used == 100

    def test_metrics_from_observation_no_metrics(self):
        """Test _metrics_from_observation with no metrics."""
        observation = Observation(success=True, outcome="Success")

        result = self.executor._metrics_from_observation(observation)
        assert isinstance(result, MetricsSnapshot)

    def test_metrics_from_observation_invalid_type(self):
        """Test _metrics_from_observation with invalid type."""
        observation = Observation(success=True, outcome="Success", metrics={})
        observation.metrics = "invalid"  # Set invalid type after construction

        with pytest.raises(TypeError, match="Observation metrics must be dict-like"):
            self.executor._metrics_from_observation(observation)

    def test_resolve_steps_budget_with_evaluator(self):
        """Test _resolve_steps_budget with evaluator."""
        self.evaluator.config.steps_budget = 20

        # No override
        assert self.executor._resolve_steps_budget(None) == 20

        # Override with smaller value
        assert self.executor._resolve_steps_budget(10) == 10

        # Override with larger value (capped by evaluator budget)
        assert self.executor._resolve_steps_budget(30) == 20

        # Invalid override
        assert self.executor._resolve_steps_budget("invalid") == 20
        assert self.executor._resolve_steps_budget(0) == 1

    def test_resolve_steps_budget_without_evaluator(self):
        """Test _resolve_steps_budget without evaluator."""
        executor = ReactiveExecutor(default_max_steps=25)

        # No override
        assert executor._resolve_steps_budget(None) == 25

        # Valid override
        assert executor._resolve_steps_budget(10) == 10

        # Invalid overrides
        assert executor._resolve_steps_budget("invalid") == 25
        assert executor._resolve_steps_budget(0) == 25
        assert executor._resolve_steps_budget(-5) == 25

    def test_derive_status_with_evaluator_success(self):
        """Test _derive_status with evaluator and success."""
        evaluation = EvaluationResult(
            gates={},
            required_gates={},
            should_stop=True,
            stop_reason="Completed successfully",
            status="success",
        )

        status, reason = self.executor._derive_status("gates", evaluation, True, [])

        assert status == "success"
        assert reason == "Completed successfully"

    def test_derive_status_with_evaluator_failed(self):
        """Test _derive_status with evaluator and failure."""
        evaluation = EvaluationResult(
            gates={}, required_gates={}, should_stop=False, stop_reason=None, status="in_progress"
        )

        status, reason = self.executor._derive_status("budget", evaluation, True, [])

        assert status == "failed"
        assert reason == "Execution stopped before gates were satisfied."

    def test_derive_status_without_evaluator_success(self):
        """Test _derive_status without evaluator and success."""
        step = StepRecord(
            action=ActionRequest(step_id="S1", thought="Test thought", tool="test", args={}),
            observation=Observation(success=True, outcome="Success"),
            metrics=MetricsSnapshot(),
            evaluation=EvaluationResult(
                gates={}, required_gates={}, should_stop=False, status="in_progress"
            ),
            step_index=1,
        )

        status, reason = self.executor._derive_status("stop_iteration", None, False, [step])

        assert status == "success"
        assert reason == "Completed"

    def test_derive_status_without_evaluator_budget_exhausted(self):
        """Test _derive_status without evaluator and budget exhausted."""
        step = StepRecord(
            action=ActionRequest(step_id="S1", thought="Test thought", tool="test", args={}),
            observation=Observation(success=True, outcome="Success"),
            metrics=MetricsSnapshot(),
            evaluation=EvaluationResult(
                gates={}, required_gates={}, should_stop=False, status="in_progress"
            ),
            step_index=1,
        )

        status, reason = self.executor._derive_status("budget", None, False, [step])

        assert status == "failed"
        assert reason == "Step budget exhausted."

    def test_derive_status_without_evaluator_no_steps(self):
        """Test _derive_status without evaluator and no steps."""
        status, reason = self.executor._derive_status("budget", None, False, [])

        assert status == "failed"
        assert reason == "No actions were executed before the step budget was reached."

    def test_derive_status_exception_condition(self):
        """Test _derive_status with exception condition."""
        evaluation = EvaluationResult(
            gates={},
            required_gates={},
            should_stop=True,
            stop_reason="Custom exception message",
            status="failed",
        )

        status, reason = self.executor._derive_status("exception", evaluation, False, [])

        assert status == "failed"
        assert reason == "Custom exception message"

    def test_derive_status_gates_condition_no_evaluator(self):
        """Test _derive_status with gates condition but no evaluator."""
        status, reason = self.executor._derive_status("gates", None, False, [])

        assert status == "failed"  # No steps
        assert reason == "Completed"

    def test_derive_status_unknown_condition(self):
        """Test _derive_status with unknown condition."""
        step = StepRecord(
            action=ActionRequest(step_id="S1", thought="Test thought", tool="test", args={}),
            observation=Observation(success=True, outcome="Success"),
            metrics=MetricsSnapshot(),
            evaluation=EvaluationResult(
                gates={}, required_gates={}, should_stop=False, status="in_progress"
            ),
            step_index=1,
        )

        status, reason = self.executor._derive_status(None, None, False, [step])

        assert status == "success"  # Has steps
        assert reason is None

    def test_run_without_evaluator(self):
        """Test run without evaluator."""
        executor = ReactiveExecutor()

        action = ActionRequest(step_id="S1", thought="Test thought", tool="test", args={})
        observation = Observation(success=True, outcome="Success")

        action_provider = MagicMock(side_effect=[action, StopIteration()])
        tool_invoker = MagicMock(return_value=observation)

        result = executor.run(self.task, action_provider, tool_invoker)

        assert result.status == "success"
        assert len(result.steps) == 1
        assert result.gates == {}
        assert result.required_gates == {}

    def test_run_metrics_extraction(self):
        """Test metrics are properly extracted from observations."""
        action = ActionRequest(step_id="S1", thought="Test thought", tool="test", args={})
        observation = Observation(
            success=True, outcome="Success", metrics={"tokens_used": 50, "time_elapsed": 1.5}
        )

        action_provider = MagicMock(return_value=action)
        tool_invoker = MagicMock(return_value=observation)

        self.evaluator.evaluate.return_value = EvaluationResult(
            gates={}, required_gates={}, should_stop=True, status="success"
        )

        result = self.executor.run(self.task, action_provider, tool_invoker)

        assert result.metrics["tokens_used"] == 50
        assert result.metrics["time_elapsed"] == 1.5

    @patch("ai_dev_agent.engine.react.loop.time")
    def test_run_runtime_calculation(self, mock_time):
        """Test runtime calculation."""
        mock_time.perf_counter.side_effect = [0.0, 2.5]  # Start and end times

        action_provider = MagicMock(side_effect=StopIteration())
        tool_invoker = MagicMock()

        result = self.executor.run(self.task, action_provider, tool_invoker)

        assert result.runtime_seconds == 2.5

    def test_run_complex_workflow(self):
        """Test complex workflow with multiple steps."""
        actions = [
            ActionRequest(step_id="S1", thought="First thought", tool="tool1", args={"p1": 1}),
            ActionRequest(step_id="S2", thought="Second thought", tool="tool2", args={"p2": 2}),
            ActionRequest(step_id="S3", thought="Third thought", tool="tool3", args={"p3": 3}),
        ]

        observations = [
            Observation(success=True, outcome="Result 1", tool="tool1"),
            Observation(success=True, outcome="Result 2", tool="tool2"),
            Observation(success=True, outcome="Result 3", tool="tool3"),
        ]

        action_provider = MagicMock(side_effect=actions)
        tool_invoker = MagicMock(side_effect=observations)

        evaluations = [
            EvaluationResult(
                gates={"g1": False},
                required_gates={"g1": True, "g2": True},
                should_stop=False,
                status="in_progress",
            ),
            EvaluationResult(
                gates={"g1": True, "g2": False},
                required_gates={"g1": True, "g2": True},
                should_stop=False,
                status="in_progress",
            ),
            EvaluationResult(
                gates={"g1": True, "g2": True},
                required_gates={"g1": True, "g2": True},
                should_stop=True,
                stop_reason="All gates satisfied",
                status="success",
            ),
        ]

        self.evaluator.evaluate.side_effect = evaluations

        result = self.executor.run(self.task, action_provider, tool_invoker)

        assert result.status == "success"
        assert result.stop_reason == "All gates satisfied"
        assert len(result.steps) == 3
        assert result.gates == {"g1": True, "g2": True}

        # Verify all actions were executed
        for i, step in enumerate(result.steps):
            assert step.action == actions[i]
            assert step.observation == observations[i]
            assert step.step_index == i + 1
