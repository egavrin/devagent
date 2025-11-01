"""Focused tests for the CLI ReAct executor helpers."""

from __future__ import annotations

from threading import RLock
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import click
import pytest

from ai_dev_agent.cli.react.budget_control import (
    BudgetManager,
    create_text_only_tool,
    get_tools_for_iteration,
)
from ai_dev_agent.cli.react.executor import (
    BudgetAwareExecutor,
    _build_json_enforcement_instructions,
    _build_phase_prompt,
    _build_synthesis_prompt,
    _execute_react_assistant,
    _extract_json,
    _record_search_query,
    _truncate_shell_history,
)
from ai_dev_agent.cli.router import IntentDecision
from ai_dev_agent.core.utils.config import Settings
from ai_dev_agent.engine.react.types import (
    ActionRequest,
    EvaluationResult,
    MetricsSnapshot,
    Observation,
    RunResult,
    StepRecord,
    TaskSpec,
    ToolResult,
)
from ai_dev_agent.providers.llm import ToolCallResult
from ai_dev_agent.providers.llm.base import Message


class DummyIntegration:
    """Track execute_with_retry invocations while delegating to the wrapped callable."""

    def __init__(self):
        self.calls: list[tuple] = []

    def execute_with_retry(self, func, *args, **kwargs):
        self.calls.append((func, args, kwargs))
        return func(*args, **kwargs)


def message(role: str, content: str) -> Message:
    """Convenience factory for Message objects."""
    return Message(role=role, content=content)


def test_budget_manager_iteration_progression():
    """Budget manager should yield phases until the budget is exhausted."""
    manager = BudgetManager(3, adaptive_scaling=False)

    contexts = []
    for _ in range(3):
        ctx = manager.next_iteration()
        assert ctx is not None
        contexts.append(ctx)

    assert contexts[-1].is_final is True
    assert manager.next_iteration() is None


def test_get_tools_for_iteration_returns_text_only_on_final():
    """Final iterations should restrict tools to the text-only submission."""
    final_context = manager_context(is_final=True, is_penultimate=False, phase="synthesis")
    available = [
        {"type": "function", "function": {"name": "run", "description": "Execute shell"}},
        {"type": "function", "function": {"name": "read", "description": "Read file"}},
    ]

    selected = get_tools_for_iteration(final_context, available)
    assert selected == [create_text_only_tool()]


def test_json_helper_functions_and_recording():
    """Utility helpers should parse JSON, build prompts, and track search queries."""
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    instructions = _build_json_enforcement_instructions(schema)
    assert '"type": "string"' in instructions

    assert _extract_json('{"answer": "42"}') == {"answer": "42"}
    assert _extract_json('```json\n{"answer": "ok"}\n```') == {"answer": "ok"}
    rich = 'prefix {"nested": [1, 2, {"deep": true}]} suffix'
    assert _extract_json(rich) == {"nested": [1, 2, {"deep": True}]}
    assert _extract_json("") is None
    assert _extract_json("```json\nnot json\n```") is None
    assert _extract_json("no json here") is None

    history = [
        message("user", "first"),
        message("assistant", "reply"),
        message("user", "second"),
        message("assistant", "final"),
        message("user", "dangling"),
    ]
    truncated = _truncate_shell_history(history, max_turns=1)
    assert [msg.content for msg in truncated] == ["second", "final", "dangling"]
    assert _truncate_shell_history(history, max_turns=0) == []

    phase_prompt = _build_phase_prompt(
        phase="exploration",
        user_query="Investigate repos",
        context="Found README.md",
        constraints="- keep it short",
        workspace="/tmp/work",
        repository_language="python",
    )
    assert "Found README.md" in phase_prompt
    assert "python" in phase_prompt.lower()

    synthesis_prompt = _build_synthesis_prompt(
        user_query="Summarize findings",
        context="Tests are green.",
        workspace="/tmp/work",
    )
    assert "submit_final_answer" in synthesis_prompt

    queries: set[str] = set()
    _record_search_query(
        ActionRequest(step_id="1", thought="", tool="find", args={"query": " TODO "}), queries
    )
    _record_search_query(
        ActionRequest(step_id="2", thought="", tool="grep", args={"pattern": " FIXME "}), queries
    )
    _record_search_query(ActionRequest(step_id="3", thought="", tool="write", args={}), queries)
    assert queries == {"TODO", "FIXME"}


def manager_context(
    *,
    is_final: bool,
    is_penultimate: bool,
    phase: str,
    number: int = 1,
    total: int = 1,
) -> any:
    """Helper to create an IterationContext without importing the class directly."""
    return type(
        "IterationContext",
        (),
        {
            "number": number,
            "total": total,
            "remaining": 0 if is_final else max(total - number, 0),
            "percent_complete": 100.0 if is_final else (number / total) * 100,
            "phase": phase,
            "is_final": is_final,
            "is_penultimate": is_penultimate,
            "reflection_count": 0,
            "reflection_allowed": False,
        },
    )()


class StubInvoker:
    """Simple tool invoker returning a successful observation."""

    def __call__(self, action: ActionRequest) -> Observation:
        return Observation(success=True, outcome="ok", tool=action.tool, raw_output="ok")


class StubActionProvider:
    """Minimal action provider implementation for driving the executor."""

    def __init__(self):
        self.session_manager = StubSessionManager()
        self.session_id = "session-1"
        self.client = MagicMock()
        self.client.complete.return_value = "All done"
        self.phase_updates: list[tuple[str, bool]] = []
        self.calls = 0

    def update_phase(self, phase: str, *, is_final: bool = False) -> None:
        self.phase_updates.append((phase, is_final))

    def __call__(self, task: TaskSpec, history):
        self.calls += 1
        return {
            "thought": "complete",
            "tool": "submit_final_answer",
            "args": {"answer": "All done"},
        }

    def last_response_text(self) -> str:
        return "All done"


class StubSession:
    """Simple session container mimicking SessionState."""

    def __init__(self):
        self.history = []
        self.metadata = {}
        self.lock = RLock()


class StubSessionManager:
    """Minimal session manager that stores a single session."""

    def __init__(self):
        self._session = StubSession()

    def ensure_session(self, session_id, system_messages, metadata):
        self._session.history.extend(system_messages)
        self._session.metadata.update(metadata)
        return self._session

    def extend_history(self, session_id, messages):
        self._session.history.extend(messages)

    def compose(self, session_id):
        return list(self._session.history)

    def remove_system_messages(self, session_id, predicate):
        self._session.history = [msg for msg in self._session.history if not predicate(msg)]

    def get_session(self, session_id):
        return self._session

    def add_system_message(self, session_id, message, location="system"):
        self._session.history.append(message)


@patch("ai_dev_agent.cli.react.executor.SessionManager")
@patch("ai_dev_agent.cli.react.executor.ContextSynthesizer")
def test_budget_aware_executor_runs_single_iteration(mock_synth, mock_session_manager):
    """BudgetAwareExecutor should honour the iteration budget and produce a RunResult."""
    mock_session_manager.get_instance.return_value = StubSessionManager()
    mock_synth.return_value = MagicMock(
        synthesize_previous_steps=lambda history, current_step, **_: ""
    )

    executor = BudgetAwareExecutor(BudgetManager(1, adaptive_scaling=False))
    action_provider = StubActionProvider()
    tool_invoker = StubInvoker()

    task = TaskSpec(identifier="task-123", goal="Do the thing")
    result = executor.run(task, action_provider, tool_invoker)

    assert result.status in {"success", "completed"}
    assert result.stop_reason.lower() in {"final_iteration", "provider_stop", "completed"}
    assert action_provider.calls == 1


@patch("ai_dev_agent.cli.react.executor.SessionManager")
def test_budget_aware_executor_handles_tool_exceptions(mock_session_manager):
    """BudgetAwareExecutor._invoke_tool should convert tool exceptions to failed observations."""
    mock_session_manager.get_instance.return_value = StubSessionManager()

    executor = BudgetAwareExecutor(BudgetManager(1, adaptive_scaling=False))
    failing_invoker = MagicMock(side_effect=RuntimeError("boom"))
    action = ActionRequest(step_id="S1", thought="test", tool="run", args={})

    observation = executor._invoke_tool(failing_invoker, action)

    assert observation.success is False
    failure_text = (observation.error or observation.outcome or "").lower()
    assert "boom" in failure_text or "exception" in failure_text


class StubActionProviderWithStopIteration:
    """Action provider that returns text without tool calls, triggering StopIteration."""

    def __init__(self, response_text: str = "The answer is 42 lines."):
        self.session_manager = StubSessionManager()
        self.session_id = "session-1"
        self.client = MagicMock()
        self.phase_updates: list[tuple[str, bool]] = []
        self.calls = 0
        self._response_text = response_text

    def update_phase(self, phase: str, *, is_final: bool = False) -> None:
        self.phase_updates.append((phase, is_final))

    def __call__(self, task: TaskSpec, history):
        self.calls += 1
        # Simulate the LLM returning text without tool calls
        raise StopIteration("No tool calls - synthesis complete")

    def last_response_text(self) -> str:
        # Return the text response that was already generated
        return self._response_text


class FailingActionProvider:
    """Action provider that emits a tool call which will fail."""

    def __init__(self):
        self.session_manager = StubSessionManager()
        self.session_id = "session-1"
        self.client = MagicMock()
        self.phase_updates: list[tuple[str, bool]] = []

    def update_phase(self, phase: str, *, is_final: bool = False) -> None:
        self.phase_updates.append((phase, is_final))

    def __call__(self, task: TaskSpec, history):
        return {
            "thought": "Attempt tool",
            "tool": "find",
            "args": {"query": "TODO"},
            "metadata": {"iteration": 1},
        }

    def last_response_text(self) -> str:
        return ""


@patch("ai_dev_agent.cli.react.executor.SessionManager")
@patch("ai_dev_agent.cli.react.executor.ContextSynthesizer")
def test_executor_uses_existing_response_when_llm_stops_without_tools(
    mock_synth, mock_session_manager, capsys
):
    """When LLM raises StopIteration with existing text response, should use that instead of forcing synthesis."""
    mock_session_manager.get_instance.return_value = StubSessionManager()
    mock_synth.return_value = MagicMock(
        synthesize_previous_steps=lambda history, current_step, **_: "",
        get_redundant_operations=lambda history: [],
        build_constraints_section=lambda redundant_ops: "",
    )

    response_text = "The file contains 142 lines of code."
    executor = BudgetAwareExecutor(BudgetManager(1, adaptive_scaling=False))
    action_provider = StubActionProviderWithStopIteration(response_text)
    tool_invoker = StubInvoker()

    task = TaskSpec(identifier="task-123", goal="Count lines in file")
    result = executor.run(task, action_provider, tool_invoker)

    # Should complete successfully without calling client.complete() for forced synthesis
    assert result.status == "success"
    assert result.stop_reason in {"provider_stop", "Completed"}

    # Should not have attempted forced synthesis (which would print an error)
    captured = capsys.readouterr()
    assert "ERROR: LLM returned empty response" not in captured.err
    assert "ERROR: Failed to force synthesis" not in captured.err

    # Verify the action provider's client.complete was not called
    # (since we should use the existing response instead)
    action_provider.client.complete.assert_not_called()


@patch("ai_dev_agent.cli.react.executor.SessionManager")
@patch("ai_dev_agent.cli.react.executor.ContextSynthesizer")
def test_executor_respects_existing_json_response(mock_synth, mock_session_manager):
    """In JSON mode, a valid cached response should avoid forced synthesis."""
    mock_session_manager.get_instance.return_value = StubSessionManager()
    mock_synth.return_value = MagicMock(
        synthesize_previous_steps=lambda *args, **kwargs: "",
        get_redundant_operations=lambda *args, **kwargs: [],
        build_constraints_section=lambda *_: "",
    )

    executor = BudgetAwareExecutor(
        BudgetManager(1, adaptive_scaling=False),
        format_schema={"type": "object", "properties": {"answer": {"type": "string"}}},
    )
    action_provider = StubActionProviderWithStopIteration('{"answer": "cached"}')
    action_provider.client = MagicMock()
    action_provider.client.complete = MagicMock()
    tool_invoker = StubInvoker()

    task = TaskSpec(identifier="task-json", goal="Provide JSON answer")
    result = executor.run(task, action_provider, tool_invoker)

    assert result.status == "success"
    action_provider.client.complete.assert_not_called()
    assert result.stop_reason in {"provider_stop", "Completed"}


@patch("ai_dev_agent.cli.react.executor.SessionManager")
@patch("ai_dev_agent.cli.react.executor.ContextSynthesizer")
def test_executor_forces_synthesis_via_invoke_tools(mock_synth, mock_session_manager):
    """Forced synthesis should fall back to invoke_tools when complete() is unavailable."""
    mock_session_manager.get_instance.return_value = StubSessionManager()
    mock_synth.return_value = MagicMock(
        synthesize_previous_steps=lambda *args, **kwargs: "",
        get_redundant_operations=lambda *_: [],
        build_constraints_section=lambda *_: "",
    )

    executor = BudgetAwareExecutor(
        BudgetManager(1, adaptive_scaling=False),
        format_schema={"type": "object", "properties": {"answer": {"type": "string"}}},
    )
    action_provider = StubActionProviderWithStopIteration("")
    session = action_provider.session_manager.get_session(action_provider.session_id)
    action_provider.session_manager.ensure_session(
        action_provider.session_id, system_messages=[message("system", "hi")], metadata={}
    )
    session.history.extend([message("assistant", "analysis"), message("user", "what now?")])

    integration = DummyIntegration()

    class InvokeToolsClient:
        def invoke_tools(self, messages, tools, temperature=0.0):
            assert tools == []
            return ToolCallResult(message_content='{"answer": "forced"}')

    client = InvokeToolsClient()
    action_provider.client = client
    action_provider.budget_integration = integration
    tool_invoker = StubInvoker()

    step_events: list[str] = []

    task = TaskSpec(identifier="task-force", goal="Force answer")
    result = executor.run(
        task,
        action_provider,
        tool_invoker,
        step_hook=lambda record, ctx: step_events.append(record.action.tool),
    )

    assert result.status == "success"
    assert step_events[-1] == "submit_final_answer"
    assert result.steps[-1].observation.raw_output == '{"answer": "forced"}'
    assert integration.calls  # ensure integration path executed


@patch("ai_dev_agent.cli.react.executor.SessionManager")
@patch("ai_dev_agent.cli.react.executor.ContextSynthesizer")
def test_executor_forces_synthesis_text_mode(mock_synth, mock_session_manager, capsys):
    """In text mode the executor should call client.complete via the budget integration."""
    mock_session_manager.get_instance.return_value = StubSessionManager()
    mock_synth.return_value = MagicMock(
        synthesize_previous_steps=lambda *args, **kwargs: "",
        get_redundant_operations=lambda *_: [],
        build_constraints_section=lambda *_: "",
    )

    executor = BudgetAwareExecutor(BudgetManager(1, adaptive_scaling=False))
    action_provider = StubActionProviderWithStopIteration("")
    integration = DummyIntegration()
    client = MagicMock()
    client.complete.return_value = "Forced narrative"
    action_provider.client = client
    action_provider.budget_integration = integration

    task = TaskSpec(identifier="task-text", goal="Summarize text")
    result = executor.run(task, action_provider, StubInvoker())

    assert result.status == "success"
    assert result.steps[-1].observation.raw_output == "Forced narrative"
    assert integration.calls  # execute_with_retry used
    captured = capsys.readouterr()
    assert "ERROR:" not in captured.err


@patch("ai_dev_agent.cli.react.executor.SessionManager")
@patch("ai_dev_agent.cli.react.executor.ContextSynthesizer")
def test_executor_forced_synthesis_failure_logs_error(mock_synth, mock_session_manager, capsys):
    """If forced synthesis raises, the executor should print the error and continue."""
    mock_session_manager.get_instance.return_value = StubSessionManager()
    mock_synth.return_value = MagicMock(
        synthesize_previous_steps=lambda *_, **__: "",
        get_redundant_operations=lambda *_: [],
        build_constraints_section=lambda *_: "",
    )

    executor = BudgetAwareExecutor(BudgetManager(1, adaptive_scaling=False))
    action_provider = StubActionProviderWithStopIteration("")
    failing_client = MagicMock()
    failing_client.complete.side_effect = RuntimeError("llm offline")
    action_provider.client = failing_client

    task = TaskSpec(identifier="task-fail", goal="Handle failure")
    result = executor.run(task, action_provider, StubInvoker())

    assert result.status == "success"
    assert result.steps == []
    captured = capsys.readouterr()
    assert "llm offline" in captured.err


class MultiStepActionProvider:
    """Action provider that emits tool calls before requiring forced synthesis."""

    def __init__(self):
        self.session_manager = StubSessionManager()
        self.session_id = "session-1"
        self.client = MagicMock()
        self.client.complete.return_value = "Synthesized final answer"
        self.calls = 0
        self._actions = [
            {
                "thought": "Investigate repository",
                "tool": "find",
                "args": {"query": "tests/"},
                "metadata": {"iteration": 1},
            },
            {
                "thought": "Read supporting file",
                "tool": "read",
                "args": {"path": "README.md"},
                "metadata": {"iteration": 2},
            },
        ]

    def update_phase(self, phase: str, *, is_final: bool = False) -> None:
        pass

    def __call__(self, task: TaskSpec, history):
        action = self._actions[self.calls]
        self.calls += 1
        return action

    def last_response_text(self) -> str:
        # Looks incomplete so the executor keeps the forced synthesis output instead.
        return "Partial findings:"


class StubToolInvokerWithFailures:
    """Invoker that fails and captures injected guidance."""

    def __init__(self):
        self.calls = 0

    def __call__(self, action: ActionRequest) -> Observation:
        self.calls += 1
        return Observation(success=False, outcome="Tool failed", tool=action.tool, error="nope")


@patch("ai_dev_agent.cli.react.executor.ContextSynthesizer")
def test_executor_forces_synthesis_after_tool_iteration(mock_synth):
    """When final iteration uses a tool, executor should force a synthesis answer."""
    mock_synth.return_value = MagicMock(
        synthesize_previous_steps=lambda history, current_step, **_: "",
        get_redundant_operations=lambda history: [],
        build_constraints_section=lambda redundant_ops: "",
    )

    executor = BudgetAwareExecutor(BudgetManager(2, adaptive_scaling=False))
    action_provider = MultiStepActionProvider()
    tool_invoker = StubInvoker()

    iteration_phases: list[tuple[str, bool]] = []

    def track_iteration(context, _history):
        iteration_phases.append((context.phase, context.is_final))

    task = TaskSpec(identifier="task-456", goal="Perform multi-step investigation")
    result = executor.run(
        task,
        action_provider,
        tool_invoker,
        iteration_hook=track_iteration,
    )

    assert result.status == "success"
    assert iteration_phases[-1][1] is True  # Final iteration reached
    action_provider.client.complete.assert_called_once()
    assert result.steps[-1].action.tool == "submit_final_answer"
    assert result.steps[-1].observation.raw_output == "Synthesized final answer"


@patch("ai_dev_agent.cli.react.executor.ContextSynthesizer")
def test_executor_post_loop_synthesis_in_json_mode(mock_synth):
    """Final forced synthesis should include JSON enforcement instructions."""
    mock_synth.return_value = MagicMock(
        synthesize_previous_steps=lambda *_, **__: "",
        get_redundant_operations=lambda *_: [],
        build_constraints_section=lambda *_: "",
    )

    class SingleToolProvider:
        def __init__(self):
            self.session_manager = StubSessionManager()
            self.session_id = "session-json"
            self.client = MagicMock()
            self.client.complete.return_value = '{"answer": "final"}'

        def update_phase(self, phase: str, *, is_final: bool = False) -> None:
            pass

        def __call__(self, task: TaskSpec, history):
            return {"thought": "Read file", "tool": "read", "args": {"path": "README.md"}}

        def last_response_text(self) -> str:
            return ""

    provider = SingleToolProvider()
    provider.session_manager.ensure_session(
        provider.session_id,
        system_messages=[message("system", "init")],
        metadata={},
    )

    executor = BudgetAwareExecutor(
        BudgetManager(1, adaptive_scaling=False),
        format_schema={"type": "object", "properties": {"answer": {"type": "string"}}},
    )
    tool_invoker = StubInvoker()

    task = TaskSpec(identifier="task-json-final", goal="Need answer")
    result = executor.run(task, provider, tool_invoker)

    provider.client.complete.assert_called_once()
    conversation = provider.client.complete.call_args[0][0]
    assert conversation[-1].content.count('"answer"') >= 1  # JSON enforcement appended
    assert result.steps[-1].observation.raw_output == '{"answer": "final"}'
    assert result.stop_reason in {"final_iteration", "Completed", "Forced synthesis"}


@patch("ai_dev_agent.cli.react.executor.ContextSynthesizer")
def test_executor_post_loop_synthesis_with_invoke_tools(mock_synth):
    """Post-loop forced synthesis should support invoke_tools fallback."""
    mock_synth.return_value = MagicMock(
        synthesize_previous_steps=lambda *_, **__: "",
        get_redundant_operations=lambda *_: [],
        build_constraints_section=lambda *_: "",
    )

    class SingleToolProvider:
        def __init__(self):
            self.session_manager = StubSessionManager()
            self.session_id = "session-invoke"
            self.client = SimpleNamespace(invoke_tools=self._invoke)
            self.budget_integration = DummyIntegration()

        def _invoke(self, messages, tools, temperature=0.0):
            assert tools == []
            return ToolCallResult(message_content="Synthesized via tools")

        def update_phase(self, *_, **__):
            pass

        def __call__(self, task: TaskSpec, history):
            return {"thought": "Inspect", "tool": "find", "args": {"query": "todo"}}

        def last_response_text(self) -> str:
            return ""

    provider = SingleToolProvider()
    executor = BudgetAwareExecutor(BudgetManager(1, adaptive_scaling=False))
    tool_invoker = StubInvoker()

    task = TaskSpec(identifier="task-invoke", goal="Force invocation")
    result = executor.run(task, provider, tool_invoker)

    assert result.status == "success"
    assert result.steps[-1].observation.raw_output == "Synthesized via tools"
    assert provider.budget_integration.calls


@patch("ai_dev_agent.cli.react.executor.FailurePatternDetector")
def test_invoke_tool_injects_failure_guidance(stub_failure_detector, capsys):
    """Failure detector guidance should be surfaced in observation outcome."""
    detector_instance = MagicMock()
    detector_instance.should_give_up.return_value = (True, "⚠️ Stop repeating this.")
    stub_failure_detector.return_value = detector_instance

    executor = BudgetAwareExecutor(BudgetManager(1, adaptive_scaling=False))
    action = ActionRequest(step_id="S1", thought="Try tool", tool="grep", args={"pattern": "TODO"})
    action.parameters = {"pattern": "TODO"}

    invoker = StubToolInvokerWithFailures()
    observation = executor._invoke_tool(invoker, action)

    assert invoker.calls == 1
    assert detector_instance.record_failure.called
    assert detector_instance.should_give_up.called
    assert "⚠️ Stop repeating this." in observation.outcome


def test_run_handles_budget_exhaustion_without_actions():
    """Budget exhaustion before any iteration should return a failed result gracefully."""
    manager = BudgetManager(1, adaptive_scaling=False)
    manager._current = manager.max_iterations  # Force exhaustion

    executor = BudgetAwareExecutor(manager)
    action_provider = MagicMock(side_effect=AssertionError("Action provider should not be called"))
    tool_invoker = MagicMock(side_effect=AssertionError("Tool invoker should not be called"))

    task = TaskSpec(identifier="task-789", goal="Nothing happens")
    result = executor.run(task, action_provider, tool_invoker)

    assert result.status == "failed"
    assert result.stop_reason == "No actions executed"
    action_provider.assert_not_called()
    tool_invoker.assert_not_called()


def test_executor_records_multi_tool_results_in_step():
    """Executor should carry batch tool results through to the RunResult."""
    executor = BudgetAwareExecutor(BudgetManager(1, adaptive_scaling=False))

    class BatchInvoker:
        def __call__(self, action: ActionRequest) -> Observation:
            assert action.tool_calls  # Ensure batch payload provided
            return Observation(
                success=True,
                outcome="Executed 2 tool(s): 2 succeeded",
                tool="batch[2]",
                results=[
                    ToolResult(tool="find", success=True, outcome="Found files."),
                    ToolResult(tool="grep", success=True, outcome="Matches located."),
                ],
            )

    action_provider = MagicMock(
        return_value={
            "thought": "Run two tools",
            "tool": "batch",
            "tool_calls": [
                {"tool": "find", "args": {"query": "foo"}},
                {"tool": "grep", "args": {"pattern": "TODO"}},
            ],
        }
    )
    action_provider.session_manager = StubSessionManager()
    action_provider.session_id = "session-1"
    action_provider.client = MagicMock()
    action_provider.client.complete.return_value = "Fallback synthesis"
    action_provider.last_response_text.return_value = (
        "Batch complete with enough detail to be treated as final."
    )

    task = TaskSpec(identifier="task-101", goal="Execute batch tools")
    result = executor.run(task, action_provider, BatchInvoker())

    assert result.status == "success"
    assert result.steps
    batch_step = result.steps[0]
    assert batch_step.observation.tool == "batch[2]"
    assert len(batch_step.observation.results) == 2


def test_execute_react_assistant_without_tool_support(monkeypatch, capsys):
    """When the client lacks tool support, router text should be echoed and history updated."""

    class RouterNoTools:
        def __init__(self, *_, **__):
            self.tools: list[dict[str, str]] = []
            self.session_id = "router-session"

        def route(self, prompt: str) -> IntentDecision:
            return IntentDecision(tool=None, arguments={"text": f"Echo: {prompt}"})

    monkeypatch.setattr(
        "ai_dev_agent.cli.react.executor._resolve_intent_router",
        lambda: RouterNoTools,
    )

    ctx = click.Context(click.Command("devagent"))
    ctx.meta["_emit_status_messages"] = True
    ctx.obj = {
        "_shell_conversation_history": [
            message("user", "Previous question"),
            message("assistant", "Prior answer"),
        ]
    }

    client = SimpleNamespace()
    result = _execute_react_assistant(
        ctx,
        client=client,
        settings=Settings(),
        user_prompt="Explain repo",
        use_planning=False,
        system_extension=None,
        format_schema=None,
    )

    assert result is None
    captured = capsys.readouterr()
    assert "⚡ Executing" in captured.out
    assert "✅ Completed" in captured.out
    history = ctx.obj["_shell_conversation_history"]
    assert any(msg.role == "assistant" and "Echo" in msg.content for msg in history)


def test_execute_react_assistant_tool_flow(monkeypatch, capsys):
    """Exercise the main tool-enabled execution path with stubbed dependencies."""

    session_manager = StubSessionManager()

    class StubTracker:
        def __init__(self, repo_root):
            self.repo_root = repo_root
            self.updated = []

        def update_from_step(self, record):
            self.updated.append(record.action.tool)

        def should_refresh_repomap(self):
            return bool(self.updated)

        def get_context_summary(self):
            return {"total_steps": len(self.updated)}

    class StubSynthesizer:
        def synthesize_previous_steps(self, *_, **__):
            return "Prior findings"

        def get_redundant_operations(self, *_):
            return []

        def build_constraints_section(self, *_):
            return "Constraints"

    class FakeActionProvider:
        def __init__(
            self,
            *,
            llm_client,
            session_manager,
            session_id,
            tools,
            budget_integration,
            format_schema,
            ctx_obj,
        ):
            self.session_manager = session_manager
            self.session_id = session_id
            self.client = llm_client
            self.budget_integration = budget_integration
            self.format_schema = format_schema
            self.ctx_obj = ctx_obj
            self.tools = tools
            self._last_text = "Provider fallback answer"
            self.phase_updates: list[tuple[str, bool]] = []

        def update_phase(self, phase: str, *, is_final: bool = False) -> None:
            self.phase_updates.append((phase, is_final))

        def __call__(self, task: TaskSpec, history):
            return {"thought": "Inspect", "tool": "find", "args": {"query": "tests"}}

        def last_response_text(self) -> str:
            return self._last_text

    class FakeToolInvoker:
        def __init__(self, *_, **__):
            pass

    class FakeExecutor:
        def __init__(self, *_, **__):
            pass

        def run(self, task, action_provider, tool_invoker, *, iteration_hook=None, step_hook=None):
            first_ctx = manager_context(
                is_final=False, is_penultimate=False, phase="exploration", number=1, total=2
            )
            if iteration_hook:
                iteration_hook(first_ctx, [])

            first_record = StepRecord(
                action=ActionRequest(
                    step_id="S1", thought="Inspect", tool="find", args={"query": "tests"}
                ),
                observation=Observation(
                    success=True,
                    outcome="Found files",
                    tool="find",
                    raw_output="tests/unit",
                    metrics={"files": ["tests/unit"]},
                    artifacts=["tests/unit"],
                ),
                metrics=MetricsSnapshot(),
                evaluation=EvaluationResult(gates={}, should_stop=False, status="in_progress"),
                step_index=1,
            )
            if step_hook:
                step_hook(first_record, first_ctx)

            final_ctx = manager_context(
                is_final=True, is_penultimate=False, phase="synthesis", number=2, total=2
            )
            if iteration_hook:
                iteration_hook(final_ctx, [first_record])

            json_answer = '{"summary": "Mission accomplished"}'
            final_record = StepRecord(
                action=ActionRequest(
                    step_id="S2",
                    thought="Conclude",
                    tool="submit_final_answer",
                    args={"answer": json_answer},
                ),
                observation=Observation(
                    success=True,
                    outcome="Synthesis complete",
                    tool="submit_final_answer",
                    raw_output=json_answer,
                ),
                metrics=MetricsSnapshot(),
                evaluation=EvaluationResult(
                    gates={}, should_stop=True, stop_reason="Completed", status="success"
                ),
                step_index=2,
            )
            if step_hook:
                step_hook(final_record, final_ctx)

            return RunResult(
                task_id=task.identifier,
                status="success",
                steps=[first_record, final_record],
                gates={},
                required_gates={},
                stop_reason="Completed",
                runtime_seconds=1.0,
                metrics={"diff_lines": 0},
            )

    monkeypatch.setattr("ai_dev_agent.cli.react.executor.DynamicContextTracker", StubTracker)
    monkeypatch.setattr("ai_dev_agent.cli.react.executor.ContextSynthesizer", StubSynthesizer)
    monkeypatch.setattr(
        "ai_dev_agent.cli.react.executor.SessionManager.get_instance",
        lambda: session_manager,
    )
    monkeypatch.setattr(
        "ai_dev_agent.cli.react.executor.AgentRegistry.get",
        lambda _agent: SimpleNamespace(system_prompt_suffix="", max_iterations=0),
    )
    monkeypatch.setattr(
        "ai_dev_agent.cli.react.executor.LLMActionProvider",
        lambda **kwargs: FakeActionProvider(**kwargs),
    )
    monkeypatch.setattr(
        "ai_dev_agent.cli.react.executor.SessionAwareToolInvoker",
        lambda **kwargs: FakeToolInvoker(),
    )
    monkeypatch.setattr(
        "ai_dev_agent.cli.react.executor.BudgetAwareExecutor",
        lambda *args, **kwargs: FakeExecutor(),
    )
    monkeypatch.setattr(
        "ai_dev_agent.cli.react.executor._collect_project_structure_outline",
        lambda _: "project structure summary",
    )
    monkeypatch.setattr(
        "ai_dev_agent.cli.react.executor._detect_repository_language",
        lambda _repo, settings=None: ("python", 5),
    )
    monkeypatch.setattr(
        "ai_dev_agent.cli.react.executor._get_structure_hints_state",
        lambda ctx: ctx.obj.setdefault("structure_state", {}),
    )
    monkeypatch.setattr(
        "ai_dev_agent.cli.react.executor._merge_structure_hints_state",
        lambda ctx, state: None,
    )
    monkeypatch.setattr(
        "ai_dev_agent.cli.react.executor._update_files_discovered",
        lambda files, metrics: files.update(set(metrics.get("files", []))),
    )
    monkeypatch.setattr(
        "ai_dev_agent.cli.react.executor.load_devagent_yaml",
        lambda: SimpleNamespace(
            react_iteration_global_cap=2,
            react_iteration_phase_thresholds=None,
            react_iteration_warning_thresholds=None,
        ),
    )

    class RouterWithTools:
        def __init__(self, *_args, **_kwargs):
            self.tools = [{"type": "function", "function": {"name": "find", "description": "Find"}}]
            self.session_id = "router-session"

    monkeypatch.setattr(
        "ai_dev_agent.cli.react.executor._resolve_intent_router",
        lambda: RouterWithTools,
    )

    ctx = click.Context(click.Command("devagent"))
    ctx.meta["_emit_status_messages"] = True
    ctx.obj = {
        "_shell_conversation_history": [message("user", "prior"), message("assistant", "reply")],
        "_project_structure_summary": None,
    }

    client = SimpleNamespace(
        invoke_tools=lambda *_, **__: None,
        complete=lambda *_, **__: "unused",
    )

    schema = {"type": "object", "properties": {"summary": {"type": "string"}}}
    outcome = _execute_react_assistant(
        ctx,
        client=client,
        settings=Settings(),
        user_prompt="Summarize repository",
        use_planning=False,
        system_extension="Be concise",
        format_schema=schema,
    )

    assert outcome["result"].status == "success"
    assert outcome["final_json"] == {"summary": "Mission accomplished"}
    captured = capsys.readouterr().out
    assert "summary" in captured
    assert ctx.obj["search_queries"] == ["tests"]
    tracker = ctx.obj["_dynamic_context"]
    assert getattr(tracker, "updated", [])


@patch("ai_dev_agent.cli.react.plan_executor.execute_with_planning")
def test_execute_react_assistant_delegates_to_planning(mock_plan):
    """Planning mode should call into the dedicated plan executor."""

    mock_plan.return_value = {"result": "planned"}
    ctx = click.Context(click.Command("devagent"))

    client = SimpleNamespace(invoke_tools=lambda *_, **__: None)
    outcome = _execute_react_assistant(
        ctx,
        client=client,
        settings=Settings(),
        user_prompt="Plan this task",
        use_planning=True,
        system_extension=None,
        format_schema=None,
    )

    assert outcome == {"result": "planned"}
    mock_plan.assert_called_once()


def test_execute_react_assistant_fallback_last_response(monkeypatch, capsys):
    """When no submit_final_answer is present, use the action provider's last response."""

    session_manager = StubSessionManager()

    class StubTracker:
        def __init__(self, *_):
            self.updated = []

        def update_from_step(self, record):
            self.updated.append(record.action.tool)

        def should_refresh_repomap(self):
            return False

        def get_context_summary(self):
            return {"total_steps": len(self.updated)}

    class StubSynthesizer:
        def synthesize_previous_steps(self, *_, **__):
            return "Summary"

        def get_redundant_operations(self, *_):
            return []

        def build_constraints_section(self, *_):
            return "Constraints"

    fallback_text = (
        "Detailed results across services confirming stability and pointing to next steps."
    )

    class FallbackActionProvider:
        def __init__(
            self,
            *,
            llm_client,
            session_manager,
            session_id,
            tools,
            budget_integration,
            format_schema,
            ctx_obj,
        ):
            self.session_manager = session_manager
            self.session_id = session_id
            self.client = llm_client
            self.tools = tools
            self.budget_integration = budget_integration
            self.format_schema = format_schema
            self.ctx_obj = ctx_obj
            self._last_text = fallback_text

        def update_phase(self, *_, **__):
            pass

        def __call__(self, task: TaskSpec, history):
            return {"thought": "Inspect", "tool": "find", "args": {"query": "tests"}}

        def last_response_text(self) -> str:
            return self._last_text

    class FallbackToolInvoker:
        def __init__(self, *_, **__):
            pass

    class FallbackExecutor:
        def __init__(self, *_, **__):
            pass

        def run(self, task, action_provider, tool_invoker, *, iteration_hook=None, step_hook=None):
            first_ctx = manager_context(
                is_final=False, is_penultimate=False, phase="exploration", number=1, total=1
            )
            if iteration_hook:
                iteration_hook(first_ctx, [])

            record = StepRecord(
                action=ActionRequest(
                    step_id="S1", thought="Inspect", tool="find", args={"query": "tests"}
                ),
                observation=Observation(
                    success=True,
                    outcome="Scanned repository",
                    tool="find",
                    raw_output="Trace",
                ),
                metrics=MetricsSnapshot(),
                evaluation=EvaluationResult(
                    gates={}, should_stop=True, stop_reason="Completed", status="success"
                ),
                step_index=1,
            )
            if step_hook:
                step_hook(record, first_ctx)

            return RunResult(
                task_id=task.identifier,
                status="success",
                steps=[record],
                gates={},
                required_gates={},
                stop_reason="Completed",
                runtime_seconds=0.5,
                metrics={"diff_lines": 0},
            )

    monkeypatch.setattr("ai_dev_agent.cli.react.executor.DynamicContextTracker", StubTracker)
    monkeypatch.setattr("ai_dev_agent.cli.react.executor.ContextSynthesizer", StubSynthesizer)
    monkeypatch.setattr(
        "ai_dev_agent.cli.react.executor.SessionManager.get_instance",
        lambda: session_manager,
    )
    monkeypatch.setattr(
        "ai_dev_agent.cli.react.executor.AgentRegistry.get",
        lambda _agent: SimpleNamespace(system_prompt_suffix="", max_iterations=0),
    )
    monkeypatch.setattr(
        "ai_dev_agent.cli.react.executor.LLMActionProvider",
        lambda **kwargs: FallbackActionProvider(**kwargs),
    )
    monkeypatch.setattr(
        "ai_dev_agent.cli.react.executor.SessionAwareToolInvoker",
        lambda **kwargs: FallbackToolInvoker(),
    )
    monkeypatch.setattr(
        "ai_dev_agent.cli.react.executor.BudgetAwareExecutor",
        lambda *args, **kwargs: FallbackExecutor(),
    )
    monkeypatch.setattr(
        "ai_dev_agent.cli.react.executor._collect_project_structure_outline",
        lambda *_: "summary",
    )
    monkeypatch.setattr(
        "ai_dev_agent.cli.react.executor._detect_repository_language",
        lambda _repo, settings=None: ("python", 10),
    )
    monkeypatch.setattr(
        "ai_dev_agent.cli.react.executor._get_structure_hints_state",
        lambda ctx: ctx.obj.setdefault("structure_state", {}),
    )
    monkeypatch.setattr(
        "ai_dev_agent.cli.react.executor._merge_structure_hints_state",
        lambda ctx, state: None,
    )
    monkeypatch.setattr(
        "ai_dev_agent.cli.react.executor._update_files_discovered",
        lambda files, metrics: None,
    )
    monkeypatch.setattr(
        "ai_dev_agent.cli.react.executor.load_devagent_yaml",
        lambda: SimpleNamespace(
            react_iteration_global_cap=1,
            react_iteration_phase_thresholds=None,
            react_iteration_warning_thresholds=None,
        ),
    )
    monkeypatch.setattr(
        "ai_dev_agent.cli.react.executor._resolve_intent_router",
        lambda: (lambda *args, **kwargs: SimpleNamespace(tools=[], session_id="router")),
    )

    ctx = click.Context(click.Command("devagent"))
    ctx.meta["_emit_status_messages"] = False
    ctx.obj = {"_shell_conversation_history": []}

    client = SimpleNamespace(invoke_tools=lambda *_, **__: None, complete=lambda *_, **__: "unused")

    outcome = _execute_react_assistant(
        ctx,
        client=client,
        settings=Settings(),
        user_prompt="Document findings",
        use_planning=False,
        system_extension=None,
        format_schema=None,
    )

    output = capsys.readouterr().out
    assert outcome["final_message"] == fallback_text
    assert fallback_text in output


@patch("ai_dev_agent.cli.react.executor.SessionManager")
@patch("ai_dev_agent.cli.react.executor.ContextSynthesizer")
def test_executor_forced_synthesis_missing_methods_logs_error(
    mock_synth, mock_session_manager, capsys
):
    mock_session_manager.get_instance.return_value = StubSessionManager()
    mock_synth.return_value = MagicMock(
        synthesize_previous_steps=lambda *_, **__: "",
        get_redundant_operations=lambda *_: [],
        build_constraints_section=lambda *_: "",
    )

    executor = BudgetAwareExecutor(BudgetManager(1, adaptive_scaling=False))
    action_provider = StubActionProviderWithStopIteration("")
    action_provider.client = object()

    result = executor.run(
        TaskSpec(identifier="task-missing", goal="answer"), action_provider, StubInvoker()
    )
    assert result.status == "success"

    captured = capsys.readouterr()
    assert "does not support forced synthesis" in captured.err


@patch("ai_dev_agent.cli.react.executor.SessionManager")
@patch("ai_dev_agent.cli.react.executor.ContextSynthesizer")
def test_executor_forced_synthesis_empty_response_warnings(
    mock_synth, mock_session_manager, capsys
):
    mock_session_manager.get_instance.return_value = StubSessionManager()
    mock_synth.return_value = MagicMock(
        synthesize_previous_steps=lambda *_, **__: "",
        get_redundant_operations=lambda *_: [],
        build_constraints_section=lambda *_: "",
    )

    executor = BudgetAwareExecutor(BudgetManager(1, adaptive_scaling=False))
    action_provider = StubActionProviderWithStopIteration("")
    empty_client = MagicMock()
    empty_client.complete.return_value = "   "
    action_provider.client = empty_client

    executor.run(TaskSpec(identifier="task-empty", goal="answer"), action_provider, StubInvoker())
    captured = capsys.readouterr()
    assert "LLM returned empty response" in captured.err


def test_executor_failed_observation_sets_stop_reason():
    executor = BudgetAwareExecutor(BudgetManager(1, adaptive_scaling=False))
    action_provider = FailingActionProvider()
    failing_invoker = StubToolInvokerWithFailures()

    result = executor.run(
        TaskSpec(identifier="task-failure", goal="investigate"), action_provider, failing_invoker
    )
    assert result.status == "failed"
    assert "Tool failed" in result.stop_reason


def test_executor_budget_exhaustion_completes_when_final_success():
    executor = BudgetAwareExecutor(BudgetManager(1, adaptive_scaling=False))
    action_provider = StubActionProvider()
    tool_invoker = StubInvoker()

    result = executor.run(
        TaskSpec(identifier="task-success", goal="complete"), action_provider, tool_invoker
    )
    assert result.status == "success"
    assert result.stop_reason == "Completed"


@patch("ai_dev_agent.cli.react.executor.SessionManager")
@patch("ai_dev_agent.cli.react.executor.ContextSynthesizer")
def test_executor_forced_synthesis_invoke_tools_handles_string_result(
    mock_synth, mock_session_manager
):
    mock_session_manager.get_instance.return_value = StubSessionManager()
    mock_synth.return_value = MagicMock(
        synthesize_previous_steps=lambda *_, **__: "",
        get_redundant_operations=lambda *_: [],
        build_constraints_section=lambda *_: "",
    )

    executor = BudgetAwareExecutor(BudgetManager(1, adaptive_scaling=False))
    action_provider = StubActionProviderWithStopIteration("")

    class InvokeToolsClient:
        def invoke_tools(self, messages, tools, temperature=0.1):
            return "forced-string"

    action_provider.client = InvokeToolsClient()

    result = executor.run(
        TaskSpec(identifier="task-string", goal="force string"), action_provider, StubInvoker()
    )
    assert result.status == "success"
    assert result.steps[-1].observation.raw_output == "forced-string"
