from ai_dev_agent.providers.llm.base import Message
from ai_dev_agent.session.context_synthesis import ContextSynthesizer
from ai_dev_agent.tools import READ, RUN, WRITE


def _tool_call(name: str, arguments: dict | None = None) -> dict:
    return {
        "function": {
            "name": name,
            "arguments": arguments or {},
        }
    }


def test_synthesize_previous_steps_initial_step() -> None:
    synthesizer = ContextSynthesizer()

    result_no_history = synthesizer.synthesize_previous_steps([], current_step=1)
    result_first_step = synthesizer.synthesize_previous_steps(
        [Message(role="assistant", content="Ready to start")],
        current_step=1,
    )

    expected = "This is your first step - no previous findings."
    assert result_no_history == expected
    assert result_first_step == expected


def test_synthesize_previous_steps_merges_recent_activity() -> None:
    diff = "--- a/ai_dev_agent/session/context_synthesis.py\n+++ b/ai_dev_agent/session/context_synthesis.py\n"
    history = [
        Message(
            role="assistant",
            tool_calls=[
                None,
                _tool_call(READ, {"paths": ["ai_dev_agent/session/context_synthesis.py"]}),
                {"function": {"name": "read_file", "arguments": ["not-a-dict"]}},
                _tool_call("search", {"query": "TODO markers"}),
            ],
        ),
        Message(
            role="assistant",
            tool_calls=[
                _tool_call(WRITE, {"diff": diff}),
                _tool_call(WRITE, {"content": "no diff provided"}),
                _tool_call("symbols", {"name": "ContextSynthesizer"}),
                _tool_call("ast_analyzer", {"path": "ai_dev_agent/session/context_synthesis.py"}),
            ],
        ),
        Message(role="tool", content="error: file not found when reading"),
        Message(
            role="tool",
            content=(
                "Discovered new invariant in merge step\n"
                "Additional detail line that is ignored\n"
            ),
        ),
        Message(role="assistant", content="We found the underlying race condition."),
    ]
    synthesizer = ContextSynthesizer()

    summary = synthesizer.synthesize_previous_steps(history, current_step=4)

    assert "Files modified: ai_dev_agent/session/context_synthesis.py" in summary
    assert "Files examined: ai_dev_agent/session/context_synthesis.py" in summary
    assert "Searches performed: TODO markers" in summary
    assert "Symbols looked up: ContextSynthesizer" in summary
    assert "Errors encountered: 1" in summary
    assert "Key discoveries:" in summary
    assert "• Discovered new invariant in merge step" in summary
    assert "Previous step result: We found the underlying race condition" in summary


def test_synthesize_previous_steps_handles_partial_tool_data() -> None:
    diff = "--- a\n+++ b\n"
    history = [
        Message(
            role="assistant",
            tool_calls=[
                _tool_call(READ, {"file_path": "docs/CHANGELOG.md"}),
                _tool_call(WRITE, {"diff": diff}),
                {"function": {"name": "search", "arguments": {"pattern": "fixme"}}},
                {"function": {"name": "symbols", "arguments": {}}},
                {"function": {"name": "ast", "arguments": {"path": ""}}},
            ],
        ),
        Message(role="assistant", content="No new information available."),
    ]
    synthesizer = ContextSynthesizer()

    summary = synthesizer.synthesize_previous_steps(history, current_step=3)

    assert "Files examined: docs/CHANGELOG.md" in summary
    assert "Files modified: " not in summary
    assert "Searches performed: fixme" in summary
    assert "Key discoveries:" not in summary


def test_synthesize_previous_steps_defaults_to_exploratory_message() -> None:
    history = [Message(role="assistant", content="No concrete findings yet.")]
    synthesizer = ContextSynthesizer()

    summary = synthesizer.synthesize_previous_steps(history, current_step=3)

    assert summary == "Completed 2 exploratory steps"


def test_synthesize_previous_steps_truncates_when_exceeding_limit() -> None:
    history = [
        Message(role="assistant", tool_calls=[_tool_call("search", {"query": "long pattern"})]),
        Message(role="tool", content="Found a lengthy explanation of the discovered issue"),
        Message(role="tool", content="Found another detailed discovery in the system"),
        Message(role="assistant", content="We found that the configuration is inconsistent."),
    ]
    synthesizer = ContextSynthesizer(max_context_chars=80)

    summary = synthesizer.synthesize_previous_steps(history, current_step=5)

    suffix = "\n... (context truncated)"
    assert summary.endswith("(context truncated)")
    assert len(summary) <= synthesizer.max_context_chars - 20 + len(suffix)


def test_get_redundant_operations_collects_targets() -> None:
    history = [
        Message(
            role="assistant",
            tool_calls=[
                None,
                _tool_call(READ, {"file_path": "README.md"}),
                _tool_call("read_file", {"file_path": "docs/CHANGELOG.md"}),
                _tool_call("search", {"pattern": "TODO"}),
                _tool_call(RUN, {"command": "pytest --maxfail=1"}),
                _tool_call("exec_runner", {"cmd": "ls"}),
            ],
        )
    ]
    synthesizer = ContextSynthesizer()

    redundant = synthesizer.get_redundant_operations(history)

    assert redundant == {
        "files_read": {"README.md", "docs/CHANGELOG.md"},
        "searches_done": {"TODO"},
        "commands_run": {"pytest --maxfail=1", "ls"},
    }


def test_build_constraints_section_formats_bullets() -> None:
    synthesizer = ContextSynthesizer()

    constraints = synthesizer.build_constraints_section(
        {
            "files_read": {"docs/CHANGELOG.md", "README.md"},
            "searches_done": {"TODO"},
        }
    )

    lines = constraints.splitlines()
    assert lines[0].startswith("• Already examined files:")
    assert "avoid re-reading" in lines[0]
    assert lines[1] == "• Already searched for: TODO (use different patterns)"

    empty_constraints = synthesizer.build_constraints_section({})
    assert empty_constraints == "• No redundant operations detected"
