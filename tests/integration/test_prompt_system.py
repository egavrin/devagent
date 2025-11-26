"""Integration tests for the prompt system.

Tests the full flow from prompt loading through rendering to consumption
by components like the planner and session builder.
"""

from pathlib import Path

import pytest

from ai_dev_agent.prompts.loader import PromptLoader
from ai_dev_agent.prompts.templates import TemplateEngine
from ai_dev_agent.session.prompt_builder import build_system_messages


class TestPromptBuilderIntegration:
    """Test build_system_messages produces valid prompts."""

    def test_build_system_messages_returns_messages(self) -> None:
        """build_system_messages should return at least one message."""
        messages = build_system_messages()

        assert len(messages) > 0
        assert all(hasattr(m, "role") and hasattr(m, "content") for m in messages)

    def test_build_system_messages_no_unsubstituted_placeholders(self) -> None:
        """Rendered messages should have no {{PLACEHOLDER}} remaining."""
        messages = build_system_messages(
            iteration_cap=10,
            repository_language="python",
        )

        for msg in messages:
            assert (
                "{{" not in msg.content
            ), f"Unsubstituted placeholder in message: {msg.content[:100]}..."

    def test_build_system_messages_with_iteration_cap(self) -> None:
        """Iteration cap should be included in the output."""
        messages = build_system_messages(iteration_cap=5)

        # At least one message should mention the iteration limit
        combined = " ".join(m.content for m in messages)
        assert "5" in combined or "five" in combined.lower()

    def test_build_system_messages_with_language_hint(self) -> None:
        """Language hint should be included when language is specified."""
        messages = build_system_messages(repository_language="python")

        combined = " ".join(m.content for m in messages)
        # Should contain python-specific guidance
        assert "python" in combined.lower() or "Python" in combined

    def test_build_system_messages_without_react_guidance(self) -> None:
        """Should be able to exclude react guidance."""
        with_react = build_system_messages(include_react_guidance=True)
        without_react = build_system_messages(include_react_guidance=False)

        # Without react should have less content
        with_content = " ".join(m.content for m in with_react)
        without_content = " ".join(m.content for m in without_react)

        # React guidance adds significant content
        assert len(with_content) >= len(without_content)


class TestPromptLoaderIntegration:
    """Test PromptLoader with actual prompt files."""

    @pytest.fixture
    def loader(self) -> PromptLoader:
        return PromptLoader()

    def test_load_system_prompt_returns_content(self, loader: PromptLoader) -> None:
        """Loading a system prompt should return non-empty content."""
        content = loader.load_system_prompt("base_context")
        assert content
        assert len(content) > 100  # Should be substantial

    def test_load_agent_prompt_returns_content(self, loader: PromptLoader) -> None:
        """Loading an agent prompt should return non-empty content."""
        context = {
            "TASK": "test task",
            "CONTEXT": "test context",
            "REPO_CONTEXT": "repo info",
        }
        content = loader.load_agent_prompt("design", context=context)
        assert content
        assert "test task" in content

    def test_render_prompt_resolves_placeholders(self, loader: PromptLoader) -> None:
        """render_prompt should resolve all provided placeholders."""
        context = {
            "GOAL": "Build a REST API",
            "CONTEXT_BLOCK": "Using FastAPI framework",
        }

        rendered = loader.render_prompt("system/planner_user.md", context=context)

        assert "Build a REST API" in rendered
        assert "FastAPI framework" in rendered
        assert "{{GOAL}}" not in rendered
        assert "{{CONTEXT_BLOCK}}" not in rendered

    def test_compose_prompt_combines_multiple(self, loader: PromptLoader) -> None:
        """compose_prompt should combine multiple prompts."""
        parts = [
            "system/react_loop.md",
            ("system/planner_user.md", {"GOAL": "Test", "CONTEXT_BLOCK": "Info"}),
        ]

        composed = loader.compose_prompt(parts)

        # Should contain content from both prompts
        assert "ReAct" in composed or "React" in composed or "Observe" in composed
        assert "Test" in composed


class TestTemplateEngineIntegration:
    """Test TemplateEngine with actual prompts."""

    @pytest.fixture
    def engine(self) -> TemplateEngine:
        return TemplateEngine()

    @pytest.fixture
    def loader(self) -> PromptLoader:
        return PromptLoader()

    def test_resolve_all_prompts_with_valid_context(
        self,
        engine: TemplateEngine,
        loader: PromptLoader,
    ) -> None:
        """All prompts should resolve when given proper context."""
        # Load base_context with full context
        template = loader.load_prompt("system/base_context.md")

        context = {
            "TOOL_READ": "read",
            "TOOL_EDIT": "edit",
            "TOOL_RUN": "run",
            "TOOL_FIND": "find",
            "TOOL_GREP": "grep",
            "TOOL_SYMBOLS": "symbols",
            "ITERATION_NOTE": "Limited to 10 steps.",
            "LANGUAGE_HINT": "Python project hints here.",
        }

        rendered = engine.resolve(template, context, strict=True)

        assert "{{" not in rendered
        assert "read" in rendered
        assert "Limited to 10 steps" in rendered

    def test_validation_detects_missing_required(
        self,
        engine: TemplateEngine,
        loader: PromptLoader,
    ) -> None:
        """Validation should detect missing required placeholders."""
        template = loader.load_prompt("system/planner_user.md")

        # Missing CONTEXT_BLOCK
        result = engine.validate(template, {"GOAL": "Test"})

        assert not result.is_valid
        assert "CONTEXT_BLOCK" in result.missing

    def test_extract_placeholders_from_real_prompt(
        self,
        engine: TemplateEngine,
        loader: PromptLoader,
    ) -> None:
        """Should extract all placeholders from real prompts."""
        template = loader.load_prompt("system/base_context.md")
        placeholders = engine.extract_placeholders(template)

        # Should find the tool placeholders
        expected = {"TOOL_READ", "TOOL_EDIT", "TOOL_RUN", "TOOL_FIND", "TOOL_GREP", "TOOL_SYMBOLS"}
        assert expected.issubset(placeholders)


class TestLazyLoadingIntegration:
    """Test that lazy-loaded prompts work correctly."""

    def test_planner_lazy_loading(self) -> None:
        """Planner should load prompts lazily without import-time errors."""
        # Import should succeed even if prompts aren't loaded yet
        from ai_dev_agent.engine.planning.planner import _get_prompts

        # First call loads prompts
        system_prompt, user_template = _get_prompts()

        assert system_prompt
        assert user_template
        assert "{{" not in system_prompt  # Should be raw, not pre-rendered

    def test_summarizer_lazy_loading(self) -> None:
        """Summarizer should load prompts lazily without import-time errors."""
        from ai_dev_agent.session.summarizer import _get_prompts

        system_prompt, user_template = _get_prompts()

        assert system_prompt
        assert user_template


class TestPromptConsistency:
    """Test consistency across prompt system components."""

    def test_tool_names_match_across_system(self) -> None:
        """Tool names in prompts should match actual tool_names constants."""
        from ai_dev_agent.tool_names import EDIT, FIND, GREP, READ, RUN, SYMBOLS

        loader = PromptLoader()

        context = {
            "TOOL_READ": READ,
            "TOOL_EDIT": EDIT,
            "TOOL_RUN": RUN,
            "TOOL_FIND": FIND,
            "TOOL_GREP": GREP,
            "TOOL_SYMBOLS": SYMBOLS,
            "ITERATION_NOTE": "",
            "LANGUAGE_HINT": "",
        }

        rendered = loader.render_prompt(
            "system/base_context.md",
            context=context,
            strict=False,
        )

        # Actual tool names should appear in rendered output
        assert READ in rendered
        assert EDIT in rendered
        assert RUN in rendered

    def test_prompt_builder_uses_uppercase_keys(self) -> None:
        """_system_prompt_context should return UPPERCASE keys."""
        from ai_dev_agent.session.prompt_builder import _system_prompt_context

        context = _system_prompt_context(
            iteration_cap=10,
            repository_language="python",
            settings=None,
        )

        # All keys should be uppercase
        for key in context.keys():
            assert key.isupper(), f"Key {key} should be UPPERCASE"

        # Expected keys should be present
        assert "TOOL_READ" in context
        assert "TOOL_EDIT" in context
        assert "ITERATION_CAP" in context
