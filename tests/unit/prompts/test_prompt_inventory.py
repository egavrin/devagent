"""Tests for prompt inventory validation.

Validates all prompt files in the codebase:
- All prompts load successfully
- Placeholder syntax is correct ({{VAR}})
- Required placeholders are documented in the manifest
- No stale placeholder syntax ({var}) remains
"""

from pathlib import Path

import pytest

from ai_dev_agent.prompts.loader import PromptLoader
from ai_dev_agent.prompts.templates import TemplateEngine
from ai_dev_agent.prompts.validation import (
    PROMPT_VARIABLE_MANIFEST,
    extract_placeholders,
    get_optional_variables,
    get_required_variables,
)


@pytest.fixture
def prompt_loader() -> PromptLoader:
    """Return a PromptLoader using the actual prompts directory."""
    return PromptLoader()


@pytest.fixture
def template_engine() -> TemplateEngine:
    """Return a TemplateEngine instance."""
    return TemplateEngine()


@pytest.fixture
def all_prompt_files(prompt_loader: PromptLoader) -> list[str]:
    """Return all prompt files in the prompts directory."""
    return prompt_loader.list_prompts()


class TestPromptInventory:
    """Test that all prompts in the codebase are valid."""

    def test_all_prompts_load_successfully(
        self,
        prompt_loader: PromptLoader,
        all_prompt_files: list[str],
    ) -> None:
        """Every prompt file should load without errors."""
        for prompt_path in all_prompt_files:
            content = prompt_loader.load_prompt(prompt_path)
            assert content, f"Prompt {prompt_path} returned empty content"

    def test_no_stale_single_brace_placeholders(
        self,
        prompt_loader: PromptLoader,
        all_prompt_files: list[str],
    ) -> None:
        """No prompts should use the old {var} syntax (outside code examples)."""
        import re

        # Pattern for old-style {placeholder} that isn't inside code blocks
        # or part of a Python f-string pattern
        old_style_pattern = re.compile(r"(?<!{)\{([a-z][a-z0-9_]*)\}(?!})")

        for prompt_path in all_prompt_files:
            content = prompt_loader.load_prompt(prompt_path)

            # Skip code blocks when checking for old-style placeholders
            # Remove fenced code blocks (```) and inline code (`) before checking
            code_block_pattern = re.compile(r"```[\s\S]*?```")
            inline_code_pattern = re.compile(r"`[^`]+`")
            content_no_code = code_block_pattern.sub("", content)
            content_no_code = inline_code_pattern.sub("", content_no_code)

            matches = old_style_pattern.findall(content_no_code)

            # Filter out known false positives:
            # - {conversation}, {max_chars} in user templates (lowercase intentional for .format())
            # - Common variable-like words that appear in prose
            known_false_positives = {
                "conversation",
                "max_chars",  # Template .format() vars
                "variable",
                "variables",  # Prose references
                "function",
                "method",
                "class",  # Programming terms
                "name",
                "value",
                "key",
                "item",  # Generic words
            }
            unexpected = [m for m in matches if m not in known_false_positives]

            assert not unexpected, (
                f"Prompt {prompt_path} uses old-style {{var}} syntax: {unexpected}. "
                "Migrate to {{VAR}} syntax."
            )

    def test_placeholder_syntax_is_uppercase(
        self,
        prompt_loader: PromptLoader,
        all_prompt_files: list[str],
        template_engine: TemplateEngine,
    ) -> None:
        """All {{PLACEHOLDER}} names should be UPPERCASE."""
        for prompt_path in all_prompt_files:
            content = prompt_loader.load_prompt(prompt_path)
            placeholders = template_engine.extract_placeholders(content)

            for placeholder in placeholders:
                assert placeholder.isupper(), (
                    f"Placeholder {{{{{placeholder}}}}} in {prompt_path} " "should be UPPERCASE"
                )


class TestPromptVariableManifest:
    """Test the prompt variable manifest for completeness."""

    def test_manifest_covers_all_prompts_with_placeholders(
        self,
        prompt_loader: PromptLoader,
        all_prompt_files: list[str],
        template_engine: TemplateEngine,
    ) -> None:
        """All prompts with placeholders should be in the manifest."""
        missing_from_manifest = []

        for prompt_path in all_prompt_files:
            content = prompt_loader.load_prompt(prompt_path)
            placeholders = template_engine.extract_placeholders(content)

            if placeholders and prompt_path not in PROMPT_VARIABLE_MANIFEST:
                missing_from_manifest.append((prompt_path, placeholders))

        if missing_from_manifest:
            details = "\n".join(f"  - {path}: {vars}" for path, vars in missing_from_manifest)
            pytest.fail(
                f"Prompts with placeholders not in manifest:\n{details}\n"
                "Add these to PROMPT_VARIABLE_MANIFEST in validation.py"
            )

    def test_manifest_variables_match_actual_placeholders(
        self,
        prompt_loader: PromptLoader,
        template_engine: TemplateEngine,
    ) -> None:
        """Manifest entries should match actual placeholders in prompts."""
        for prompt_path, manifest_vars in PROMPT_VARIABLE_MANIFEST.items():
            try:
                content = prompt_loader.load_prompt(prompt_path)
            except FileNotFoundError:
                pytest.fail(f"Manifest references non-existent prompt: {prompt_path}")
                continue

            actual_placeholders = template_engine.extract_placeholders(content)
            manifest_names = set(manifest_vars.keys())

            # Check for missing in manifest
            missing = actual_placeholders - manifest_names
            if missing:
                pytest.fail(f"Prompt {prompt_path} has placeholders not in manifest: {missing}")

            # Check for extra in manifest (stale entries)
            extra = manifest_names - actual_placeholders
            if extra:
                pytest.fail(f"Manifest has stale entries for {prompt_path}: {extra}")

    def test_get_required_variables_returns_correct_set(self) -> None:
        """get_required_variables should return only required vars."""
        required = get_required_variables("system/base_context.md")

        assert "TOOL_READ" in required
        assert "TOOL_EDIT" in required
        assert "ITERATION_NOTE" not in required  # Optional

    def test_get_optional_variables_returns_correct_set(self) -> None:
        """get_optional_variables should return only optional vars."""
        optional = get_optional_variables("system/base_context.md")

        assert "ITERATION_NOTE" in optional
        assert "LANGUAGE_HINT" in optional
        assert "TOOL_READ" not in optional  # Required

    def test_unknown_prompt_returns_empty_sets(self) -> None:
        """Unknown prompts should return empty sets."""
        assert get_required_variables("nonexistent/prompt.md") == set()
        assert get_optional_variables("nonexistent/prompt.md") == set()


class TestPromptRenderingWithContext:
    """Test that prompts can be rendered with expected context."""

    def test_base_context_renders_with_tool_names(
        self,
        prompt_loader: PromptLoader,
    ) -> None:
        """base_context.md should render with tool name context."""
        context = {
            "TOOL_READ": "read",
            "TOOL_EDIT": "edit",
            "TOOL_RUN": "run",
            "TOOL_FIND": "find",
            "TOOL_GREP": "grep",
            "TOOL_SYMBOLS": "symbols",
            "ITERATION_NOTE": "",
            "LANGUAGE_HINT": "",
        }

        rendered = prompt_loader.render_prompt(
            "system/base_context.md",
            context=context,
            strict=False,
        )

        assert "{{" not in rendered, "Unsubstituted placeholders remain"
        assert "read" in rendered
        assert "edit" in rendered

    def test_react_loop_renders_with_optional_iteration_note(
        self,
        prompt_loader: PromptLoader,
    ) -> None:
        """react_loop.md should render with or without iteration note."""
        # With iteration note
        rendered_with = prompt_loader.render_prompt(
            "system/react_loop.md",
            context={"ITERATION_NOTE": "Limited to 10 iterations."},
            strict=False,
        )
        assert "Limited to 10 iterations" in rendered_with

        # Without iteration note (empty string)
        rendered_without = prompt_loader.render_prompt(
            "system/react_loop.md",
            context={"ITERATION_NOTE": ""},
            strict=False,
        )
        assert "{{" not in rendered_without

    def test_planner_user_renders_with_goal_and_context(
        self,
        prompt_loader: PromptLoader,
    ) -> None:
        """planner_user.md should render with goal and context block."""
        context = {
            "GOAL": "Implement user authentication",
            "CONTEXT_BLOCK": "Using OAuth2 with Google provider",
        }

        rendered = prompt_loader.render_prompt(
            "system/planner_user.md",
            context=context,
        )

        assert "Implement user authentication" in rendered
        assert "OAuth2 with Google provider" in rendered
        assert "{{" not in rendered


class TestPromptCategoryOrganization:
    """Test that prompts are properly organized by category."""

    def test_system_prompts_exist(
        self,
        prompt_loader: PromptLoader,
    ) -> None:
        """Expected system prompts should exist."""
        system_prompts = prompt_loader.list_prompts(category="system")

        expected = [
            "system/base_context.md",
            "system/react_loop.md",
        ]

        for expected_prompt in expected:
            assert (
                expected_prompt in system_prompts
            ), f"Missing expected system prompt: {expected_prompt}"

    def test_agent_prompts_exist(
        self,
        prompt_loader: PromptLoader,
    ) -> None:
        """Expected agent prompts should exist."""
        agent_prompts = prompt_loader.list_prompts(category="agents")

        expected = [
            "agents/design.md",
            "agents/implementation.md",
            "agents/test.md",
            "agents/review.md",
        ]

        for expected_prompt in expected:
            assert (
                expected_prompt in agent_prompts
            ), f"Missing expected agent prompt: {expected_prompt}"

    def test_format_prompts_exist(
        self,
        prompt_loader: PromptLoader,
    ) -> None:
        """Expected format prompts should exist."""
        format_prompts = prompt_loader.list_prompts(category="formats")

        # edit_block.md is a known format prompt
        assert any("edit_block" in p for p in format_prompts), "Missing edit_block format prompt"
