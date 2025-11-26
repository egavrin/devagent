"""Tests for the TemplateEngine with {{VAR}} syntax."""

import pytest

from ai_dev_agent.prompts.templates.template_engine import TemplateEngine, ValidationResult


class TestTemplateEngine:
    """Test the {{PLACEHOLDER}} template engine."""

    @pytest.fixture
    def engine(self):
        return TemplateEngine()

    def test_resolve_simple_placeholder(self, engine):
        """Simple placeholder substitution."""
        result = engine.resolve("Hello {{NAME}}", {"NAME": "World"})
        assert result == "Hello World"

    def test_resolve_multiple_placeholders(self, engine):
        """Multiple placeholders in one template."""
        result = engine.resolve(
            "{{GREETING}} {{NAME}}!",
            {"GREETING": "Hello", "NAME": "World"},
        )
        assert result == "Hello World!"

    def test_resolve_same_placeholder_multiple_times(self, engine):
        """Same placeholder used multiple times."""
        result = engine.resolve(
            "{{NAME}} said: Hello {{NAME}}!",
            {"NAME": "Alice"},
        )
        assert result == "Alice said: Hello Alice!"

    def test_no_conflict_with_python_dict(self, engine):
        """Python dict syntax is preserved."""
        template = '{"key": "value"}\nTool: {{TOOL_NAME}}'
        result = engine.resolve(template, {"TOOL_NAME": "read"})
        assert '{"key": "value"}' in result
        assert "Tool: read" in result

    def test_no_conflict_with_fstring(self, engine):
        """F-string syntax is preserved."""
        template = 'f"{first} {last}"\n{{PLACEHOLDER}}'
        result = engine.resolve(template, {"PLACEHOLDER": "value"})
        assert 'f"{first} {last}"' in result
        assert "value" in result

    def test_no_conflict_with_json(self, engine):
        """JSON in template is preserved."""
        template = """```json
{"name": "test", "value": 123}
```
Variable: {{VAR}}"""
        result = engine.resolve(template, {"VAR": "resolved"})
        assert '{"name": "test", "value": 123}' in result
        assert "Variable: resolved" in result

    def test_no_conflict_with_single_braces(self, engine):
        """Single braces are not treated as placeholders."""
        template = "{single} {{DOUBLE}}"
        result = engine.resolve(template, {"DOUBLE": "works"})
        assert "{single}" in result
        assert "works" in result

    def test_strict_mode_raises_on_missing(self, engine):
        """Strict mode raises on missing placeholders."""
        with pytest.raises(ValueError, match="Missing"):
            engine.resolve("{{MISSING}}", {}, strict=True)

    def test_non_strict_mode_keeps_unresolved(self, engine):
        """Non-strict mode keeps unresolved placeholders."""
        result = engine.resolve("{{MISSING}}", {}, strict=False)
        assert result == "{{MISSING}}"

    def test_case_sensitive_keys(self, engine):
        """Keys are case-sensitive (must be UPPERCASE)."""
        # lowercase key won't match UPPERCASE placeholder
        with pytest.raises(ValueError, match="Missing.*NAME"):
            engine.resolve("{{NAME}}", {"name": "value"}, strict=True)

    def test_uppercase_convention(self, engine):
        """Only UPPERCASE placeholders are matched."""
        # lowercase {{name}} won't be matched
        template = "{{name}} vs {{NAME}}"
        result = engine.resolve(template, {"NAME": "UPPER"}, strict=False)
        assert "{{name}}" in result  # lowercase not matched
        assert "UPPER" in result

    def test_extract_placeholders(self, engine):
        """Extract all placeholder names."""
        template = "{{FIRST}} and {{SECOND}} and {{FIRST}} again"
        placeholders = engine.extract_placeholders(template)
        assert placeholders == {"FIRST", "SECOND"}

    def test_extract_placeholders_empty(self, engine):
        """Empty set for template without placeholders."""
        placeholders = engine.extract_placeholders("No placeholders here")
        assert placeholders == set()

    def test_validate_all_provided(self, engine):
        """Validation passes when all placeholders provided."""
        result = engine.validate("{{A}} {{B}}", {"A": 1, "B": 2})
        assert result.is_valid
        assert result.missing == set()
        assert result.unused == set()

    def test_validate_missing_placeholders(self, engine):
        """Validation detects missing placeholders."""
        result = engine.validate("{{A}} {{B}} {{C}}", {"A": 1})
        assert not result.is_valid
        assert result.missing == {"B", "C"}

    def test_validate_unused_context(self, engine):
        """Validation detects unused context keys."""
        result = engine.validate("{{A}}", {"A": 1, "B": 2, "C": 3})
        assert result.is_valid  # Still valid
        assert result.unused == {"B", "C"}

    def test_has_placeholders_true(self, engine):
        """Detect presence of placeholders."""
        assert engine.has_placeholders("{{PLACEHOLDER}}")
        assert engine.has_placeholders("text {{VAR}} more")

    def test_has_placeholders_false(self, engine):
        """Detect absence of placeholders."""
        assert not engine.has_placeholders("no placeholders")
        assert not engine.has_placeholders("{single_brace}")
        assert not engine.has_placeholders('{"json": "dict"}')


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_valid_result(self):
        """Valid result has no missing placeholders."""
        result = ValidationResult(is_valid=True)
        assert result.is_valid
        assert result.missing == set()
        assert result.unused == set()

    def test_invalid_result(self):
        """Invalid result has missing placeholders."""
        result = ValidationResult(
            is_valid=False,
            missing={"A", "B"},
            unused={"C"},
        )
        assert not result.is_valid
        assert result.missing == {"A", "B"}
        assert result.unused == {"C"}


class TestRealWorldScenarios:
    """Test real-world prompt scenarios."""

    @pytest.fixture
    def engine(self):
        return TemplateEngine()

    def test_system_prompt_with_tools(self, engine):
        """System prompt with tool name substitution."""
        template = """Use {{TOOL_READ}} to read files.
Use {{TOOL_EDIT}} to modify files.
Use {{TOOL_RUN}} to execute commands."""

        result = engine.resolve(
            template,
            {
                "TOOL_READ": "read",
                "TOOL_EDIT": "edit",
                "TOOL_RUN": "run",
            },
        )
        assert "Use read to read files" in result
        assert "Use edit to modify files" in result
        assert "Use run to execute commands" in result

    def test_code_example_preservation(self, engine):
        """Code examples in prompts are preserved."""
        template = """Example Python code:
```python
config = {"debug": True, "timeout": 30}
name = f"{first} {last}"
```

Use {{TOOL_NAME}} for this task."""

        result = engine.resolve(template, {"TOOL_NAME": "example_tool"})
        assert '{"debug": True, "timeout": 30}' in result
        assert 'f"{first} {last}"' in result
        assert "Use example_tool for this task" in result

    def test_multiline_prompt(self, engine):
        """Multi-line prompt with mixed content."""
        template = """# Task: {{TASK}}

## Context
{{CONTEXT}}

## Instructions
1. Use {{TOOL_READ}} to examine files
2. Make changes with {{TOOL_EDIT}}

```json
{"status": "pending"}
```
"""
        result = engine.resolve(
            template,
            {
                "TASK": "Implement feature",
                "CONTEXT": "Working on module X",
                "TOOL_READ": "read",
                "TOOL_EDIT": "edit",
            },
        )
        assert "# Task: Implement feature" in result
        assert "Working on module X" in result
        assert "Use read to examine files" in result
        assert '{"status": "pending"}' in result
