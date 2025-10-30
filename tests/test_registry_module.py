"""Tests for the tool registry module."""

import json
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from jsonschema import Draft7Validator

from ai_dev_agent.tools.registry import (
    ToolContext,
    ToolRegistry,
    ToolSpec,
    _load_validator,
    registry,
)


class TestToolContext:
    """Test ToolContext dataclass."""

    def test_tool_context_creation(self):
        """Test creating a ToolContext."""
        repo_root = Path("/test/repo")
        settings = {"key": "value"}
        sandbox = MagicMock()

        context = ToolContext(repo_root=repo_root, settings=settings, sandbox=sandbox)

        assert context.repo_root == repo_root
        assert context.settings == settings
        assert context.sandbox == sandbox
        assert context.devagent_config is None
        assert context.metrics_collector is None
        assert context.extra is None

    def test_tool_context_with_extras(self):
        """Test ToolContext with optional fields."""
        context = ToolContext(
            repo_root=Path("/test"),
            settings={},
            sandbox=None,
            devagent_config={"config": "value"},
            metrics_collector=MagicMock(),
            extra={"extra": "data"},
        )

        assert context.devagent_config == {"config": "value"}
        assert context.metrics_collector is not None
        assert context.extra == {"extra": "data"}


class TestToolSpec:
    """Test ToolSpec dataclass."""

    def test_tool_spec_minimal(self):
        """Test creating minimal ToolSpec."""

        def handler(payload, context):
            return {"result": "ok"}

        spec = ToolSpec(
            name="test_tool", handler=handler, request_schema_path=None, response_schema_path=None
        )

        assert spec.name == "test_tool"
        assert spec.handler == handler
        assert spec.request_schema_path is None
        assert spec.response_schema_path is None
        assert spec.description == ""
        assert spec.display_name is None
        assert spec.category is None

    def test_tool_spec_full(self):
        """Test creating ToolSpec with all fields."""
        handler = MagicMock()
        req_path = Path("/schemas/req.json")
        resp_path = Path("/schemas/resp.json")

        spec = ToolSpec(
            name="full_tool",
            handler=handler,
            request_schema_path=req_path,
            response_schema_path=resp_path,
            description="A full tool",
            display_name="Full Tool",
            category="testing",
        )

        assert spec.name == "full_tool"
        assert spec.description == "A full tool"
        assert spec.display_name == "Full Tool"
        assert spec.category == "testing"


class TestToolRegistry:
    """Test ToolRegistry class."""

    @pytest.fixture
    def registry(self):
        """Create a fresh registry for testing."""
        return ToolRegistry()

    @pytest.fixture
    def mock_handler(self):
        """Create a mock handler function."""

        def handler(payload: Mapping[str, Any], context: ToolContext) -> Mapping[str, Any]:
            return {"output": payload.get("input", "default")}

        return handler

    @pytest.fixture
    def sample_spec(self, mock_handler):
        """Create a sample ToolSpec."""
        return ToolSpec(
            name="sample_tool",
            handler=mock_handler,
            request_schema_path=None,
            response_schema_path=None,
            description="Sample tool",
            display_name="Sample Tool",
            category="test_category",
        )

    def test_init(self, registry):
        """Test registry initialization."""
        assert registry._tools == {}
        assert registry._display_names == {}
        assert registry._categories == {}
        assert registry._category_members == {}

    def test_register_tool(self, registry, sample_spec):
        """Test registering a tool."""
        registry.register(sample_spec)

        assert "sample_tool" in registry._tools
        assert registry._tools["sample_tool"] == sample_spec
        assert registry._display_names["sample_tool"] == "Sample Tool"
        assert registry._categories["sample_tool"] == "test_category"
        assert "sample_tool" in registry._category_members["test_category"]

    def test_register_tool_overwrite(self, registry, sample_spec):
        """Test overwriting an existing tool registration."""
        # First registration
        registry.register(sample_spec)
        assert "sample_tool" in registry._tools

        # Register again - should overwrite without error
        registry.register(sample_spec)
        assert "sample_tool" in registry._tools
        assert registry._tools["sample_tool"] == sample_spec

    def test_get_registered_tool(self, registry, sample_spec):
        """Test getting a registered tool."""
        registry.register(sample_spec)

        retrieved_spec = registry.get("sample_tool")
        assert retrieved_spec == sample_spec

    def test_get_unregistered_tool(self, registry):
        """Test getting an unregistered tool raises error."""
        with pytest.raises(KeyError) as exc_info:
            registry.get("nonexistent")

        assert "Tool 'nonexistent' is not registered" in str(exc_info.value)

    def test_available(self, registry):
        """Test listing available tools."""
        # Register multiple tools
        for i in range(3):
            spec = ToolSpec(
                name=f"tool_{i}",
                handler=lambda p, c: {},
                request_schema_path=None,
                response_schema_path=None,
            )
            registry.register(spec)

        available = list(registry.available())
        assert available == ["tool_0", "tool_1", "tool_2"]  # Should be sorted

    def test_invoke_simple(self, registry, sample_spec):
        """Test invoking a tool without schema validation."""
        registry.register(sample_spec)

        context = ToolContext(repo_root=Path("/test"), settings={}, sandbox=None)

        result = registry.invoke("sample_tool", {"input": "test_value"}, context)

        assert result == {"output": "test_value"}

    def test_invoke_with_request_validation(self, registry, mock_handler):
        """Test invoking tool with request schema validation."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            schema = {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {"required_field": {"type": "string"}},
                "required": ["required_field"],
            }
            json.dump(schema, f)
            schema_path = Path(f.name)

        try:
            spec = ToolSpec(
                name="validated_tool",
                handler=mock_handler,
                request_schema_path=schema_path,
                response_schema_path=None,
            )
            registry.register(spec)

            context = ToolContext(Path("/test"), {}, None)

            # Valid request
            result = registry.invoke("validated_tool", {"required_field": "value"}, context)
            assert result is not None

            # Invalid request - missing required field
            with pytest.raises(ValueError) as exc_info:
                registry.invoke("validated_tool", {"wrong_field": "value"}, context)
            assert "Invalid input for validated_tool" in str(exc_info.value)
        finally:
            schema_path.unlink()

    def test_invoke_with_response_validation(self, registry):
        """Test invoking tool with response schema validation."""

        def bad_handler(payload, context):
            return {"wrong_field": "value"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            schema = {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {"expected_field": {"type": "string"}},
                "required": ["expected_field"],
            }
            json.dump(schema, f)
            schema_path = Path(f.name)

        try:
            spec = ToolSpec(
                name="response_validated",
                handler=bad_handler,
                request_schema_path=None,
                response_schema_path=schema_path,
            )
            registry.register(spec)

            context = ToolContext(Path("/test"), {}, None)

            # Should raise error due to invalid response
            with pytest.raises(ValueError) as exc_info:
                registry.invoke("response_validated", {}, context)

            assert "Tool response_validated returned invalid response" in str(exc_info.value)
        finally:
            schema_path.unlink()

    def test_canonical_name(self, registry, sample_spec):
        """Test canonical_name method."""
        registry.register(sample_spec)

        # Registered tool returns its name
        assert registry.canonical_name("sample_tool") == "sample_tool"

        # Unregistered tool returns the input name
        assert registry.canonical_name("unknown") == "unknown"

        # None/empty returns "generic"
        assert registry.canonical_name(None) == "generic"
        assert registry.canonical_name("") == "generic"

    def test_display_name(self, registry, sample_spec):
        """Test display_name method."""
        registry.register(sample_spec)

        assert registry.display_name("sample_tool") == "Sample Tool"

        # Unknown tool returns canonical name
        assert registry.display_name("unknown") == "unknown"

        # None returns "generic"
        assert registry.display_name(None) == "generic"

    def test_category(self, registry, sample_spec):
        """Test category method."""
        registry.register(sample_spec)

        assert registry.category("sample_tool") == "test_category"

        # Unknown tool returns "generic"
        assert registry.category("unknown") == "generic"
        assert registry.category(None) == "generic"

    def test_aliases(self, registry, sample_spec):
        """Test aliases method."""
        registry.register(sample_spec)

        # Registered tool returns itself
        assert registry.aliases("sample_tool") == ("sample_tool",)
        assert registry.aliases("sample_tool", include_canonical=False) == ()

        # Unregistered tool behavior
        assert registry.aliases("unknown") == ("unknown",)
        assert registry.aliases("unknown", include_canonical=False) == ()

        # Generic case
        assert registry.aliases(None) == ()
        assert registry.aliases("") == ()

    def test_aliases_by_category(self, registry):
        """Test aliases_by_category method."""
        # Register multiple tools in same category
        for i in range(3):
            spec = ToolSpec(
                name=f"cat_tool_{i}",
                handler=lambda p, c: {},
                request_schema_path=None,
                response_schema_path=None,
                category="shared_category",
            )
            registry.register(spec)

        # Register tool in different category
        other_spec = ToolSpec(
            name="other_tool",
            handler=lambda p, c: {},
            request_schema_path=None,
            response_schema_path=None,
            category="other_category",
        )
        registry.register(other_spec)

        # Check category members
        shared_aliases = registry.aliases_by_category("shared_category")
        assert shared_aliases == ("cat_tool_0", "cat_tool_1", "cat_tool_2")

        other_aliases = registry.aliases_by_category("other_category")
        assert other_aliases == ("other_tool",)

        # Unknown category returns empty
        assert registry.aliases_by_category("unknown") == ()

    def test_tool_in_category(self, registry, sample_spec):
        """Test tool_in_category method."""
        registry.register(sample_spec)

        assert registry.tool_in_category("sample_tool", "test_category") is True
        assert registry.tool_in_category("sample_tool", "wrong_category") is False

        # Unknown tool
        assert registry.tool_in_category("unknown", "any_category") is False
        assert registry.tool_in_category(None, "any_category") is False

    def test_rebuild_indices(self, registry):
        """Test _rebuild_indices internal method."""
        # Register tools with different properties
        spec1 = ToolSpec(
            name="tool1",
            handler=lambda p, c: {},
            request_schema_path=None,
            response_schema_path=None,
            display_name="Tool One",
            category="cat1",
        )

        spec2 = ToolSpec(
            name="tool2",
            handler=lambda p, c: {},
            request_schema_path=None,
            response_schema_path=None,
            # No display_name or category - should default
        )

        registry.register(spec1)
        registry.register(spec2)

        # Check indices were built correctly
        assert registry._display_names["tool1"] == "Tool One"
        assert registry._display_names["tool2"] == "tool2"  # Default to name

        assert registry._categories["tool1"] == "cat1"
        assert registry._categories["tool2"] == "generic"  # Default

        assert "tool1" in registry._category_members["cat1"]
        assert "tool2" in registry._category_members["generic"]


class TestLoadValidator:
    """Test _load_validator function."""

    def test_load_validator(self):
        """Test loading a JSON schema validator."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            schema = {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {"name": {"type": "string"}},
            }
            json.dump(schema, f)
            schema_path = Path(f.name)

        try:
            validator = _load_validator(schema_path)

            assert isinstance(validator, Draft7Validator)
            assert validator.schema == schema

            # Test caching - should return same instance
            validator2 = _load_validator(schema_path)
            assert validator is validator2
        finally:
            schema_path.unlink()
            _load_validator.cache_clear()

    def test_load_validator_invalid_json(self):
        """Test loading invalid JSON raises error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json")
            schema_path = Path(f.name)

        try:
            with pytest.raises(json.JSONDecodeError):
                _load_validator(schema_path)
        finally:
            schema_path.unlink()
            _load_validator.cache_clear()


class TestGlobalRegistry:
    """Test the global registry instance."""

    def test_global_registry_exists(self):
        """Test that global registry is available."""
        assert registry is not None
        assert isinstance(registry, ToolRegistry)
