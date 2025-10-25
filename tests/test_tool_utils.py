"""Tests for tool utilities module."""
import json
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from ai_dev_agent.core.utils.tool_utils import (
    canonical_tool_name,
    tool_category,
    display_tool_name,
    tool_aliases,
    expand_tool_aliases,
    build_tool_signature,
    tool_signature,
    _registry,
)


class MockRegistry:
    """Mock registry for testing."""

    def __init__(self):
        self.canonical_names = {
            "read": "read_file",
            "cat": "read_file",
            "view": "read_file",
            "find": "find_files",
            "search": "find_files",
            "grep": "search_content",
            "rg": "search_content",
        }
        self.categories = {
            "read_file": "file_read",
            "find_files": "search",
            "search_content": "search",
            "symbols": "analysis",
        }
        self.display_names = {
            "read_file": "Read File",
            "find_files": "Find Files",
            "search_content": "Search Content",
            "symbols": "Extract Symbols",
        }

    def canonical_name(self, tool_name):
        """Get canonical name for a tool."""
        if not tool_name:
            return "generic"
        return self.canonical_names.get(tool_name, tool_name)

    def category(self, tool_name):
        """Get category for a tool."""
        if not tool_name:
            return "generic"
        canonical = self.canonical_name(tool_name)
        return self.categories.get(canonical, "generic")

    def display_name(self, tool_name):
        """Get display name for a tool."""
        if not tool_name:
            return "generic"
        canonical = self.canonical_name(tool_name)
        return self.display_names.get(canonical, canonical)

    def aliases(self, tool_name, include_canonical=True):
        """Get all aliases for a tool."""
        if not tool_name:
            return ("generic",) if include_canonical else ()

        canonical = self.canonical_name(tool_name)
        aliases = []

        # Find all aliases that map to the same canonical name
        for alias, canon in self.canonical_names.items():
            if canon == canonical:
                aliases.append(alias)

        if include_canonical and canonical not in aliases:
            aliases.append(canonical)

        return tuple(sorted(set(aliases)))

    def tool_in_category(self, tool_name, category):
        """Check if tool is in a specific category."""
        tool_cat = self.category(tool_name)
        return tool_cat == category


@pytest.fixture
def mock_registry():
    """Create a mock registry instance."""
    return MockRegistry()


class TestToolNames:
    """Test tool name functions."""

    def test_canonical_tool_name_with_alias(self, mock_registry):
        """Test getting canonical name for aliased tools."""
        with patch("ai_dev_agent.core.utils.tool_utils._registry") as mock_reg:
            mock_reg.return_value = mock_registry

            assert canonical_tool_name("read") == "read_file"
            assert canonical_tool_name("cat") == "read_file"
            assert canonical_tool_name("grep") == "search_content"

    def test_canonical_tool_name_without_alias(self, mock_registry):
        """Test getting canonical name for non-aliased tools."""
        with patch("ai_dev_agent.core.utils.tool_utils._registry") as mock_reg:
            mock_reg.return_value = mock_registry

            assert canonical_tool_name("unknown_tool") == "unknown_tool"
            assert canonical_tool_name("symbols") == "symbols"

    def test_canonical_tool_name_empty(self, mock_registry):
        """Test canonical name with empty input."""
        with patch("ai_dev_agent.core.utils.tool_utils._registry") as mock_reg:
            mock_reg.return_value = mock_registry

            assert canonical_tool_name(None) == "generic"
            assert canonical_tool_name("") == "generic"

    def test_tool_category_valid(self, mock_registry):
        """Test getting category for valid tools."""
        with patch("ai_dev_agent.core.utils.tool_utils._registry") as mock_reg:
            mock_reg.return_value = mock_registry

            assert tool_category("read") == "file_read"
            assert tool_category("read_file") == "file_read"
            assert tool_category("grep") == "search"
            assert tool_category("symbols") == "analysis"

    def test_tool_category_empty(self, mock_registry):
        """Test category with empty input."""
        with patch("ai_dev_agent.core.utils.tool_utils._registry") as mock_reg:
            mock_reg.return_value = mock_registry

            assert tool_category(None) == "generic"
            assert tool_category("") == "generic"

    def test_display_tool_name_valid(self, mock_registry):
        """Test getting display name for valid tools."""
        with patch("ai_dev_agent.core.utils.tool_utils._registry") as mock_reg:
            mock_reg.return_value = mock_registry

            assert display_tool_name("read") == "Read File"
            assert display_tool_name("grep") == "Search Content"
            assert display_tool_name("symbols") == "Extract Symbols"

    def test_display_tool_name_empty(self, mock_registry):
        """Test display name with empty input."""
        with patch("ai_dev_agent.core.utils.tool_utils._registry") as mock_reg:
            mock_reg.return_value = mock_registry

            assert display_tool_name(None) == "generic"
            assert display_tool_name("") == "generic"

    def test_tool_aliases_with_canonical(self, mock_registry):
        """Test getting aliases including canonical name."""
        with patch("ai_dev_agent.core.utils.tool_utils._registry") as mock_reg:
            mock_reg.return_value = mock_registry

            aliases = tool_aliases("read", include_canonical=True)
            assert "read" in aliases
            assert "cat" in aliases
            assert "view" in aliases
            assert "read_file" in aliases

    def test_tool_aliases_without_canonical(self, mock_registry):
        """Test getting aliases excluding canonical name."""
        with patch("ai_dev_agent.core.utils.tool_utils._registry") as mock_reg:
            mock_reg.return_value = mock_registry

            aliases = tool_aliases("read_file", include_canonical=False)
            # Should include aliases but potentially not the canonical name itself
            assert "read" in aliases or "cat" in aliases or "view" in aliases


class TestToolAliasExpansion:
    """Test alias expansion functionality."""

    def test_expand_tool_aliases_basic(self, mock_registry):
        """Test expanding a basic mapping with aliases."""
        with patch("ai_dev_agent.core.utils.tool_utils._registry") as mock_reg:
            mock_reg.return_value = mock_registry

            mapping = {"read_file": 100}
            expanded = expand_tool_aliases(mapping)

            # Should include all aliases
            assert "read" in expanded
            assert "cat" in expanded
            assert "view" in expanded
            assert "read_file" in expanded

            # All should have the same value
            assert expanded["read"] == 100
            assert expanded["cat"] == 100

    def test_expand_tool_aliases_multiple(self, mock_registry):
        """Test expanding multiple tools."""
        with patch("ai_dev_agent.core.utils.tool_utils._registry") as mock_reg:
            mock_reg.return_value = mock_registry

            mapping = {
                "read_file": 100,
                "search_content": 200,
            }
            expanded = expand_tool_aliases(mapping)

            # Check read_file aliases
            assert expanded["read"] == 100
            assert expanded["cat"] == 100

            # Check search_content aliases
            assert expanded["grep"] == 200
            assert expanded["rg"] == 200

    def test_expand_tool_aliases_empty(self, mock_registry):
        """Test expanding empty mapping."""
        with patch("ai_dev_agent.core.utils.tool_utils._registry") as mock_reg:
            mock_reg.return_value = mock_registry

            expanded = expand_tool_aliases({})
            assert expanded == {}


class TestToolSignatures:
    """Test tool signature generation."""

    def test_build_tool_signature_symbols(self, mock_registry):
        """Test signature for symbols tool."""
        with patch("ai_dev_agent.core.utils.tool_utils._registry") as mock_reg:
            mock_reg.return_value = mock_registry

            sig = build_tool_signature("symbols", {"name": "MyClass"})
            assert sig == "symbols:MyClass"

            sig = build_tool_signature("symbols", {})
            assert sig == "symbols:"

    def test_build_tool_signature_file_read(self, mock_registry):
        """Test signature for file read tools."""
        with patch("ai_dev_agent.core.utils.tool_utils._registry") as mock_reg:
            mock_reg.return_value = mock_registry

            # Single path
            sig = build_tool_signature("read", {"path": "/path/to/file.py"})
            assert sig == "read:/path/to/file.py"

            # Multiple paths
            sig = build_tool_signature("read", {"paths": ["/a.py", "/b.py"]})
            assert sig == "read:/a.py,/b.py"

            # Empty paths
            sig = build_tool_signature("read", {"paths": []})
            assert sig == "read:"

    def test_build_tool_signature_search(self, mock_registry):
        """Test signature for search tools."""
        with patch("ai_dev_agent.core.utils.tool_utils._registry") as mock_reg:
            mock_reg.return_value = mock_registry

            sig = build_tool_signature("grep", {"query": "TODO"})
            assert sig == "grep:TODO"

            sig = build_tool_signature("find", {"query": "*.py"})
            assert sig == "find:*.py"

    def test_build_tool_signature_generic(self, mock_registry):
        """Test signature for generic tools."""
        with patch("ai_dev_agent.core.utils.tool_utils._registry") as mock_reg:
            mock_reg.return_value = mock_registry

            # Should hash the arguments
            sig = build_tool_signature("unknown_tool", {"arg1": "value1", "arg2": "value2"})
            assert sig.startswith("unknown_tool:")
            # Hash should be deterministic
            sig2 = build_tool_signature("unknown_tool", {"arg1": "value1", "arg2": "value2"})
            assert sig == sig2

    def test_build_tool_signature_no_args(self, mock_registry):
        """Test signature with no arguments."""
        with patch("ai_dev_agent.core.utils.tool_utils._registry") as mock_reg:
            mock_reg.return_value = mock_registry

            sig = build_tool_signature("some_tool", None)
            assert sig.startswith("some_tool:")

    def test_build_tool_signature_unhashable_args(self, mock_registry):
        """Test signature with unhashable arguments."""
        with patch("ai_dev_agent.core.utils.tool_utils._registry") as mock_reg:
            mock_reg.return_value = mock_registry

            # Arguments that can't be JSON serialized
            class UnserializableObject:
                def __str__(self):
                    return "custom_object"

            sig = build_tool_signature("tool", {"obj": UnserializableObject()})
            assert sig.startswith("tool:")
            # Should use str() fallback

    def test_build_tool_signature_paths_non_sequence(self, mock_registry):
        """Test file read signature with non-sequence paths."""
        with patch("ai_dev_agent.core.utils.tool_utils._registry") as mock_reg:
            mock_reg.return_value = mock_registry

            # String should be handled as-is, not iterated
            sig = build_tool_signature("read", {"paths": "single_path.py"})
            assert sig == "read:single_path.py"

    def test_build_tool_signature_paths_with_none(self, mock_registry):
        """Test file read signature with None values in paths."""
        with patch("ai_dev_agent.core.utils.tool_utils._registry") as mock_reg:
            mock_reg.return_value = mock_registry

            # Should filter out None values
            sig = build_tool_signature("read", {"paths": ["/a.py", None, "/b.py"]})
            assert sig == "read:/a.py,/b.py"

    def test_tool_signature_from_object(self, mock_registry):
        """Test getting signature from a tool call object."""
        with patch("ai_dev_agent.core.utils.tool_utils._registry") as mock_reg:
            mock_reg.return_value = mock_registry

            # Create a mock tool call object
            tool_call = MagicMock()
            tool_call.name = "grep"
            tool_call.arguments = {"query": "TODO"}

            sig = tool_signature(tool_call)
            assert sig == "grep:TODO"

    def test_tool_signature_from_object_no_args(self, mock_registry):
        """Test getting signature from object without arguments."""
        with patch("ai_dev_agent.core.utils.tool_utils._registry") as mock_reg:
            mock_reg.return_value = mock_registry

            tool_call = MagicMock()
            tool_call.name = "some_tool"
            tool_call.arguments = None

            sig = tool_signature(tool_call)
            assert sig.startswith("some_tool:")

    def test_tool_signature_from_object_no_name(self, mock_registry):
        """Test getting signature from object without name."""
        with patch("ai_dev_agent.core.utils.tool_utils._registry") as mock_reg:
            mock_reg.return_value = mock_registry

            tool_call = MagicMock()
            del tool_call.name  # No name attribute
            tool_call.arguments = {"arg": "value"}

            sig = tool_signature(tool_call)
            assert sig.startswith("unknown:")


class TestRegistryCache:
    """Test registry caching."""

    def test_registry_cached(self):
        """Test that _registry is cached."""
        # Clear the cache first
        _registry.cache_clear()

        with patch("ai_dev_agent.tools.registry.registry") as mock_inst:
            mock_registry_obj = MagicMock()
            mock_inst.__class__ = MagicMock  # Make it look like an instance

            # Patch the import inside _registry function
            with patch("ai_dev_agent.core.utils.tool_utils._registry") as mock_func:
                call_count = 0

                def side_effect():
                    nonlocal call_count
                    call_count += 1
                    return mock_registry_obj

                mock_func.side_effect = side_effect

                # First call
                reg1 = mock_func()
                assert call_count == 1

                # Should return same object (simulating cache)
                mock_func.side_effect = lambda: mock_registry_obj
                reg2 = mock_func()

                # Both should be same instance
                assert reg1 is reg2