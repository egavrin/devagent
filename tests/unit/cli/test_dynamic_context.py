"""Unit tests for dynamic context tracking."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ai_dev_agent.cli.dynamic_context import DynamicContextTracker


@pytest.fixture
def tmp_workspace(tmp_path):
    """Create temporary workspace directory."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace


@pytest.fixture
def tracker(tmp_workspace):
    """Create a DynamicContextTracker instance."""
    return DynamicContextTracker(tmp_workspace)


class TestInitialization:
    """Test tracker initialization."""

    def test_initializes_with_empty_state(self, tmp_workspace):
        """Test that tracker initializes with empty state."""
        tracker = DynamicContextTracker(tmp_workspace)

        assert tracker.workspace == tmp_workspace
        assert tracker.mentioned_files == set()
        assert tracker.mentioned_symbols == set()
        assert tracker.step_count == 0
        assert tracker.last_refresh_context_hash is None


class TestUpdateFromStep:
    """Test update_from_step method."""

    def test_increments_step_count(self, tracker):
        """Test that step count is incremented."""
        step_record = MagicMock()
        step_record.action = None
        step_record.observation = None

        tracker.update_from_step(step_record)

        assert tracker.step_count == 1

    def test_extracts_file_from_read_action(self, tracker):
        """Test extracting file from read tool action."""
        step_record = MagicMock()
        step_record.action = MagicMock()
        step_record.action.tool = "read"
        step_record.action.parameters = {"file_path": "src/main.py"}
        step_record.observation = None

        tracker.update_from_step(step_record)

        assert "src/main.py" in tracker.mentioned_files

    def test_extracts_file_from_edit_action(self, tracker):
        """Test extracting file from edit tool action."""
        step_record = MagicMock()
        step_record.action = MagicMock()
        step_record.action.tool = "edit"
        step_record.action.parameters = {"file_path": "src/utils.py"}
        step_record.observation = None

        tracker.update_from_step(step_record)

        assert "src/utils.py" in tracker.mentioned_files

    def test_extracts_file_from_write_action(self, tracker):
        """Test extracting file from write tool action."""
        step_record = MagicMock()
        step_record.action = MagicMock()
        step_record.action.tool = "write"
        step_record.action.parameters = {"path": "docs/README.md"}
        step_record.observation = None

        tracker.update_from_step(step_record)

        assert "docs/README.md" in tracker.mentioned_files

    def test_extracts_pattern_from_grep_action(self, tracker):
        """Test extracting search pattern from grep action."""
        step_record = MagicMock()
        step_record.action = MagicMock()
        step_record.action.tool = "grep"
        step_record.action.parameters = {"pattern": "ContextManager"}
        step_record.observation = None

        tracker.update_from_step(step_record)

        assert "ContextManager" in tracker.mentioned_symbols

    def test_extracts_query_from_find_action(self, tracker):
        """Test extracting query from find action."""
        step_record = MagicMock()
        step_record.action = MagicMock()
        step_record.action.tool = "find"
        step_record.action.parameters = {"query": "UserSettings"}
        step_record.observation = None

        tracker.update_from_step(step_record)

        assert "UserSettings" in tracker.mentioned_symbols

    def test_extracts_from_observation_outcome(self, tracker):
        """Test extracting from observation outcome."""
        step_record = MagicMock()
        step_record.action = None
        step_record.observation = MagicMock()
        step_record.observation.outcome = "Found in src/config.py: ConfigLoader class"
        step_record.observation.display_message = None
        step_record.observation.artifacts = None

        tracker.update_from_step(step_record)

        assert "src/config.py" in tracker.mentioned_files
        assert "ConfigLoader" in tracker.mentioned_symbols

    def test_extracts_from_observation_display_message(self, tracker):
        """Test extracting from observation display message."""
        step_record = MagicMock()
        step_record.action = None
        step_record.observation = MagicMock()
        step_record.observation.outcome = None
        step_record.observation.display_message = "Modified tests/test_utils.py"
        step_record.observation.artifacts = None

        tracker.update_from_step(step_record)

        assert "tests/test_utils.py" in tracker.mentioned_files

    def test_extracts_from_observation_artifacts(self, tracker):
        """Test extracting from observation artifacts."""
        step_record = MagicMock()
        step_record.action = None
        step_record.observation = MagicMock()
        step_record.observation.outcome = None
        step_record.observation.display_message = None
        step_record.observation.artifacts = ["output/report.html", "output/data.json"]

        tracker.update_from_step(step_record)

        assert "output/report.html" in tracker.mentioned_files
        assert "output/data.json" in tracker.mentioned_files

    def test_handles_missing_action_attributes(self, tracker):
        """Test handling step with missing action attributes."""
        step_record = MagicMock()
        step_record.action = MagicMock(spec=[])  # No attributes
        step_record.observation = None

        # Should not raise
        tracker.update_from_step(step_record)
        assert tracker.step_count == 1

    def test_handles_non_dict_parameters(self, tracker):
        """Test handling action with non-dict parameters."""
        step_record = MagicMock()
        step_record.action = MagicMock()
        step_record.action.tool = "read"
        step_record.action.parameters = "not-a-dict"
        step_record.observation = None

        # Should not raise
        tracker.update_from_step(step_record)


class TestUpdateFromText:
    """Test update_from_text method."""

    def test_extracts_files_from_text(self, tracker):
        """Test extracting file mentions from arbitrary text."""
        text = "I modified src/main.py and tests/test_main.py"

        tracker.update_from_text(text)

        assert "src/main.py" in tracker.mentioned_files
        assert "tests/test_main.py" in tracker.mentioned_files

    def test_extracts_symbols_from_text(self, tracker):
        """Test extracting symbols from text."""
        text = "The ConfigLoader and UserSettings classes need updating"

        tracker.update_from_text(text)

        assert "ConfigLoader" in tracker.mentioned_symbols
        assert "UserSettings" in tracker.mentioned_symbols

    def test_ignores_urls(self, tracker):
        """Test that URLs are not treated as files."""
        text = "See https://example.com/docs.html for details"

        tracker.update_from_text(text)

        # URL should not be in mentioned files
        assert not any("https://" in f for f in tracker.mentioned_files)

    def test_handles_empty_text(self, tracker):
        """Test handling empty text."""
        tracker.update_from_text("")
        tracker.update_from_text(None)

        assert len(tracker.mentioned_files) == 0


class TestAddFile:
    """Test _add_file method."""

    def test_normalizes_relative_path(self, tracker, tmp_workspace):
        """Test normalizing relative paths to workspace."""
        # Create actual file
        test_file = tmp_workspace / "src" / "test.py"
        test_file.parent.mkdir(parents=True)
        test_file.touch()

        tracker._add_file("src/test.py")

        assert "src/test.py" in tracker.mentioned_files

    def test_handles_absolute_paths(self, tracker):
        """Test handling absolute paths."""
        abs_path = "/absolute/path/to/file.py"

        tracker._add_file(abs_path)

        assert abs_path in tracker.mentioned_files

    def test_handles_nonexistent_files(self, tracker):
        """Test handling paths to nonexistent files."""
        tracker._add_file("nonexistent/file.py")

        # Should still add it
        assert "nonexistent/file.py" in tracker.mentioned_files


class TestGetContextSummary:
    """Test get_context_summary method."""

    def test_returns_summary_dict(self, tracker):
        """Test that summary returns proper dict structure."""
        tracker.mentioned_files.add("src/main.py")
        tracker.mentioned_symbols.add("ConfigLoader")
        tracker.step_count = 3

        summary = tracker.get_context_summary()

        assert "files" in summary
        assert "symbols" in summary
        assert "step_count" in summary
        assert "total_mentions" in summary
        assert summary["step_count"] == 3
        assert summary["total_mentions"] == 2

    def test_summary_with_empty_state(self, tracker):
        """Test summary with no tracked context."""
        summary = tracker.get_context_summary()

        assert summary["files"] == []
        assert summary["symbols"] == []
        assert summary["step_count"] == 0
        assert summary["total_mentions"] == 0


class TestShouldRefreshRepomap:
    """Test should_refresh_repomap method."""

    def test_no_refresh_on_first_step(self, tracker):
        """Test that no refresh happens before any steps."""
        assert not tracker.should_refresh_repomap()

    def test_refresh_on_even_steps(self, tracker):
        """Test that refresh happens on even step numbers."""
        tracker.step_count = 1
        tracker.mentioned_files.add("test.py")
        assert not tracker.should_refresh_repomap()

        tracker.step_count = 2
        assert tracker.should_refresh_repomap()

    def test_refresh_with_many_files(self, tracker):
        """Test that refresh happens when many files mentioned."""
        tracker.step_count = 1
        tracker.mentioned_files = {"file1.py", "file2.py", "file3.py", "file4.py"}

        assert tracker.should_refresh_repomap()

    def test_refresh_with_many_symbols(self, tracker):
        """Test that refresh happens when many symbols mentioned."""
        tracker.step_count = 1
        tracker.mentioned_symbols = {"Sym1", "Sym2", "Sym3", "Sym4", "Sym5", "Sym6"}

        assert tracker.should_refresh_repomap()

    def test_no_refresh_when_context_unchanged(self, tracker):
        """Test that no refresh happens when context hasn't changed."""
        tracker.step_count = 2
        tracker.mentioned_files.add("test.py")

        # First call should refresh
        assert tracker.should_refresh_repomap()

        # Second call with same context should not
        assert not tracker.should_refresh_repomap()

    def test_refresh_when_context_changes(self, tracker):
        """Test that refresh happens when context changes."""
        tracker.step_count = 2
        tracker.mentioned_files.add("test1.py")

        # First refresh
        assert tracker.should_refresh_repomap()

        # Add new file
        tracker.step_count = 4
        tracker.mentioned_files.add("test2.py")

        # Should refresh again due to new context
        assert tracker.should_refresh_repomap()


class TestClear:
    """Test clear method."""

    def test_clears_all_state(self, tracker):
        """Test that clear resets tracked context but keeps refresh hash."""
        tracker.mentioned_files.add("test.py")
        tracker.mentioned_symbols.add("TestClass")
        tracker.step_count = 5
        tracker.last_refresh_context_hash = 12345

        tracker.clear()

        assert len(tracker.mentioned_files) == 0
        assert len(tracker.mentioned_symbols) == 0
        assert tracker.step_count == 0
        # Note: last_refresh_context_hash is intentionally NOT reset
        # This preserves the refresh state across clear operations


class TestSymbolExtraction:
    """Test symbol extraction patterns."""

    def test_extracts_camel_case(self, tracker):
        """Test extraction of CamelCase identifiers."""
        text = "UserManager and DataLoader are classes"

        tracker.update_from_text(text)

        assert "UserManager" in tracker.mentioned_symbols
        assert "DataLoader" in tracker.mentioned_symbols

    def test_extracts_snake_case(self, tracker):
        """Test extraction of snake_case identifiers."""
        text = "The parse_config_file and load_user_data functions"

        tracker.update_from_text(text)

        assert "parse_config_file" in tracker.mentioned_symbols
        assert "load_user_data" in tracker.mentioned_symbols

    def test_filters_short_snake_case(self, tracker):
        """Test that short snake_case words are filtered."""
        text = "The is_ok and has_it flags"

        tracker.update_from_text(text)

        # Short words should be filtered
        assert "is_ok" not in tracker.mentioned_symbols

    def test_extracts_constant_case(self, tracker):
        """Test extraction of CONSTANT_CASE identifiers."""
        text = "Use MAX_RETRY_COUNT and DEFAULT_TIMEOUT constants"

        tracker.update_from_text(text)

        assert "MAX_RETRY_COUNT" in tracker.mentioned_symbols
        assert "DEFAULT_TIMEOUT" in tracker.mentioned_symbols
