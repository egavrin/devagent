"""Tests for the dynamic context tracking module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ai_dev_agent.cli.dynamic_context import DynamicContextTracker


class TestDynamicContextTracker:
    """Test suite for DynamicContextTracker."""

    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def tracker(self, temp_workspace):
        """Create a DynamicContextTracker instance."""
        return DynamicContextTracker(temp_workspace)

    def test_init(self, temp_workspace):
        """Test tracker initialization."""
        tracker = DynamicContextTracker(temp_workspace)

        assert tracker.workspace == temp_workspace
        assert len(tracker.mentioned_files) == 0
        assert len(tracker.mentioned_symbols) == 0
        assert tracker.step_count == 0
        assert tracker.last_refresh_context_hash is None

    def test_update_from_step_with_read_action(self, tracker):
        """Test updating from a step with read action."""
        step_record = MagicMock()
        action = MagicMock()
        action.tool = "read"
        action.parameters = {"file_path": "src/main.py"}
        step_record.action = action
        step_record.observation = None

        tracker.update_from_step(step_record)

        assert tracker.step_count == 1
        assert "src/main.py" in tracker.mentioned_files

    def test_update_from_step_with_edit_action(self, tracker):
        """Test updating from a step with edit action."""
        step_record = MagicMock()
        action = MagicMock()
        action.tool = "edit"
        action.parameters = {"path": "tests/test.py"}
        step_record.action = action
        step_record.observation = None

        tracker.update_from_step(step_record)

        assert "tests/test.py" in tracker.mentioned_files

    def test_update_from_step_with_write_action(self, tracker):
        """Test updating from a step with write action."""
        step_record = MagicMock()
        action = MagicMock()
        action.tool = "write"
        action.parameters = {"file_path": "output.txt"}
        step_record.action = action
        step_record.observation = None

        tracker.update_from_step(step_record)

        assert "output.txt" in tracker.mentioned_files

    def test_update_from_step_with_grep_action(self, tracker):
        """Test updating from a step with grep action."""
        step_record = MagicMock()
        action = MagicMock()
        action.tool = "grep"
        action.parameters = {"pattern": "UserManager"}
        step_record.action = action
        step_record.observation = None

        tracker.update_from_step(step_record)

        assert "UserManager" in tracker.mentioned_symbols

    def test_update_from_step_with_find_action(self, tracker):
        """Test updating from a step with find action."""
        step_record = MagicMock()
        action = MagicMock()
        action.tool = "find"
        action.parameters = {"query": "DatabaseConnection"}
        step_record.action = action
        step_record.observation = None

        tracker.update_from_step(step_record)

        assert "DatabaseConnection" in tracker.mentioned_symbols

    def test_update_from_step_with_observation(self, tracker):
        """Test updating from step observation."""
        step_record = MagicMock()
        step_record.action = None
        observation = MagicMock()
        observation.outcome = "Found UserAuth class in auth/user.py"
        observation.display_message = None
        observation.artifacts = None
        step_record.observation = observation

        tracker.update_from_step(step_record)

        assert "UserAuth" in tracker.mentioned_symbols
        assert "auth/user.py" in tracker.mentioned_files

    def test_update_from_step_with_display_message(self, tracker):
        """Test updating from observation display message."""
        step_record = MagicMock()
        step_record.action = None
        observation = MagicMock()
        observation.outcome = None
        observation.display_message = "Check config/settings.json for API_KEY"
        observation.artifacts = None
        step_record.observation = observation

        tracker.update_from_step(step_record)

        assert "config/settings.json" in tracker.mentioned_files
        assert "API_KEY" in tracker.mentioned_symbols

    def test_update_from_step_with_artifacts(self, tracker):
        """Test updating from observation artifacts."""
        step_record = MagicMock()
        step_record.action = None
        observation = MagicMock()
        observation.outcome = None
        observation.display_message = None
        observation.artifacts = ["output/report.pdf", "logs/debug.log"]
        step_record.observation = observation

        tracker.update_from_step(step_record)

        assert "output/report.pdf" in tracker.mentioned_files
        assert "logs/debug.log" in tracker.mentioned_files

    def test_update_from_text_basic(self, tracker):
        """Test extracting context from plain text."""
        text = "The UserManager class in src/users.py handles authentication"

        tracker.update_from_text(text)

        assert "UserManager" in tracker.mentioned_symbols
        assert "src/users.py" in tracker.mentioned_files

    def test_update_from_text_quoted_paths(self, tracker):
        """Test extracting quoted file paths."""
        text = "Check \"config/database.yaml\" and 'scripts/setup.sh' for configuration"

        tracker.update_from_text(text)

        assert "config/database.yaml" in tracker.mentioned_files
        assert "scripts/setup.sh" in tracker.mentioned_files

    def test_extract_symbols_camelcase(self, tracker):
        """Test extracting CamelCase symbols."""
        text = "DatabaseManager and UserProfile are important classes"

        tracker._extract_symbols_from_text(text)

        assert "DatabaseManager" in tracker.mentioned_symbols
        assert "UserProfile" in tracker.mentioned_symbols

    def test_extract_symbols_snakecase(self, tracker):
        """Test extracting snake_case symbols."""
        text = "Call get_user_data and validate_input functions"

        tracker._extract_symbols_from_text(text)

        assert "get_user_data" in tracker.mentioned_symbols
        assert "validate_input" in tracker.mentioned_symbols

    def test_extract_symbols_constant_case(self, tracker):
        """Test extracting CONSTANT_CASE symbols."""
        text = "Set MAX_CONNECTIONS and DEFAULT_TIMEOUT values"

        tracker._extract_symbols_from_text(text)

        assert "MAX_CONNECTIONS" in tracker.mentioned_symbols
        assert "DEFAULT_TIMEOUT" in tracker.mentioned_symbols

    def test_skip_http_urls(self, tracker):
        """Test that HTTP URLs are not treated as files."""
        text = "Download from https://example.com/file.zip"

        tracker.update_from_text(text)

        assert "https://example.com/file.zip" not in tracker.mentioned_files

    def test_add_file_normalization(self, tracker):
        """Test file path adding (normalization only occurs for existing files)."""
        tracker._add_file("./src/main.py")
        tracker._add_file("../lib/utils.js")
        tracker._add_file("~/documents/report.pdf")

        # Files are stored as-is when they don't exist
        assert "./src/main.py" in tracker.mentioned_files
        assert "../lib/utils.js" in tracker.mentioned_files
        assert "~/documents/report.pdf" in tracker.mentioned_files

    def test_get_context_summary(self, tracker):
        """Test getting context summary."""
        tracker.mentioned_files.add("src/main.py")
        tracker.mentioned_files.add("tests/test.py")
        tracker.mentioned_symbols.add("UserClass")
        tracker.mentioned_symbols.add("get_data")
        tracker.step_count = 5

        summary = tracker.get_context_summary()

        assert len(summary["files"]) == 2
        assert len(summary["symbols"]) == 2
        assert summary["step_count"] == 5
        assert summary["total_mentions"] == 4
        assert "src/main.py" in summary["files"]
        assert "UserClass" in summary["symbols"]

    def test_clear_context(self, tracker):
        """Test clearing accumulated context."""
        tracker.mentioned_files.add("test.py")
        tracker.mentioned_symbols.add("TestClass")
        tracker.step_count = 10

        tracker.clear()

        assert len(tracker.mentioned_files) == 0
        assert len(tracker.mentioned_symbols) == 0
        assert tracker.step_count == 0
        assert tracker.last_refresh_context_hash is None

    def test_multiple_updates(self, tracker):
        """Test multiple updates accumulate context."""
        step1 = MagicMock()
        action1 = MagicMock()
        action1.tool = "read"
        action1.parameters = {"file_path": "file1.py"}
        step1.action = action1
        step1.observation = None

        step2 = MagicMock()
        action2 = MagicMock()
        action2.tool = "grep"
        action2.parameters = {
            "pattern": "ClassName"
        }  # Use CamelCase pattern that will be recognized
        step2.action = action2
        step2.observation = None

        step3 = MagicMock()
        action3 = MagicMock()
        action3.tool = "edit"
        action3.parameters = {"path": "file2.py"}
        step3.action = action3
        step3.observation = None

        tracker.update_from_step(step1)
        tracker.update_from_step(step2)
        tracker.update_from_step(step3)

        assert tracker.step_count == 3
        assert "file1.py" in tracker.mentioned_files
        assert "file2.py" in tracker.mentioned_files
        assert "ClassName" in tracker.mentioned_symbols

    def test_empty_step_handling(self, tracker):
        """Test handling steps with missing data."""
        # Step with no action
        step1 = MagicMock()
        step1.action = None
        step1.observation = None

        tracker.update_from_step(step1)
        assert tracker.step_count == 1

        # Step with empty parameters
        step2 = MagicMock()
        action = MagicMock()
        action.tool = "read"
        action.parameters = {}
        step2.action = action
        step2.observation = None

        tracker.update_from_step(step2)
        assert tracker.step_count == 2

    def test_should_refresh_repomap(self, tracker):
        """Test checking if repomap refresh is needed."""
        # Initially should not need refresh (no steps)
        assert not tracker.should_refresh_repomap()

        # After 2 steps (even number), should refresh
        tracker.step_count = 2
        assert tracker.should_refresh_repomap()

        # Check that hash is updated - calling again without changes returns False
        assert not tracker.should_refresh_repomap()

        # Add new context and check again
        tracker.mentioned_files.add("new_file.py")
        tracker.step_count = 4  # Even step count
        assert tracker.should_refresh_repomap()
