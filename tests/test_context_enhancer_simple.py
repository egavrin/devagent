"""Simplified tests for the context enhancer module - focuses on core functionality."""
import json
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch
import tempfile
import pytest

from ai_dev_agent.cli.context_enhancer import ContextEnhancer
from ai_dev_agent.core.utils.config import Settings


class TestContextEnhancerSimple:
    """Simplified test suite for ContextEnhancer focusing on working tests."""

    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            (workspace / "test.py").write_text("def test_function():\n    pass")
            (workspace / "src").mkdir(exist_ok=True)
            (workspace / "src" / "main.py").write_text("class MainClass:\n    pass")
            yield workspace

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings with all features disabled."""
        settings = MagicMock(spec=Settings)
        settings.enable_memory_bank = False
        settings.enable_playbook = False
        settings.enable_dynamic_instructions = False
        settings.enable_repo_map = True
        settings.repo_map_style = "aider"
        return settings

    @pytest.fixture
    def enhancer(self, temp_workspace, mock_settings):
        """Create a ContextEnhancer instance."""
        return ContextEnhancer(workspace=temp_workspace, settings=mock_settings)

    def test_init_basic(self, temp_workspace, mock_settings):
        """Test basic initialization."""
        enhancer = ContextEnhancer(workspace=temp_workspace, settings=mock_settings)

        assert enhancer.workspace == temp_workspace
        assert enhancer.settings == mock_settings
        assert enhancer._repo_map is None
        assert enhancer._initialized is False

    def test_extract_symbols_basic(self, enhancer):
        """Test basic symbol extraction."""
        text = "The UserAuth class and validate_token function need work"

        symbols, files = enhancer.extract_symbols_and_files(text)

        assert "UserAuth" in symbols  # CamelCase
        assert "validate_token" in symbols  # snake_case

    def test_extract_files_basic(self, enhancer):
        """Test basic file extraction."""
        text = "Check src/auth.py and tests/test_auth.py for issues"

        symbols, files = enhancer.extract_symbols_and_files(text)

        assert "src/auth.py" in files
        assert "tests/test_auth.py" in files

    def test_extract_quoted_files(self, enhancer):
        """Test extraction of quoted filenames."""
        text = 'The file "config/settings.json" has the configuration'

        symbols, files = enhancer.extract_symbols_and_files(text)

        assert "config/settings.json" in files

    def test_get_important_files_scoring(self, enhancer):
        """Test file importance scoring."""
        all_files = ["main.py", "test.py", "README.md", ".gitignore"]

        important = enhancer._get_important_files(all_files, max_files=2)

        assert len(important) == 2
        # Check structure
        for item in important:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], str)
            assert isinstance(item[1], float)

    def test_extract_camelcase_symbols(self, enhancer):
        """Test CamelCase symbol extraction."""
        text = "DatabaseManager handles UserProfile objects"

        symbols, _ = enhancer.extract_symbols_and_files(text)

        assert "DatabaseManager" in symbols
        assert "UserProfile" in symbols

    def test_extract_snakecase_symbols(self, enhancer):
        """Test snake_case symbol extraction."""
        text = "The get_user_data and process_request functions"

        symbols, _ = enhancer.extract_symbols_and_files(text)

        assert "get_user_data" in symbols
        assert "process_request" in symbols

    def test_extract_constant_case_symbols(self, enhancer):
        """Test CONSTANT_CASE symbol extraction."""
        text = "Set MAX_RETRY_COUNT and DEFAULT_TIMEOUT values"

        symbols, _ = enhancer.extract_symbols_and_files(text)

        assert "MAX_RETRY_COUNT" in symbols
        assert "DEFAULT_TIMEOUT" in symbols

    def test_file_mention_limit(self, enhancer):
        """Test that file extraction respects the limit."""
        # Create text with many file mentions
        files_list = [f"file{i}.py" for i in range(50)]
        text = " ".join(files_list)

        _, files = enhancer.extract_symbols_and_files(text)

        # Should not exceed FILE_MENTION_LIMIT
        assert len(files) <= enhancer.FILE_MENTION_LIMIT

    def test_stop_words_filtering(self, enhancer):
        """Test that common stop words are filtered from symbols."""
        text = "Find all files where the UserManager class is used"

        symbols, _ = enhancer.extract_symbols_and_files(text)

        # Stop words should not be in symbols
        assert "find" not in symbols
        assert "all" not in symbols
        assert "files" not in symbols
        assert "where" not in symbols
        assert "the" not in symbols
        # But actual symbol should be
        assert "UserManager" in symbols

    @patch('ai_dev_agent.cli.context_enhancer.MEMORY_SYSTEM_AVAILABLE', False)
    @patch('ai_dev_agent.cli.context_enhancer.PLAYBOOK_SYSTEM_AVAILABLE', False)
    @patch('ai_dev_agent.cli.context_enhancer.DYNAMIC_INSTRUCTIONS_AVAILABLE', False)
    def test_init_with_all_disabled(self, temp_workspace):
        """Test initialization with all optional features disabled."""
        settings = MagicMock(spec=Settings)
        settings.enable_memory_bank = False
        settings.enable_playbook = False
        settings.enable_dynamic_instructions = False

        enhancer = ContextEnhancer(workspace=temp_workspace, settings=settings)

        assert enhancer._memory_store is None
        assert enhancer._playbook_manager is None
        assert enhancer._dynamic_instruction_manager is None

    def test_extract_mixed_symbols(self, enhancer):
        """Test extraction of mixed symbol types."""
        text = """
        The UserAuth class uses get_token() method with MAX_RETRIES.
        Check DatabaseManager and user_profile table.
        """

        symbols, _ = enhancer.extract_symbols_and_files(text)

        assert "UserAuth" in symbols  # PascalCase
        assert "get_token" in symbols  # snake_case (without parentheses)
        assert "MAX_RETRIES" in symbols  # CONSTANT_CASE
        assert "DatabaseManager" in symbols  # PascalCase
        assert "user_profile" in symbols  # snake_case

    def test_extract_files_with_paths(self, enhancer):
        """Test extraction of files with various path formats."""
        text = """
        Check these files:
        - src/main.py
        - ./utils/helper.js
        - ../lib/common.go
        - test/unit/test_auth.py
        """

        _, files = enhancer.extract_symbols_and_files(text)

        assert "src/main.py" in files
        assert "./utils/helper.js" in files
        assert "../lib/common.go" in files
        assert "test/unit/test_auth.py" in files