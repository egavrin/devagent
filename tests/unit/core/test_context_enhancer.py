"""Tests for the simplified ContextEnhancer module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ai_dev_agent.cli.context_enhancer import ContextEnhancer, get_context_enhancer
from ai_dev_agent.core.utils.config import Settings


class TestContextEnhancer:
    """Test suite for ContextEnhancer."""

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
        """Create mock settings."""
        settings = MagicMock(spec=Settings)
        settings.enable_memory_bank = False
        settings.enable_repo_map = True
        settings.repo_map_style = "aider"
        settings.repomap_debug_stdout = False
        settings.memory_retrieval_limit = 5
        settings.memory_similarity_threshold = 0.3
        settings.memory_prune_threshold = 0.2
        return settings

    @pytest.fixture
    def enhancer(self, temp_workspace, mock_settings):
        """Create a ContextEnhancer instance."""
        return ContextEnhancer(workspace=temp_workspace, settings=mock_settings)

    def test_init(self, temp_workspace, mock_settings):
        """Test ContextEnhancer initialization."""
        enhancer = ContextEnhancer(workspace=temp_workspace, settings=mock_settings)

        assert enhancer.workspace == temp_workspace
        assert enhancer.settings == mock_settings
        assert enhancer._repo_map is None
        assert enhancer._initialized is False
        assert enhancer._memory_provider is not None  # Should have memory provider

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        with patch("ai_dev_agent.cli.context_enhancer.Path.cwd") as mock_cwd:
            mock_cwd.return_value = Path("/test/workspace")
            enhancer = ContextEnhancer()

            assert enhancer.workspace == Path("/test/workspace")
            assert isinstance(enhancer.settings, Settings)

    @patch("ai_dev_agent.cli.memory_provider.MEMORY_SYSTEM_AVAILABLE", True)
    @patch("ai_dev_agent.cli.memory_provider.MemoryStore")
    def test_init_with_memory_system(self, mock_memory_store, temp_workspace):
        """Test initialization with memory system enabled."""
        settings = MagicMock(spec=Settings)
        settings.enable_memory_bank = True

        mock_store = MagicMock()
        mock_store._memories = [1, 2, 3]
        mock_memory_store.return_value = mock_store

        enhancer = ContextEnhancer(workspace=temp_workspace, settings=settings)

        # Memory provider should be initialized
        assert enhancer._memory_provider is not None
        # Memory store initialization happens inside MemoryProvider now
        expected_path = temp_workspace / ".devagent" / "memory" / "reasoning_bank.json"
        mock_memory_store.assert_called_once_with(store_path=expected_path)

    @patch("ai_dev_agent.cli.memory_provider.MEMORY_SYSTEM_AVAILABLE", False)
    def test_init_memory_system_unavailable(self, temp_workspace):
        """Test initialization when memory system is not available."""
        settings = MagicMock(spec=Settings)
        settings.enable_memory_bank = True

        enhancer = ContextEnhancer(workspace=temp_workspace, settings=settings)

        # Memory provider should still be initialized but won't have store
        assert enhancer._memory_provider is not None
        assert enhancer._memory_provider._memory_store is None

    def test_repo_map_property(self, enhancer):
        """Test repo_map property initialization."""
        # First access should create the repo map
        repo_map = enhancer.repo_map

        assert repo_map is not None
        assert enhancer._initialized is True
        assert enhancer._repo_map == repo_map

        # Second access should return cached instance
        repo_map2 = enhancer.repo_map
        assert repo_map2 is repo_map
        # Should return the same instance
        assert enhancer._repo_map is repo_map

    def test_get_important_files(self, enhancer):
        """Test file importance scoring."""
        files = [
            "README.md",
            "setup.py",
            "main.py",
            "test.txt",
            "config.yaml",
            "package.json",
            ".gitignore",
        ]

        scored_files = enhancer._get_important_files(files, max_files=5)

        # Should prioritize README, setup.py, main.py, package.json, config.yaml
        assert len(scored_files) <= 5
        # README.md should have highest score
        assert any("README.md" in str(f) for f in scored_files)

    # Removed tests for private methods (_extract_file_mentions, _find_mentioned_files)
    # These are implementation details that can change

    @patch("ai_dev_agent.core.repo_map.RepoMapManager")
    def test_get_repomap_messages(self, mock_repo_map_manager, enhancer):
        """Test getting RepoMap messages for a query."""
        mock_manager = MagicMock()
        mock_manager.context.files = ["file1.py", "file2.py"]
        mock_manager.find_files_from_query.return_value = []
        mock_manager.build_context.return_value = "RepoMap context"
        mock_repo_map_manager.get_instance.return_value = mock_manager

        query = "Fix the bug in main.py"
        original_query, messages = enhancer.get_repomap_messages(query)

        assert original_query == query
        assert messages is not None
        assert len(messages) > 0

    def test_get_memory_context(self, enhancer):
        """Test getting memory context."""
        # Mock the memory provider
        mock_memories = [
            {"id": "1", "content": "Test memory 1", "metadata": {"effectiveness": 0.8}},
            {"id": "2", "content": "Test memory 2", "metadata": {"effectiveness": 0.6}},
        ]
        enhancer._memory_provider.retrieve_relevant_memories = MagicMock(return_value=mock_memories)
        enhancer._memory_provider.format_memories_for_context = MagicMock(
            return_value="Formatted memory context"
        )

        messages, memory_ids = enhancer.get_memory_context("test query")

        assert messages is not None
        assert len(messages) == 2  # System message + assistant message
        assert memory_ids == ["1", "2"]

    def test_store_memory(self, enhancer):
        """Test storing a memory."""
        enhancer._memory_provider.store_memory = MagicMock(return_value="memory_123")

        memory_id = enhancer.store_memory(
            query="test query", response="test response", task_type="testing", success=True
        )

        assert memory_id == "memory_123"
        enhancer._memory_provider.store_memory.assert_called_once()

    def test_get_context_enhancer_singleton(self):
        """Test singleton pattern for context enhancer."""
        enhancer1 = get_context_enhancer()
        enhancer2 = get_context_enhancer()

        assert enhancer1 is enhancer2

        # Note: Current implementation doesn't create new instance for different workspace
        # This is the actual behavior - singleton is truly global
        enhancer3 = get_context_enhancer(workspace=Path("/different"))
        assert enhancer3 is enhancer1  # Same instance regardless of workspace
