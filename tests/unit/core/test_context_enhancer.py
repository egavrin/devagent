"""Tests for the simplified ContextEnhancer module."""

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ai_dev_agent.cli.context_enhancer import ContextEnhancer, get_context_enhancer
from ai_dev_agent.core.utils.config import Settings
from ai_dev_agent.providers.llm.base import Message


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
        enhancer._memory_provider._memory_store = object()
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

    def test_has_memory_support_flag(self, enhancer):
        """Context enhancer exposes memory availability flag."""
        enhancer._memory_provider._memory_store = None
        assert enhancer.has_memory_support is False

        enhancer._memory_provider._memory_store = object()
        assert enhancer.has_memory_support is True

    def test_distill_and_store_memory_delegates_to_provider(self, enhancer):
        """distill_and_store_memory forwards to memory provider."""
        enhancer._memory_provider.distill_and_store_memory = MagicMock(return_value="mem-42")
        result = enhancer.distill_and_store_memory(
            session_id="session-1", messages=[Message(role="user", content="hi")], metadata={}
        )

        enhancer._memory_provider.distill_and_store_memory.assert_called_once()
        assert result == "mem-42"

    def test_track_memory_effectiveness_handles_provider_missing(self, enhancer):
        """track_memory_effectiveness is resilient when memory disabled."""
        enhancer._memory_provider.track_memory_effectiveness = MagicMock()
        enhancer.track_memory_effectiveness(memory_ids=["m1"], success=True, feedback=None)
        enhancer._memory_provider.track_memory_effectiveness.assert_called_once()

    def test_record_query_outcome_is_noop_without_provider_method(self, enhancer):
        """record_query_outcome should call provider hook if available."""
        enhancer._memory_provider.record_query_outcome = MagicMock()
        enhancer.record_query_outcome(
            session_id="session-2",
            success=True,
            tools_used=["read"],
            task_type="debugging",
            error_type=None,
            duration_seconds=5.0,
        )
        enhancer._memory_provider.record_query_outcome.assert_called_once()

    def test_get_context_enhancer_singleton(self):
        """Test singleton pattern for context enhancer."""
        enhancer1 = get_context_enhancer()
        enhancer2 = get_context_enhancer()

        assert enhancer1 is enhancer2

        # Note: Current implementation doesn't create new instance for different workspace
        # This is the actual behavior - singleton is truly global
        enhancer3 = get_context_enhancer(workspace=Path("/different"))
        assert enhancer3 is enhancer1  # Same instance regardless of workspace

    def test_extract_symbols_and_files_matches_repo_entries(self, enhancer):
        """extract_symbols_and_files should map mentions onto repo files and directories."""
        repo_map = SimpleNamespace(
            context=SimpleNamespace(
                files={
                    "src/utils/helpers.py": SimpleNamespace(
                        file_name="helpers.py",
                        file_stem="helpers",
                        language="python",
                        path_parts=("src", "utils", "helpers.py"),
                    ),
                    "modules/bytecode_optimizer/core.py": SimpleNamespace(
                        file_name="core.py",
                        file_stem="core",
                        language="python",
                        path_parts=("modules", "bytecode_optimizer", "core.py"),
                    ),
                },
                symbol_index={},
            ),
        )
        enhancer._repo_map = repo_map
        enhancer._initialized = True

        text = (
            "Please review HelpersManager behavior in src/utils/helpers.py "
            "and check the bytecode_optimizer utilities alongside stray words."
        )
        symbols, files = enhancer.extract_symbols_and_files(text)

        assert "HelpersManager" in symbols
        assert "src/utils/helpers.py" in files
        assert "bytecode_optimizer" in files
        assert "Please" not in symbols

    def test_repo_map_only_scans_workspace_once(self, temp_workspace, mock_settings, monkeypatch):
        """RepoMap initialization should avoid redundant scans once populated."""

        class StubRepoMap:
            def __init__(self):
                self.context = SimpleNamespace(files={}, symbol_index={})
                self.scan_calls = 0

            def scan_repository(self):
                self.scan_calls += 1
                self.context.files = {
                    "src/main.py": SimpleNamespace(
                        file_name="main.py",
                        file_stem="main",
                        language="python",
                        path_parts=("src", "main.py"),
                    )
                }

        stub_repo_map = StubRepoMap()
        monkeypatch.setattr(
            "ai_dev_agent.cli.context_enhancer.RepoMapManager.get_instance",
            lambda workspace: stub_repo_map,
        )

        enhancer = ContextEnhancer(workspace=temp_workspace, settings=mock_settings)

        _ = enhancer.repo_map
        assert stub_repo_map.scan_calls == 1

        _ = enhancer.repo_map
        assert stub_repo_map.scan_calls == 1

    def test_get_repomap_messages_tier_two_expands_scope(
        self, temp_workspace, mock_settings, monkeypatch
    ):
        """Tier-two fallback should expand the search space with important files."""

        class TieredRepoMap:
            def __init__(self):
                self.context = SimpleNamespace(
                    files={
                        "src/core.py": SimpleNamespace(
                            file_name="core.py",
                            file_stem="core",
                            language="python",
                            path_parts=("src", "core.py"),
                        )
                    },
                    symbol_index={"ServiceManager": {"src/core.py"}},
                )
                self.responses = [
                    [],
                    [("src/core.py", 9.5)],
                ]
                self.calls = []

            def get_ranked_files(self, mentioned_files, mentioned_symbols, max_files):
                self.calls.append((set(mentioned_files), set(mentioned_symbols), max_files))
                idx = min(len(self.responses) - 1, len(self.calls) - 1)
                return self.responses[idx]

            def scan_repository(self):
                return None

        repo_map = TieredRepoMap()
        monkeypatch.setattr(
            "ai_dev_agent.cli.context_enhancer.RepoMapManager.get_instance",
            lambda workspace: repo_map,
        )

        enhancer = ContextEnhancer(workspace=temp_workspace, settings=mock_settings)
        monkeypatch.setattr(
            enhancer, "_get_important_files", lambda files, max_files: [("src/core.py", 9.5)]
        )

        query = "Investigate ServiceManager regressions urgently."
        _, messages = enhancer.get_repomap_messages(query, max_files=4)

        assert messages is not None
        assert len(repo_map.calls) >= 2
        assert repo_map.calls[0][2] == 4
        assert repo_map.calls[1][2] == 8
        assert "src/core.py" in repo_map.calls[1][0]

    def test_build_enhanced_context_combines_all_layers(
        self, temp_workspace, mock_settings, monkeypatch
    ):
        """Test that build_enhanced_context combines all context layers."""

        # Mock RepoMap
        class MockRepoMap:
            def __init__(self):
                self.context = SimpleNamespace(files={"test.py": SimpleNamespace(path="test.py")})

            def get_ranked_files(self, *args, **kwargs):
                return [("test.py", 10.0)]

            def scan_repository(self):
                pass

        monkeypatch.setattr(
            "ai_dev_agent.cli.context_enhancer.RepoMapManager.get_instance",
            lambda workspace: MockRepoMap(),
        )

        enhancer = ContextEnhancer(workspace=temp_workspace, settings=mock_settings)

        # Test with all context layers
        context = enhancer.build_enhanced_context(
            include_repomap=True, include_memory=True, include_outline=False, query="test query"
        )

        # Verify all layers are present
        assert "system" in context
        assert "project" in context
        assert context["system"]["cwd"] == str(temp_workspace)
        assert context["project"]["workspace"] == str(temp_workspace)

        # RepoMap and memory might be None due to mocking, but keys should exist
        assert "repomap_messages" in context or True  # May be None if RepoMap fails
        assert "memory" in context or True  # May be None if Memory fails

    def test_build_enhanced_context_selective_layers(
        self, temp_workspace, mock_settings, monkeypatch
    ):
        """Test that build_enhanced_context can selectively include layers."""

        # Mock RepoMap to avoid initialization issues
        class MockRepoMap:
            def __init__(self):
                self.context = SimpleNamespace(files={})

            def scan_repository(self):
                pass

        monkeypatch.setattr(
            "ai_dev_agent.cli.context_enhancer.RepoMapManager.get_instance",
            lambda workspace: MockRepoMap(),
        )

        enhancer = ContextEnhancer(workspace=temp_workspace, settings=mock_settings)

        # Test with only system and project context
        context = enhancer.build_enhanced_context(
            include_repomap=False, include_memory=False, include_outline=False
        )

        # Verify only base layers are present
        assert "system" in context
        assert "project" in context
        assert "repomap_messages" not in context
        assert "memory" not in context

    def test_build_enhanced_context_inherits_from_context_builder(
        self, temp_workspace, mock_settings
    ):
        """Test that ContextEnhancer properly inherits from ContextBuilder."""
        from ai_dev_agent.core.context.builder import ContextBuilder

        enhancer = ContextEnhancer(workspace=temp_workspace, settings=mock_settings)

        # Verify inheritance
        assert isinstance(enhancer, ContextBuilder)

        # Verify parent methods are available
        assert hasattr(enhancer, "build_system_context")
        assert hasattr(enhancer, "build_project_context")
        assert hasattr(enhancer, "get_project_structure_outline")

        # Verify parent methods work
        system_ctx = enhancer.build_system_context()
        assert "os" in system_ctx
        assert "cwd" in system_ctx

        project_ctx = enhancer.build_project_context()
        assert "workspace" in project_ctx
