from collections import defaultdict
from types import SimpleNamespace

import pytest

from ai_dev_agent.cli.context_enhancer import ContextEnhancer
from ai_dev_agent.core.utils.config import Settings


class FakeRepoMap:
    def __init__(self, responses):
        file_info = SimpleNamespace(
            path="app.py",
            language="python",
            file_name="app.py",
            file_stem="app",
            path_parts=("app.py",),
            dependencies=set(),
            symbols=[],
            imports=[],
            references={},
            symbols_used=[],
        )
        self.context = SimpleNamespace(
            files={"app.py": file_info},
            symbol_index=defaultdict(set),
        )
        self.responses = list(responses)
        self.calls = []

    def get_ranked_files(self, mentioned_files, mentioned_symbols, max_files):
        self.calls.append(
            (set(mentioned_files), set(mentioned_symbols), max_files),
        )
        idx = min(len(self.responses) - 1, len(self.calls) - 1)
        return self.responses[idx]

    def scan_repository(self):
        return None


@pytest.fixture
def disabled_settings():
    settings = Settings()
    settings.enable_memory_bank = False
    settings.enable_playbook = False
    settings.enable_dynamic_instructions = False
    return settings


def test_enhance_query_with_context_adds_ranked_files(monkeypatch, tmp_path, disabled_settings):
    repo_map = FakeRepoMap([[("app.py", 12.0)]])
    repo_map.context.symbol_index["Helper"].add("app.py")

    monkeypatch.setattr(
        "ai_dev_agent.cli.context_enhancer.RepoMapManager.get_instance",
        lambda workspace: repo_map,
    )

    enhancer = ContextEnhancer(workspace=tmp_path, settings=disabled_settings)
    query = "How does Helper work in app.py?"

    enhanced = enhancer.enhance_query_with_context(query, max_files=5)

    assert "[Automatic Context from RepoMap]" in enhanced
    assert "app.py" in enhanced
    assert repo_map.calls
    mentioned_files, mentioned_symbols, _ = repo_map.calls[0]
    assert "app.py" in mentioned_files
    assert "Helper" in mentioned_symbols


def test_get_repomap_messages_uses_fallback_tiers(monkeypatch, tmp_path, disabled_settings):
    responses = [
        [],  # Tier 1 attempt
        [],  # Tier 2 attempt
        [("core/main.py", 5.0)],  # Tier 3 fallback result
    ]
    repo_map = FakeRepoMap(responses)
    repo_map.context.files["core/main.py"] = SimpleNamespace(
        path="core/main.py",
        language="python",
        file_name="main.py",
        file_stem="main",
        path_parts=("core", "main.py"),
        dependencies=set(),
        symbols=[],
        imports=[],
        references={},
        symbols_used=[],
    )

    monkeypatch.setattr(
        "ai_dev_agent.cli.context_enhancer.RepoMapManager.get_instance",
        lambda workspace: repo_map,
    )

    enhancer = ContextEnhancer(workspace=tmp_path, settings=disabled_settings)

    monkeypatch.setattr(
        enhancer,
        "_get_important_files",
        lambda all_files, max_files: [("core/main.py", 7.0)],
    )

    query = "Investigate ServiceManager symbols"
    query, messages = enhancer.get_repomap_messages(
        query,
        max_files=5,
        additional_files={"seed.py"},
        additional_symbols={"ServiceManager"},
    )

    assert messages is not None
    assert len(repo_map.calls) >= 3
    assert "high-level view" in messages[1]["content"]


def test_enhance_query_with_missing_mentions(monkeypatch, tmp_path, disabled_settings):
    repo_map = FakeRepoMap([[]])
    repo_map.context.files.clear()

    monkeypatch.setattr(
        "ai_dev_agent.cli.context_enhancer.RepoMapManager.get_instance",
        lambda workspace: repo_map,
    )

    enhancer = ContextEnhancer(workspace=tmp_path, settings=disabled_settings)

    enhanced = enhancer.enhance_query_with_context("Inspect missing.py and SymbolX")

    assert "[RepoMap Notice]" in enhanced
