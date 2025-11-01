"""Tests for the CLI memory provider helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ai_dev_agent.cli.memory_provider import MemoryProvider


@pytest.fixture
def workspace(tmp_path):
    """Provide a temporary workspace path."""
    return tmp_path


def _make_provider(workspace, store=None, *, enable_memory=True):
    provider = MemoryProvider(workspace, enable_memory=enable_memory)
    provider._memory_store = store
    return provider


def test_retrieve_relevant_memories_formats_results(workspace):
    with patch("ai_dev_agent.cli.memory_provider.MemoryStore") as mock_store_cls:
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store

        memory = SimpleNamespace(
            memory_id="mem-1",
            content="Investigated intermittent CI failure",
            effectiveness_score=0.9,
            task_type="debugging",
        )
        mock_store.search_similar.return_value = [(memory, 0.82)]

        provider = _make_provider(workspace, mock_store)

        memories = provider.retrieve_relevant_memories("Fix CI issue", task_type="debugging")

        assert memories is not None
        assert memories[0]["id"] == "mem-1"
        assert memories[0]["metadata"]["task_type"] == "debugging"
        assert pytest.approx(memories[0]["metadata"]["similarity"]) == 0.82


def test_store_memory_uses_distiller(workspace):
    with (
        patch("ai_dev_agent.cli.memory_provider.MemoryStore") as mock_store_cls,
        patch("ai_dev_agent.cli.memory_provider.MemoryDistiller") as mock_distiller_cls,
    ):
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store
        memory = SimpleNamespace(memory_id="mem-2", title="CI fix")
        mock_distiller = mock_distiller_cls.return_value
        mock_distiller.distill_from_session.return_value = memory
        mock_store.add_memory.return_value = "mem-2"

        provider = _make_provider(workspace, mock_store)

        result = provider.store_memory("Fix CI failure", "Resolved missing dependency.")

        assert result == "mem-2"
        mock_distiller.distill_from_session.assert_called_once()
        mock_store.add_memory.assert_called_once_with(memory)


def test_track_memory_effectiveness_updates_store(workspace):
    with patch("ai_dev_agent.cli.memory_provider.MemoryStore") as mock_store_cls:
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store

        provider = _make_provider(workspace, mock_store)

        provider.track_memory_effectiveness(["mem-3"], success=True, feedback="helpful")

        mock_store.update_effectiveness.assert_called_once_with("mem-3", 0.1, "helpful")


def test_collect_statistics_returns_snapshot(workspace):
    with patch("ai_dev_agent.cli.memory_provider.MemoryStore") as mock_store_cls:
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store
        mock_store._memories = {"mem-4": object()}
        mock_store._usage_stats = {"session-1": {"successes": 1}}

        provider = _make_provider(workspace, mock_store)

        stats = provider.collect_statistics()

        assert stats["total_memories"] == 1
        assert stats["usage_stats"]["session-1"]["successes"] == 1


def test_retrieve_memories_returns_none_without_store(workspace):
    provider = _make_provider(workspace, store=None, enable_memory=False)
    assert provider.retrieve_relevant_memories("anything") is None


def test_retrieve_memories_handles_empty_results(workspace):
    with patch("ai_dev_agent.cli.memory_provider.MemoryStore") as mock_store_cls:
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store
        mock_store.search_similar.return_value = []

        provider = _make_provider(workspace, mock_store)
        assert provider.retrieve_relevant_memories("query") is None


def test_retrieve_memories_handles_exception(workspace):
    with patch("ai_dev_agent.cli.memory_provider.MemoryStore") as mock_store_cls:
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store
        mock_store.search_similar.side_effect = RuntimeError("bad search")

        provider = _make_provider(workspace, mock_store)
        assert provider.retrieve_relevant_memories("query") is None


def test_store_memory_returns_none_without_store(workspace):
    provider = _make_provider(workspace, store=None, enable_memory=False)
    assert provider.store_memory("q", "r") is None


def test_store_memory_handles_distiller_none(workspace):
    with (
        patch("ai_dev_agent.cli.memory_provider.MemoryStore") as mock_store_cls,
        patch("ai_dev_agent.cli.memory_provider.MemoryDistiller") as mock_distiller_cls,
    ):
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store

        mock_distiller = mock_distiller_cls.return_value
        mock_distiller.distill_from_session.return_value = None

        provider = _make_provider(workspace, mock_store)
        assert provider.store_memory("q", "r") is None
        mock_store.add_memory.assert_not_called()


def test_store_memory_handles_distiller_exception(workspace):
    with (
        patch("ai_dev_agent.cli.memory_provider.MemoryStore") as mock_store_cls,
        patch("ai_dev_agent.cli.memory_provider.MemoryDistiller") as mock_distiller_cls,
    ):
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store

        mock_distiller = mock_distiller_cls.return_value
        mock_distiller.distill_from_session.side_effect = RuntimeError("boom")

        provider = _make_provider(workspace, mock_store)
        assert provider.store_memory("q", "r") is None
        mock_store.add_memory.assert_not_called()


def test_format_memories_for_context_labels_effectiveness(workspace):
    provider = _make_provider(workspace, store=None, enable_memory=False)
    formatted = provider.format_memories_for_context(
        [
            {"content": "High win", "metadata": {"effectiveness": 0.9}},
            {"content": "Middling", "metadata": {"effectiveness": 0.5}},
            {"content": "Low win", "metadata": {"effectiveness": 0.1}},
        ]
    )

    assert "Highly effective" in formatted
    assert "limited success" in formatted.lower()


def test_has_store_property_reflects_state(workspace):
    provider = _make_provider(workspace, store=None, enable_memory=False)
    assert provider.has_store is False

    with patch("ai_dev_agent.cli.memory_provider.MemoryStore") as mock_store_cls:
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store

        provider_with_store = _make_provider(workspace, mock_store)
        assert provider_with_store.has_store is True


def test_distill_and_store_memory_success(workspace):
    with (
        patch("ai_dev_agent.cli.memory_provider.MemoryStore") as mock_store_cls,
        patch("ai_dev_agent.cli.memory_provider.MemoryDistiller") as mock_distiller_cls,
    ):
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store
        mock_store.add_memory.return_value = "mem-abc"

        distilled = SimpleNamespace(memory_id="mem-abc", title="Saved memory")
        mock_distiller_cls.return_value.distill_from_session.return_value = distilled

        provider = _make_provider(workspace, mock_store)
        memory_id = provider.distill_and_store_memory(
            "session-1", [{"role": "user", "content": "hi"}]
        )

        assert memory_id == "mem-abc"
        mock_store.add_memory.assert_called_once_with(distilled)


def test_distill_and_store_memory_handles_empty_messages(workspace):
    with patch("ai_dev_agent.cli.memory_provider.MemoryStore") as mock_store_cls:
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store

        provider = _make_provider(workspace, mock_store)
        assert provider.distill_and_store_memory("session", []) is None
        mock_store.add_memory.assert_not_called()


def test_distill_and_store_memory_handles_distiller_failure(workspace):
    with (
        patch("ai_dev_agent.cli.memory_provider.MemoryStore") as mock_store_cls,
        patch("ai_dev_agent.cli.memory_provider.MemoryDistiller") as mock_distiller_cls,
    ):
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store

        distiller = mock_distiller_cls.return_value
        distiller.distill_from_session.side_effect = RuntimeError("oops")

        provider = _make_provider(workspace, mock_store)
        assert (
            provider.distill_and_store_memory("session", [{"role": "user", "content": "hi"}])
            is None
        )
        mock_store.add_memory.assert_not_called()


def test_track_memory_effectiveness_without_store(workspace):
    provider = _make_provider(workspace, store=None, enable_memory=False)
    provider.track_memory_effectiveness(["id-1"], success=True)
    # Should not raise and no work performed


def test_record_query_outcome_updates_usage_stats(workspace):
    with patch("ai_dev_agent.cli.memory_provider.MemoryStore") as mock_store_cls:
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store
        mock_store._usage_stats = {}

        provider = _make_provider(workspace, mock_store)
        provider.record_query_outcome(
            session_id="s1",
            success=False,
            tools_used=["read", "run"],
            task_type="debugging",
            error_type="RuntimeError",
            duration_seconds=3.2,
        )

        stats = mock_store._usage_stats["s1"]
        assert stats["failures"] == 1
        assert stats["tools"]["read"] == 1
        assert stats["last_error"] == "RuntimeError"
        assert stats["durations"] == [3.2]


def test_record_query_outcome_without_store(workspace):
    provider = _make_provider(workspace, store=None, enable_memory=False)
    provider.record_query_outcome(
        session_id="s1",
        success=True,
        tools_used=[],
        task_type="testing",
    )  # Should not raise


def test_collect_statistics_without_store(workspace):
    provider = _make_provider(workspace, store=None)
    assert provider.collect_statistics() == {}
