"""Tests for session cleanup and memory management."""

import time
from datetime import datetime, timedelta

from ai_dev_agent.session.manager import SessionManager


class TestSessionCleanup:
    """Test session cleanup functionality."""

    def test_session_ttl_cleanup(self):
        """Test that expired sessions are cleaned up."""
        manager = SessionManager()

        # Create a few sessions
        session1 = manager.ensure_session("test-1")
        session2 = manager.ensure_session("test-2")
        session3 = manager.ensure_session("test-3")

        # Manually set last_accessed times
        now = datetime.now()
        session1.last_accessed = now - timedelta(minutes=45)  # Expired
        session2.last_accessed = now - timedelta(minutes=20)  # Not expired
        session3.last_accessed = now - timedelta(minutes=35)  # Expired

        # Run cleanup
        removed_count = manager.cleanup_expired_sessions()

        # Check that expired sessions were removed
        assert removed_count == 2
        assert not manager.has_session("test-1")
        assert manager.has_session("test-2")
        assert not manager.has_session("test-3")

    def test_max_sessions_eviction(self):
        """Test that oldest session is evicted when at max capacity."""
        manager = SessionManager()
        manager.set_max_sessions(3)

        # Create sessions up to max
        manager.ensure_session("test-1")
        time.sleep(0.1)
        manager.ensure_session("test-2")
        time.sleep(0.1)
        manager.ensure_session("test-3")

        # All should exist
        assert manager.has_session("test-1")
        assert manager.has_session("test-2")
        assert manager.has_session("test-3")

        # Create one more - oldest should be evicted
        manager.ensure_session("test-4")

        # test-1 should be evicted as it's the oldest
        assert not manager.has_session("test-1")
        assert manager.has_session("test-2")
        assert manager.has_session("test-3")
        assert manager.has_session("test-4")

    def test_session_last_accessed_updates(self):
        """Test that last_accessed is updated on session access."""
        manager = SessionManager()
        session = manager.ensure_session("test-session")

        initial_time = session.last_accessed
        time.sleep(0.1)

        # Access the session
        accessed_session = manager.get_session("test-session")

        # Last accessed should be updated
        assert accessed_session.last_accessed > initial_time

    def test_session_stats(self):
        """Test session statistics reporting."""
        manager = SessionManager()

        # Create a few sessions
        manager.ensure_session("test-1")
        manager.ensure_session("test-2")

        # Add some messages
        manager.add_user_message("test-1", "Hello")
        manager.add_assistant_message("test-1", "Hi there")

        stats = manager.get_session_stats()

        assert stats["total_sessions"] == 2
        assert stats["max_sessions"] == manager._max_sessions
        assert stats["ttl_minutes"] == 30  # Default TTL
        assert len(stats["sessions"]) == 2

        # Check session details
        session_stats = next(s for s in stats["sessions"] if s["id"] == "test-1")
        assert session_stats["message_count"] == 2
        assert session_stats["age_seconds"] >= 0
        assert session_stats["idle_seconds"] >= 0

    def test_session_ttl_configuration(self):
        """Test configuring session TTL."""
        manager = SessionManager()

        # Set custom TTL
        manager.set_session_ttl(5)  # 5 minutes

        # Create a session
        session = manager.ensure_session("test-ttl")

        # Set last accessed to 6 minutes ago
        session.last_accessed = datetime.now() - timedelta(minutes=6)

        # Run cleanup
        removed_count = manager.cleanup_expired_sessions()

        assert removed_count == 1
        assert not manager.has_session("test-ttl")

    def test_cleanup_timer_starts_automatically(self):
        """Test that cleanup timer starts when manager is created."""
        manager = SessionManager()

        # Timer should be set
        assert manager._cleanup_timer is not None
        assert manager._cleanup_timer.daemon is True

        # Cleanup
        manager.shutdown()
        assert manager._cleanup_timer is None

    def test_evict_oldest_with_empty_sessions(self):
        """Test that evict_oldest handles empty sessions gracefully."""
        manager = SessionManager()

        # Should not raise error
        manager._evict_oldest_session()

        # Verify no sessions
        assert len(manager._sessions) == 0

    def test_cleanup_with_explicit_times(self):
        """Test cleanup with explicitly set times."""
        manager = SessionManager()

        # Create sessions
        session1 = manager.ensure_session("old-session")
        session2 = manager.ensure_session("new-session")

        # Manually set last_accessed times
        now = datetime.now()
        session1.last_accessed = now - timedelta(minutes=40)  # Older than TTL
        session2.last_accessed = now - timedelta(minutes=10)  # Within TTL

        # Run cleanup
        removed = manager.cleanup_expired_sessions()

        # Only old session should be removed (40 min > 30 min TTL)
        assert removed == 1
        assert not manager.has_session("old-session")
        assert manager.has_session("new-session")
