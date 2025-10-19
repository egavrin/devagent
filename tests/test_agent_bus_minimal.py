"""Tests for minimal agent communication bus - production used features only.

This test file covers only the functionality actually used in production,
after removing unused methods identified by vulture analysis.
"""

import time
from unittest.mock import Mock

from ai_dev_agent.agents.communication.bus import (
    AgentBus, AgentEvent, EventType
)


class TestAgentEvent:
    """Test AgentEvent data structure."""

    def test_event_creation(self):
        """Test creating an event."""
        event = AgentEvent(
            event_type=EventType.TASK_STARTED,
            source_agent="test",
            target_agent="executor",
            data={"task_id": "123"}
        )

        assert event.event_type == EventType.TASK_STARTED
        assert event.source_agent == "test"
        assert event.target_agent == "executor"
        assert event.data["task_id"] == "123"

    def test_event_serialization(self):
        """Test event can be serialized."""
        event = AgentEvent(
            event_type=EventType.TASK_COMPLETED,
            source_agent="worker"
        )

        event_dict = event.to_dict()

        assert event_dict["event_type"] == "TASK_COMPLETED"
        assert event_dict["source_agent"] == "worker"


class TestAgentBus:
    """Test AgentBus core functionality used in production."""

    def test_bus_initialization(self):
        """Test bus initialization."""
        bus = AgentBus()

        assert bus._event_queue is not None
        assert bus._running is False

    def test_publish_event(self):
        """Test publishing an event to the queue."""
        bus = AgentBus()

        event = AgentEvent(
            event_type=EventType.TASK_STARTED,
            source_agent="test"
        )

        bus.publish(event)

        # Event should be in queue
        assert not bus._event_queue.empty()

    def test_start_stop_bus(self):
        """Test starting and stopping the bus."""
        bus = AgentBus()

        bus.start()
        assert bus._running is True

        bus.stop()
        assert bus._running is False

    def test_context_manager(self):
        """Test using bus as context manager."""
        with AgentBus() as bus:
            assert bus._running is True

            event = AgentEvent(
                event_type=EventType.TASK_STARTED,
                source_agent="test"
            )
            bus.publish(event)

        # Bus should be stopped after context
        assert bus._running is False

    def test_multiple_events(self):
        """Test publishing multiple events."""
        bus = AgentBus()

        event1 = AgentEvent(
            event_type=EventType.ERROR,
            source_agent="test"
        )

        event2 = AgentEvent(
            event_type=EventType.PROGRESS_UPDATE,
            source_agent="test"
        )

        bus.publish(event1)
        bus.publish(event2)

        # Both events should be queued
        assert bus._event_queue.qsize() == 2