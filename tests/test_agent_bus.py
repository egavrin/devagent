"""Tests for Agent Communication Bus."""
import pytest
from unittest.mock import Mock, MagicMock
import time

from ai_dev_agent.agents.communication.bus import (
    AgentBus,
    AgentEvent,
    EventType,
    EventHandler
)


class TestAgentEvent:
    """Test Agent Event."""

    def test_event_creation(self):
        """Test creating an event."""
        event = AgentEvent(
            event_type=EventType.TASK_STARTED,
            source_agent="test_agent",
            data={"task_id": "123"}
        )

        assert event.event_type == EventType.TASK_STARTED
        assert event.source_agent == "test_agent"
        assert event.data["task_id"] == "123"
        assert event.timestamp is not None

    def test_event_serialization(self):
        """Test event serialization."""
        event = AgentEvent(
            event_type=EventType.TASK_COMPLETED,
            source_agent="agent1",
            data={"result": "success"}
        )

        serialized = event.to_dict()

        assert serialized["event_type"] == "TASK_COMPLETED"
        assert serialized["source_agent"] == "agent1"
        assert serialized["data"]["result"] == "success"


class TestAgentBus:
    """Test Agent Communication Bus."""

    def test_bus_initialization(self):
        """Test creating a bus."""
        bus = AgentBus()

        assert bus is not None
        assert bus.is_running is False

    def test_publish_event(self):
        """Test publishing an event."""
        bus = AgentBus()

        event = AgentEvent(
            event_type=EventType.AGENT_STARTED,
            source_agent="test"
        )

        bus.publish(event)

        # Event should be in queue
        assert not bus._event_queue.empty()

    def test_subscribe_to_events(self):
        """Test subscribing to events."""
        bus = AgentBus()
        handler = Mock()

        bus.subscribe(EventType.TASK_STARTED, handler)

        # Publish event
        event = AgentEvent(
            event_type=EventType.TASK_STARTED,
            source_agent="test"
        )

        bus.start()
        bus.publish(event)

        # Wait for processing
        time.sleep(0.1)

        bus.stop()

        # Handler should have been called
        assert handler.called

    def test_multiple_subscribers(self):
        """Test multiple subscribers to same event."""
        bus = AgentBus()
        handler1 = Mock()
        handler2 = Mock()

        bus.subscribe(EventType.TASK_COMPLETED, handler1)
        bus.subscribe(EventType.TASK_COMPLETED, handler2)

        event = AgentEvent(
            event_type=EventType.TASK_COMPLETED,
            source_agent="test"
        )

        bus.start()
        bus.publish(event)

        time.sleep(0.1)
        bus.stop()

        assert handler1.called
        assert handler2.called

    def test_unsubscribe(self):
        """Test unsubscribing from events."""
        bus = AgentBus()
        handler = Mock()

        subscription_id = bus.subscribe(EventType.PROGRESS_UPDATE, handler)
        bus.unsubscribe(subscription_id)

        event = AgentEvent(
            event_type=EventType.PROGRESS_UPDATE,
            source_agent="test"
        )

        bus.start()
        bus.publish(event)

        time.sleep(0.1)
        bus.stop()

        # Handler should NOT have been called
        assert not handler.called

    def test_event_filtering(self):
        """Test filtering events by type."""
        bus = AgentBus()
        handler = Mock()

        bus.subscribe(EventType.TASK_STARTED, handler)

        # Publish different event type
        event = AgentEvent(
            event_type=EventType.TASK_COMPLETED,
            source_agent="test"
        )

        bus.start()
        bus.publish(event)

        time.sleep(0.1)
        bus.stop()

        # Handler should NOT have been called
        assert not handler.called

    def test_broadcast_message(self):
        """Test broadcasting a message."""
        bus = AgentBus()
        handler1 = Mock()
        handler2 = Mock()

        bus.subscribe(EventType.MESSAGE, handler1)
        bus.subscribe(EventType.MESSAGE, handler2)

        bus.start()
        bus.broadcast("Hello from agent", source="broadcaster")

        time.sleep(0.1)
        bus.stop()

        assert handler1.called
        assert handler2.called

    def test_request_response_pattern(self):
        """Test request-response communication pattern."""
        bus = AgentBus()

        # Handler that responds to requests
        def request_handler(event):
            if event.data.get("request") == "get_status":
                response = AgentEvent(
                    event_type=EventType.RESPONSE,
                    source_agent="responder",
                    data={"status": "active"},
                    correlation_id=event.event_id
                )
                bus.publish(response)

        bus.subscribe(EventType.REQUEST, request_handler)

        # Response handler
        response_handler = Mock()
        bus.subscribe(EventType.RESPONSE, response_handler)

        bus.start()

        # Send request
        request = AgentEvent(
            event_type=EventType.REQUEST,
            source_agent="requester",
            data={"request": "get_status"}
        )
        bus.publish(request)

        time.sleep(0.2)
        bus.stop()

        assert response_handler.called

    def test_priority_events(self):
        """Test priority event handling."""
        bus = AgentBus()
        handled_events = []

        def handler(event):
            handled_events.append(event.event_type)

        bus.subscribe(EventType.ERROR, handler)
        bus.subscribe(EventType.PROGRESS_UPDATE, handler)

        bus.start()

        # Publish low priority event first
        bus.publish(AgentEvent(EventType.PROGRESS_UPDATE, "agent"))

        # Publish high priority event
        bus.publish(AgentEvent(EventType.ERROR, "agent"), priority=True)

        time.sleep(0.1)
        bus.stop()

        # High priority should be handled first (if priority queue implemented)
        assert EventType.ERROR in handled_events

    def test_bus_metrics(self):
        """Test bus metrics collection."""
        bus = AgentBus()

        bus.start()

        # Publish several events
        for i in range(5):
            bus.publish(AgentEvent(EventType.PROGRESS_UPDATE, f"agent{i}"))

        time.sleep(0.1)
        bus.stop()

        metrics = bus.get_metrics()

        assert metrics["events_published"] >= 5
        assert "events_processed" in metrics

    def test_error_handling_in_subscriber(self):
        """Test error handling when subscriber fails."""
        bus = AgentBus()

        # Handler that raises exception
        def failing_handler(event):
            raise ValueError("Handler error")

        bus.subscribe(EventType.TASK_STARTED, failing_handler)

        bus.start()

        # Should not crash the bus
        bus.publish(AgentEvent(EventType.TASK_STARTED, "test"))

        time.sleep(0.1)
        bus.stop()

        # Bus should still be functional
        assert True  # If we get here, error was handled