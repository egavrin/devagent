"""Agent Communication Bus for inter-agent messaging."""
from __future__ import annotations

import uuid
import threading
import queue
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Callable


class EventType(Enum):
    """Types of events in the system."""
    AGENT_STARTED = "AGENT_STARTED"
    AGENT_STOPPED = "AGENT_STOPPED"
    TASK_STARTED = "TASK_STARTED"
    TASK_COMPLETED = "TASK_COMPLETED"
    TASK_FAILED = "TASK_FAILED"
    PROGRESS_UPDATE = "PROGRESS_UPDATE"
    MESSAGE = "MESSAGE"
    REQUEST = "REQUEST"
    RESPONSE = "RESPONSE"
    ERROR = "ERROR"
    WARNING = "WARNING"


@dataclass
class AgentEvent:
    """Event published on the bus."""
    event_type: EventType
    source_agent: str
    data: Dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    correlation_id: Optional[str] = None
    target_agent: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        result = asdict(self)
        result["event_type"] = self.event_type.value
        return result


# Type alias for event handlers
EventHandler = Callable[[AgentEvent], None]


class AgentBus:
    """Central communication bus for agents."""

    def __init__(self):
        """Initialize the agent bus."""
        self._event_queue: queue.Queue = queue.Queue()
        self._subscribers: Dict[EventType, List[tuple[str, EventHandler]]] = {}
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._metrics = {
            "events_published": 0,
            "events_processed": 0,
            "errors": 0
        }
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        """Check if bus is running."""
        return self._running

    def start(self) -> None:
        """Start the event bus."""
        if self._running:
            return

        self._running = True
        self._worker_thread = threading.Thread(target=self._process_events, daemon=True)
        self._worker_thread.start()

    def stop(self) -> None:
        """Stop the event bus."""
        self._running = False

        if self._worker_thread:
            self._worker_thread.join(timeout=2.0)
            self._worker_thread = None

    def publish(self, event: AgentEvent, priority: bool = False) -> None:
        """
        Publish an event on the bus.

        Args:
            event: Event to publish
            priority: If True, process with higher priority
        """
        with self._lock:
            self._metrics["events_published"] += 1

        # For now, priority is ignored (could use PriorityQueue)
        self._event_queue.put(event)

    def subscribe(
        self,
        event_type: EventType,
        handler: EventHandler
    ) -> str:
        """
        Subscribe to events of a specific type.

        Args:
            event_type: Type of events to subscribe to
            handler: Function to call when event occurs

        Returns:
            Subscription ID for later unsubscription
        """
        subscription_id = str(uuid.uuid4())

        if event_type not in self._subscribers:
            self._subscribers[event_type] = []

        self._subscribers[event_type].append((subscription_id, handler))

        return subscription_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe from events.

        Args:
            subscription_id: ID returned from subscribe()

        Returns:
            True if unsubscribed, False if ID not found
        """
        for event_type, subscribers in self._subscribers.items():
            for i, (sub_id, _) in enumerate(subscribers):
                if sub_id == subscription_id:
                    del subscribers[i]
                    return True

        return False

    def broadcast(self, message: str, source: str, data: Optional[Dict[str, Any]] = None) -> None:
        """
        Broadcast a message to all subscribers.

        Args:
            message: Message content
            source: Source agent name
            data: Additional data
        """
        event_data = {"message": message}
        if data:
            event_data.update(data)

        event = AgentEvent(
            event_type=EventType.MESSAGE,
            source_agent=source,
            data=event_data
        )

        self.publish(event)

    def _process_events(self) -> None:
        """Process events from the queue (runs in background thread)."""
        while self._running:
            try:
                # Get event with timeout to allow checking _running flag
                event = self._event_queue.get(timeout=0.1)

                # Notify subscribers
                self._notify_subscribers(event)

                with self._lock:
                    self._metrics["events_processed"] += 1

                self._event_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                # Log error but don't crash the bus
                with self._lock:
                    self._metrics["errors"] += 1

    def _notify_subscribers(self, event: AgentEvent) -> None:
        """
        Notify all subscribers of an event.

        Args:
            event: Event to notify about
        """
        subscribers = self._subscribers.get(event.event_type, [])

        for subscription_id, handler in subscribers:
            try:
                handler(event)
            except Exception as e:
                # Log error but continue notifying other subscribers
                with self._lock:
                    self._metrics["errors"] += 1

    def get_metrics(self) -> Dict[str, int]:
        """
        Get bus metrics.

        Returns:
            Dictionary of metrics
        """
        with self._lock:
            return self._metrics.copy()

    def clear_metrics(self) -> None:
        """Clear metrics."""
        with self._lock:
            self._metrics = {
                "events_published": 0,
                "events_processed": 0,
                "errors": 0
            }

    def wait_for_completion(self, timeout: Optional[float] = None) -> None:
        """
        Wait for all events to be processed.

        Args:
            timeout: Maximum time to wait (seconds)
        """
        self._event_queue.join()  # Wait for all tasks to be done

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False