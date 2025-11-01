"""Deterministic tests for performance helpers and communication bus."""

from __future__ import annotations

from typing import Callable

import pytest

from ai_dev_agent.agents.communication.bus import AgentBus, AgentEvent, EventType


class PerformanceMetrics:
    """Collect simple performance measurements."""

    def __init__(self):
        self.measurements: dict[str, list[float]] = {}

    def record(self, operation: str, duration: float) -> None:
        self.measurements.setdefault(operation, []).append(duration)

    def get_stats(self, operation: str) -> dict[str, float]:
        durations = self.measurements.get(operation, [])
        if not durations:
            return {}
        total = sum(durations)
        return {
            "count": len(durations),
            "total": total,
            "mean": total / len(durations),
            "min": min(durations),
            "max": max(durations),
        }

    def report(self) -> str:
        lines = ["Performance Metrics Report", "=" * 50]
        for operation, durations in sorted(self.measurements.items()):
            stats = self.get_stats(operation)
            lines.append(f"\n{operation}:")
            lines.append(f"  Count: {stats['count']}")
            lines.append(f"  Total: {stats['total']:.4f}s")
            lines.append(f"  Mean: {stats['mean']:.4f}s")
            lines.append(f"  Min: {stats['min']:.4f}s")
            lines.append(f"  Max: {stats['max']:.4f}s")
        return "\n".join(lines)


@pytest.fixture
def metrics():
    return PerformanceMetrics()


def test_metrics_record_and_report(metrics):
    metrics.record("delegation", 0.05)
    metrics.record("delegation", 0.10)
    metrics.record("planning", 0.20)

    delegation = metrics.get_stats("delegation")
    report = metrics.report()

    assert delegation["count"] == 2
    assert pytest.approx(delegation["mean"], rel=1e-3) == 0.075
    assert "Performance Metrics Report" in report
    assert "planning" in report


def test_metrics_handles_unknown_operation(metrics):
    assert metrics.get_stats("missing") == {}


def _subscribe(bus: AgentBus, handler: Callable[[AgentEvent], None]) -> None:
    bus.subscribe(EventType.MESSAGE, handler)


def test_agent_bus_processes_events(metrics):
    processed: list[AgentEvent] = []

    with AgentBus() as bus:
        _subscribe(bus, processed.append)

        for i in range(3):
            bus.publish(
                AgentEvent(
                    event_type=EventType.MESSAGE,
                    source_agent="test",
                    data={"payload": i},
                )
            )

        bus._event_queue.join()

    assert len(processed) == 3
    assert bus._metrics["events_published"] == 3
    assert bus._metrics["events_processed"] == 3
