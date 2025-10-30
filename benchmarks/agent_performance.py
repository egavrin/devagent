#!/usr/bin/env python3
"""Simple performance benchmarks for multi-agent system.

Run with: python benchmarks/agent_performance.py
"""
import time

from ai_dev_agent.agents.base import AgentContext
from ai_dev_agent.agents.communication.bus import AgentBus, AgentEvent, EventType
from ai_dev_agent.agents.enhanced_registry import EnhancedAgentRegistry
from ai_dev_agent.agents.specialized import (
    DesignAgent,
    ImplementationAgent,
    OrchestratorAgent,
    ReviewAgent,
    TestingAgent,
)


def benchmark_single_agent():
    """Benchmark single agent execution."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Single Agent Execution")
    print("=" * 60)

    agent = DesignAgent()
    context = AgentContext(session_id="bench-001")

    # Warm-up
    agent.execute("Design simple API", context)

    # Measure
    iterations = 20
    start = time.time()
    for i in range(iterations):
        result = agent.execute(f"Design feature {i}", context)
        assert result.success

    duration = time.time() - start
    print(f"Iterations: {iterations}")
    print(f"Total time: {duration:.4f}s")
    print(f"Mean time: {duration/iterations:.4f}s")
    print(f"Throughput: {iterations/duration:.1f} ops/sec")


def benchmark_orchestrator_delegation():
    """Benchmark orchestrator delegation overhead."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Orchestrator Delegation")
    print("=" * 60)

    # Setup
    registry = EnhancedAgentRegistry()
    registry.register_agent(DesignAgent())
    registry.register_agent(TestingAgent())
    registry.register_agent(ImplementationAgent())
    registry.register_agent(ReviewAgent())

    orchestrator = OrchestratorAgent()
    for agent_name in ["design_agent", "test_agent", "implementation_agent", "review_agent"]:
        agent = registry.get_agent(agent_name)
        orchestrator.register_subagent(agent_name, agent)

    context = AgentContext(session_id="bench-002")

    # Warm-up
    orchestrator.delegate_task("design_agent", "Design API", context)

    # Measure
    iterations = 20
    start = time.time()
    for i in range(iterations):
        result = orchestrator.delegate_task("design_agent", f"Design feature {i}", context)
        assert result.success

    duration = time.time() - start
    print(f"Iterations: {iterations}")
    print(f"Total time: {duration:.4f}s")
    print(f"Mean time: {duration/iterations:.4f}s")
    print(f"Throughput: {iterations/duration:.1f} ops/sec")


def benchmark_event_bus():
    """Benchmark communication bus throughput."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Event Bus Throughput")
    print("=" * 60)

    bus = AgentBus()
    event_count = 0

    def handler(event: AgentEvent):
        nonlocal event_count
        event_count += 1

    bus.subscribe(EventType.TASK_STARTED, handler)

    # Warm-up
    for i in range(100):
        event = AgentEvent(
            event_type=EventType.TASK_STARTED, source_agent="test", data={"task_id": i}
        )
        bus.publish(event)

    time.sleep(0.1)  # Let events process
    event_count = 0  # Reset

    # Measure
    iterations = 5000
    start = time.time()

    for i in range(iterations):
        event = AgentEvent(
            event_type=EventType.TASK_STARTED, source_agent="test", data={"task_id": i}
        )
        bus.publish(event)

    duration = time.time() - start

    # Wait for processing
    time.sleep(0.2)
    bus.stop()

    throughput = iterations / duration
    print(f"Events published: {iterations}")
    print(f"Events processed: {event_count}")
    print(f"Total time: {duration:.4f}s")
    print(f"Throughput: {throughput:.0f} events/sec")


def benchmark_registry_lookups():
    """Benchmark agent registry lookups."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Registry Lookups")
    print("=" * 60)

    registry = EnhancedAgentRegistry()
    registry.register_agent(DesignAgent())
    registry.register_agent(TestingAgent())
    registry.register_agent(ImplementationAgent())
    registry.register_agent(ReviewAgent())

    # By name
    iterations = 10000
    start = time.time()
    for _i in range(iterations):
        agent = registry.get_agent("design_agent")
        assert agent is not None
    duration = time.time() - start

    print(f"By-name lookups: {iterations}")
    print(f"Total time: {duration:.4f}s")
    print(f"Mean time: {duration/iterations*1000:.4f}ms")

    # By capability
    start = time.time()
    for _i in range(iterations):
        agents = registry.find_agents_by_capability("technical_design")
        assert len(agents) > 0
    duration = time.time() - start

    print(f"\nBy-capability lookups: {iterations}")
    print(f"Total time: {duration:.4f}s")
    print(f"Mean time: {duration/iterations*1000:.4f}ms")


def main():
    """Run all benchmarks."""
    print("\n" + "#" * 60)
    print("# MULTI-AGENT SYSTEM PERFORMANCE BENCHMARKS")
    print("#" * 60)

    try:
        benchmark_single_agent()
        benchmark_orchestrator_delegation()
        benchmark_event_bus()
        benchmark_registry_lookups()

        print("\n" + "=" * 60)
        print("ALL BENCHMARKS COMPLETED SUCCESSFULLY")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
