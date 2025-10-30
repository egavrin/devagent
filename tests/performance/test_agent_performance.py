"""Performance tests for multi-agent system.

Tests agent coordination efficiency, delegation overhead, and communication bus performance.
"""

import time

import pytest

from ai_dev_agent.agents.base import AgentContext
from ai_dev_agent.agents.communication.bus import AgentBus, AgentEvent, EventType
from ai_dev_agent.agents.enhanced_registry import EnhancedAgentRegistry
from ai_dev_agent.agents.integration.planning_integration import AutomatedWorkflow
from ai_dev_agent.agents.specialized.design_agent import DesignAgent
from ai_dev_agent.agents.specialized.implementation_agent import ImplementationAgent
from ai_dev_agent.agents.specialized.orchestrator_agent import OrchestratorAgent
from ai_dev_agent.agents.specialized.review_agent import ReviewAgent
from ai_dev_agent.agents.specialized.testing_agent import TestingAgent
from ai_dev_agent.agents.work_planner import Priority, Task, WorkPlan


class PerformanceMetrics:
    """Collects and reports performance metrics."""

    def __init__(self):
        self.measurements: dict[str, list[float]] = {}

    def record(self, operation: str, duration: float):
        """Record a performance measurement."""
        if operation not in self.measurements:
            self.measurements[operation] = []
        self.measurements[operation].append(duration)

    def get_stats(self, operation: str) -> dict[str, float]:
        """Get statistics for an operation."""
        if operation not in self.measurements:
            return {}

        durations = self.measurements[operation]
        return {
            "count": len(durations),
            "total": sum(durations),
            "mean": sum(durations) / len(durations),
            "min": min(durations),
            "max": max(durations),
        }

    def report(self) -> str:
        """Generate performance report."""
        lines = ["Performance Metrics Report", "=" * 50]

        for operation, durations in self.measurements.items():
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
    """Provide performance metrics collector."""
    return PerformanceMetrics()


@pytest.fixture
def registry():
    """Provide agent registry with all agents."""
    reg = EnhancedAgentRegistry()
    reg.register_agent(DesignAgent())
    reg.register_agent(TestingAgent())
    reg.register_agent(ImplementationAgent())
    reg.register_agent(ReviewAgent())
    reg.register_agent(OrchestratorAgent())
    return reg


class TestAgentDelegationPerformance:
    """Test performance of agent delegation and coordination."""

    @pytest.mark.llm
    def test_single_agent_execution_overhead(self, metrics):
        """Measure overhead of single agent execution."""
        agent = DesignAgent()
        context = AgentContext(session_id="perf-001")

        # Warm-up
        agent.execute("Design simple API", context)

        # Measure
        iterations = 10
        for i in range(iterations):
            start = time.time()
            result = agent.execute(f"Design feature {i}", context)
            duration = time.time() - start
            metrics.record("single_agent_execution", duration)
            assert result.success

        stats = metrics.get_stats("single_agent_execution")
        assert stats["mean"] < 0.1, "Single agent execution too slow"
        print(f"\nSingle agent mean: {stats['mean']:.4f}s")

    @pytest.mark.llm
    def test_orchestrator_delegation_overhead(self, metrics, registry):
        """Measure overhead of orchestrator delegating to agents."""
        orchestrator = OrchestratorAgent()

        # Register subagents
        for agent_name in ["design_agent", "test_agent", "implementation_agent", "review_agent"]:
            agent = registry.get_agent(agent_name)
            orchestrator.register_subagent(agent_name, agent)

        context = AgentContext(session_id="perf-002")

        # Warm-up
        orchestrator.delegate_task("design", "Design API", context)

        # Measure delegation overhead
        iterations = 10
        for i in range(iterations):
            start = time.time()
            result = orchestrator.delegate_task("design_agent", f"Design feature {i}", context)
            duration = time.time() - start
            metrics.record("orchestrator_delegation", duration)
            assert result.success

        stats = metrics.get_stats("orchestrator_delegation")

        # Compare with direct execution
        direct_stats = metrics.get_stats("single_agent_execution")
        if direct_stats:
            overhead = stats["mean"] - direct_stats["mean"]
            overhead_percent = (overhead / direct_stats["mean"]) * 100
            print(f"\nDelegation overhead: {overhead:.4f}s ({overhead_percent:.1f}%)")
            assert overhead_percent < 50, "Delegation overhead too high"

    @pytest.mark.llm
    def test_parallel_task_execution(self, metrics, registry):
        """Measure performance of parallel task execution."""
        orchestrator = OrchestratorAgent()

        for agent_name in ["design_agent", "test_agent", "implementation_agent", "review_agent"]:
            agent = registry.get_agent(agent_name)
            orchestrator.register_subagent(agent_name, agent)

        context = AgentContext(session_id="perf-003")

        # Define parallel tasks
        tasks = [
            {"agent": "design_agent", "task": "Design feature A"},
            {"agent": "test_agent", "task": "Generate tests for feature B"},
            {"agent": "implementation_agent", "task": "Implement feature C"},
            {"agent": "review_agent", "task": "Review feature D"},
        ]

        # Warm-up
        orchestrator.execute_parallel(tasks, context)

        # Measure parallel execution
        iterations = 5
        for i in range(iterations):
            start = time.time()
            results = orchestrator.execute_parallel(tasks, context)
            duration = time.time() - start
            metrics.record("parallel_execution", duration)
            assert all(r.success for r in results)

        # Measure sequential execution for comparison
        for i in range(iterations):
            start = time.time()
            for task in tasks:
                orchestrator.delegate_task(task["agent"], task["task"], context)
            duration = time.time() - start
            metrics.record("sequential_execution", duration)

        parallel_stats = metrics.get_stats("parallel_execution")
        sequential_stats = metrics.get_stats("sequential_execution")

        speedup = sequential_stats["mean"] / parallel_stats["mean"]
        print(f"\nParallel speedup: {speedup:.2f}x")
        print(f"Parallel mean: {parallel_stats['mean']:.4f}s")
        print(f"Sequential mean: {sequential_stats['mean']:.4f}s")

        # Parallel should be faster (at least 1.5x for 4 tasks)
        assert speedup >= 1.5, "Parallel execution not providing sufficient speedup"


class TestCommunicationBusPerformance:
    """Test performance of event-driven communication bus."""

    def test_event_publishing_throughput(self, metrics):
        """Measure event publishing throughput."""
        bus = AgentBus()
        bus.start()  # Start the bus
        event_count = 0

        def handler(event: AgentEvent):
            nonlocal event_count
            event_count += 1

        bus.subscribe(EventType.TASK_STARTED, handler)

        # Warm-up
        for i in range(10):
            event = AgentEvent(
                event_type=EventType.TASK_STARTED, source_agent="test", data={"task_id": i}
            )
            bus.publish(event)

        time.sleep(0.1)  # Let events process

        # Measure throughput
        iterations = 1000
        start = time.time()

        for i in range(iterations):
            event = AgentEvent(
                event_type=EventType.TASK_STARTED, source_agent="test", data={"task_id": i}
            )
            bus.publish(event)

        duration = time.time() - start
        metrics.record("event_publishing", duration)

        # Wait for all events to process
        time.sleep(0.2)
        bus.stop()

        throughput = iterations / duration
        print(f"\nEvent publishing throughput: {throughput:.0f} events/sec")
        print(f"Total duration: {duration:.4f}s")

        assert throughput > 1000, "Event publishing throughput too low"

    def test_event_subscription_scalability(self, metrics):
        """Measure performance with many subscribers."""
        handler_counts = {}

        def make_handler(handler_id: int):
            def handler(event: AgentEvent):
                if handler_id not in handler_counts:
                    handler_counts[handler_id] = 0
                handler_counts[handler_id] += 1

            return handler

        # Subscribe multiple handlers
        subscriber_counts = [1, 10, 50, 100]

        for count in subscriber_counts:
            bus = AgentBus()
            bus.start()  # Start the bus
            handler_counts.clear()

            # Register subscribers
            for i in range(count):
                bus.subscribe(EventType.TASK_COMPLETED, make_handler(i))

            # Publish events
            event_count = 100
            start = time.time()

            for i in range(event_count):
                event = AgentEvent(
                    event_type=EventType.TASK_COMPLETED, source_agent="test", data={"task_id": i}
                )
                bus.publish(event)

            duration = time.time() - start
            time.sleep(0.1)  # Let events process
            bus.stop()

            metrics.record(f"bus_with_{count}_subscribers", duration)

            print(f"\n{count} subscribers: {duration:.4f}s for {event_count} events")

        # Verify scalability - should be roughly linear
        stats_1 = metrics.get_stats("bus_with_1_subscribers")
        stats_100 = metrics.get_stats("bus_with_100_subscribers")

        if stats_1 and stats_100:
            ratio = stats_100["mean"] / stats_1["mean"]
            print(f"\nScalability ratio (100/1 subscribers): {ratio:.2f}x")
            # Should be less than 150x slower (ideally closer to 100x)
            assert ratio < 150, "Bus scalability poor with many subscribers"

    def test_request_response_latency(self, metrics):
        """Measure request-response pattern latency."""
        bus = AgentBus()
        bus.start()  # Start the bus

        def responder(event: AgentEvent):
            # Simulate processing
            response_event = AgentEvent(
                event_type=EventType.TASK_COMPLETED,
                source_agent="responder",
                data={"response": f"processed_{event.data.get('request_id')}"},
            )
            bus.publish(response_event)

        bus.subscribe(EventType.TASK_STARTED, responder)

        # Warm-up
        for i in range(10):
            event = AgentEvent(
                event_type=EventType.TASK_STARTED, source_agent="requester", data={"request_id": i}
            )
            bus.publish(event)

        time.sleep(0.1)

        # Measure latency
        iterations = 100
        for i in range(iterations):
            start = time.time()

            event = AgentEvent(
                event_type=EventType.TASK_STARTED, source_agent="requester", data={"request_id": i}
            )
            bus.publish(event)

            # Small delay to simulate waiting for response
            time.sleep(0.001)

            duration = time.time() - start
            metrics.record("request_response_latency", duration)

        bus.stop()

        stats = metrics.get_stats("request_response_latency")
        print(f"\nRequest-response mean latency: {stats['mean']*1000:.2f}ms")
        assert stats["mean"] < 0.01, "Request-response latency too high"


class TestWorkflowPerformance:
    """Test performance of automated workflow execution."""

    def test_small_plan_execution(self, metrics, registry):
        """Measure execution time for small work plan."""
        orchestrator = OrchestratorAgent()

        for agent_name in ["design_agent", "test_agent", "implementation_agent", "review_agent"]:
            agent = registry.get_agent(agent_name)
            orchestrator.register_subagent(agent_name, agent)

        # Create small plan (4 tasks)
        plan = WorkPlan(
            id="perf-plan-001",
            goal="Implement small feature",
            tasks=[
                Task(
                    id="task-1",
                    title="Design feature",
                    description="Create design",
                    priority=Priority.HIGH,
                    tags=["design"],
                ),
                Task(
                    id="task-2",
                    title="Write tests",
                    description="Generate tests",
                    priority=Priority.HIGH,
                    dependencies=["task-1"],
                    tags=["test"],
                ),
                Task(
                    id="task-3",
                    title="Implement code",
                    description="Write code",
                    priority=Priority.HIGH,
                    dependencies=["task-2"],
                    tags=["implement"],
                ),
                Task(
                    id="task-4",
                    title="Review code",
                    description="Code review",
                    priority=Priority.MEDIUM,
                    dependencies=["task-3"],
                    tags=["review"],
                ),
            ],
        )

        workflow = AutomatedWorkflow()
        context = AgentContext(session_id="perf-workflow-001")

        # Warm-up
        workflow.execute_plan_automatically(plan, context, stop_on_failure=False)

        # Measure
        iterations = 5
        for i in range(iterations):
            # Reset plan
            for task in plan.tasks:
                task.status = task.status.__class__.PENDING

            start = time.time()
            workflow.execute_plan_automatically(plan, context, stop_on_failure=False)
            duration = time.time() - start
            metrics.record("small_plan_execution", duration)

        stats = metrics.get_stats("small_plan_execution")
        print(f"\nSmall plan (4 tasks) mean execution: {stats['mean']:.4f}s")
        print(f"Per-task overhead: {stats['mean']/4:.4f}s")

        assert stats["mean"] < 2.0, "Small plan execution too slow"

    def test_medium_plan_execution(self, metrics, registry):
        """Measure execution time for medium work plan."""
        orchestrator = OrchestratorAgent()

        for agent_name in ["design_agent", "test_agent", "implementation_agent", "review_agent"]:
            agent = registry.get_agent(agent_name)
            orchestrator.register_subagent(agent_name, agent)

        # Create medium plan (12 tasks - 3 features)
        tasks = []
        for feature_idx in range(3):
            base_id = feature_idx * 4
            tasks.extend(
                [
                    Task(
                        id=f"task-{base_id+1}",
                        title=f"Design feature {feature_idx+1}",
                        description="Create design",
                        priority=Priority.HIGH,
                        tags=["design"],
                    ),
                    Task(
                        id=f"task-{base_id+2}",
                        title=f"Write tests for feature {feature_idx+1}",
                        description="Generate tests",
                        priority=Priority.HIGH,
                        dependencies=[f"task-{base_id+1}"],
                        tags=["test"],
                    ),
                    Task(
                        id=f"task-{base_id+3}",
                        title=f"Implement feature {feature_idx+1}",
                        description="Write code",
                        priority=Priority.HIGH,
                        dependencies=[f"task-{base_id+2}"],
                        tags=["implement"],
                    ),
                    Task(
                        id=f"task-{base_id+4}",
                        title=f"Review feature {feature_idx+1}",
                        description="Code review",
                        priority=Priority.MEDIUM,
                        dependencies=[f"task-{base_id+3}"],
                        tags=["review"],
                    ),
                ]
            )

        plan = WorkPlan(id="perf-plan-002", goal="Implement 3 features", tasks=tasks)

        workflow = AutomatedWorkflow()
        context = AgentContext(session_id="perf-workflow-002")

        # Measure (fewer iterations for medium plan)
        iterations = 3
        for i in range(iterations):
            # Reset plan
            for task in plan.tasks:
                task.status = task.status.__class__.PENDING

            start = time.time()
            workflow.execute_plan_automatically(plan, context, stop_on_failure=False)
            duration = time.time() - start
            metrics.record("medium_plan_execution", duration)

        stats = metrics.get_stats("medium_plan_execution")
        print(f"\nMedium plan (12 tasks) mean execution: {stats['mean']:.4f}s")
        print(f"Per-task overhead: {stats['mean']/12:.4f}s")

        # Should scale roughly linearly
        small_stats = metrics.get_stats("small_plan_execution")
        if small_stats:
            ratio = stats["mean"] / small_stats["mean"]
            print(f"Scaling ratio (12/4 tasks): {ratio:.2f}x (expected ~3x)")


class TestMemoryAndScalability:
    """Test memory usage and scalability."""

    @pytest.mark.llm
    def test_agent_session_memory(self, metrics):
        """Measure memory impact of agent sessions."""

        agent = DesignAgent()
        context = AgentContext(session_id="perf-mem-001")

        # Create many sessions
        session_count = 100
        sessions = []

        start = time.time()
        for i in range(session_count):
            session = agent.create_session(AgentContext(session_id=f"session-{i}"))
            # Add some history
            from ai_dev_agent.agents.base import Message

            session.add_message(Message(role="user", content=f"Request {i}"))
            session.add_result(agent.execute(f"Design feature {i}", context))
            sessions.append(session)

        duration = time.time() - start
        metrics.record("session_creation", duration)

        print(f"\nCreated {session_count} sessions in {duration:.4f}s")
        print(f"Per-session time: {duration/session_count*1000:.2f}ms")

        assert duration < 5.0, "Session creation too slow"

    def test_registry_lookup_performance(self, metrics, registry):
        """Measure agent registry lookup performance."""
        iterations = 1000

        # Test by name lookup
        start = time.time()
        for i in range(iterations):
            agent = registry.get_agent("design_agent")
            assert agent is not None
        duration = time.time() - start
        metrics.record("registry_lookup_by_name", duration)

        print(f"\nRegistry lookup by name: {duration/iterations*1000:.4f}ms per lookup")

        # Test by capability lookup
        start = time.time()
        for i in range(iterations):
            agents = registry.find_agents_by_capability("technical_design")
            assert len(agents) > 0
        duration = time.time() - start
        metrics.record("registry_lookup_by_capability", duration)

        print(f"Registry lookup by capability: {duration/iterations*1000:.4f}ms per lookup")

        assert duration < 1.0, "Registry lookups too slow"


@pytest.fixture(scope="session", autouse=True)
def print_performance_summary(request):
    """Print performance summary at end of session."""
    metrics = PerformanceMetrics()

    def finalizer():
        if metrics.measurements:
            print("\n" + "=" * 60)
            print("PERFORMANCE TEST SUMMARY")
            print("=" * 60)
            print(metrics.report())
            print("=" * 60)

    request.addfinalizer(finalizer)
    return metrics
