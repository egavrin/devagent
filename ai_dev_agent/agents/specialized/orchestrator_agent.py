"""Orchestrator Agent for coordinating multiple agents."""
from __future__ import annotations

import re
from typing import Dict, List, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..base import BaseAgent, AgentContext, AgentResult, AgentCapability, AgentStatus


class OrchestratorAgent(BaseAgent):
    """Agent specialized in coordinating multiple specialized agents."""

    def __init__(self):
        """Initialize Orchestrator Agent."""
        super().__init__(
            name="orchestrator_agent",
            description="Coordinates multiple agents to complete complex workflows",
            capabilities=[
                "coordination",
                "task_delegation",
                "workflow_management",
                "parallel_execution"
            ],
            tools=["read", "write", "grep", "find", "run"],
            max_iterations=50  # Higher for complex coordination
        )

        # Registered subagents
        self._subagents: Dict[str, BaseAgent] = {}

        # Workflow tracking
        self._workflows: Dict[str, Dict[str, Any]] = {}

        # Register capabilities
        self._register_capabilities()

    def _register_capabilities(self):
        """Register agent capabilities."""
        capabilities = [
            AgentCapability(
                name="coordination",
                description="Coordinate multiple agents",
                required_tools=[],
                optional_tools=["read", "write"]
            ),
            AgentCapability(
                name="task_delegation",
                description="Delegate tasks to appropriate agents",
                required_tools=[],
                optional_tools=[]
            ),
            AgentCapability(
                name="workflow_management",
                description="Manage complex workflows",
                required_tools=[],
                optional_tools=["write"]
            ),
            AgentCapability(
                name="parallel_execution",
                description="Execute tasks in parallel",
                required_tools=[],
                optional_tools=[]
            )
        ]

        for capability in capabilities:
            self.register_capability(capability)

    def register_subagent(self, name: str, agent: BaseAgent) -> None:
        """
        Register a subagent.

        Args:
            name: Agent identifier
            agent: Agent instance
        """
        self._subagents[name] = agent

    def has_subagent(self, name: str) -> bool:
        """Check if subagent is registered."""
        return name in self._subagents

    def get_subagent(self, name: str) -> Optional[BaseAgent]:
        """Get a registered subagent."""
        return self._subagents.get(name)

    def delegate_task(
        self,
        agent_name: str,
        task: str,
        context: AgentContext
    ) -> AgentResult:
        """
        Delegate a task to a subagent.

        Args:
            agent_name: Name of agent to delegate to
            task: Task description
            context: Execution context

        Returns:
            Result from subagent
        """
        if agent_name not in self._subagents:
            return AgentResult(
                success=False,
                output="",
                error=f"Unknown agent: {agent_name}"
            )

        agent = self._subagents[agent_name]

        # Create child context
        child_context = agent.create_child_context(context)

        # Execute task
        return agent.execute(task, child_context)

    def delegate_with_retry(
        self,
        agent_name: str,
        task: str,
        context: AgentContext,
        max_retries: int = 3
    ) -> AgentResult:
        """
        Delegate task with retry on failure.

        Args:
            agent_name: Name of agent
            task: Task description
            context: Execution context
            max_retries: Maximum retry attempts

        Returns:
            Result from subagent
        """
        last_result = None

        for attempt in range(max_retries):
            result = self.delegate_task(agent_name, task, context)

            if result.success:
                return result

            last_result = result

        return last_result or AgentResult(
            success=False,
            output="",
            error="Max retries exceeded"
        )

    def coordinate_workflow(
        self,
        workflow: Dict[str, Any],
        context: AgentContext
    ) -> Dict[str, Any]:
        """
        Coordinate a multi-step workflow.

        Args:
            workflow: Workflow specification
            context: Execution context

        Returns:
            Workflow execution results
        """
        steps = workflow.get("steps", [])
        results = []
        steps_completed = 0

        for step in steps:
            agent_name = step.get("agent")
            task = step.get("task")

            result = self.delegate_task(agent_name, task, context)
            results.append(result)

            if result.success:
                steps_completed += 1
            else:
                # Stop on failure unless workflow specifies continue
                if not workflow.get("continue_on_failure", False):
                    break

        return {
            "success": steps_completed == len(steps),
            "steps_completed": steps_completed,
            "total_steps": len(steps),
            "results": results
        }

    def execute_parallel(
        self,
        tasks: List[Dict[str, Any]],
        context: AgentContext,
        max_workers: int = 4
    ) -> List[AgentResult]:
        """
        Execute multiple tasks in parallel.

        Args:
            tasks: List of task specifications
            context: Execution context
            max_workers: Maximum parallel workers

        Returns:
            List of results from all tasks
        """
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {}
            for task in tasks:
                agent_name = task.get("agent")
                task_desc = task.get("task")

                future = executor.submit(
                    self.delegate_task,
                    agent_name,
                    task_desc,
                    context
                )
                futures[future] = task

            # Collect results
            for future in as_completed(futures):
                result = future.result()
                results.append(result)

        return results

    def execute_sequential(
        self,
        tasks: List[Dict[str, Any]],
        context: AgentContext
    ) -> List[AgentResult]:
        """
        Execute tasks sequentially respecting dependencies.

        Args:
            tasks: List of task specifications with dependencies
            context: Execution context

        Returns:
            List of results in execution order
        """
        results = []
        completed_tasks = set()

        # Keep executing until all tasks are done
        while len(completed_tasks) < len(tasks):
            progress_made = False

            for i, task in enumerate(tasks):
                # Skip if already completed
                if i in completed_tasks:
                    continue

                # Check dependencies
                depends_on = task.get("depends_on", [])
                if all(dep in completed_tasks or self._is_task_by_agent(dep, completed_tasks, tasks)
                       for dep in depends_on):

                    # Execute task
                    agent_name = task.get("agent")
                    task_desc = task.get("task")

                    result = self.delegate_task(agent_name, task_desc, context)
                    results.append(result)

                    completed_tasks.add(i)
                    progress_made = True

                    # Stop on failure
                    if not result.success:
                        return results

            # Prevent infinite loop if dependencies can't be satisfied
            if not progress_made:
                break

        return results

    def _is_task_by_agent(self, agent_or_index: Any, completed: set, tasks: List) -> bool:
        """Check if a dependency is satisfied."""
        # Handle both agent names and task indices
        if isinstance(agent_or_index, int):
            return agent_or_index in completed
        elif isinstance(agent_or_index, str):
            # Check if any completed task matches this agent
            for idx in completed:
                if tasks[idx].get("agent") == agent_or_index:
                    return True
        return False

    def select_agent_for_task(
        self,
        task_type: str,
        required_capabilities: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Select the best agent for a task.

        Args:
            task_type: Type of task
            required_capabilities: Required capabilities

        Returns:
            Name of selected agent
        """
        required_capabilities = required_capabilities or []

        # Find agents with required capabilities
        candidates = []

        for name, agent in self._subagents.items():
            if all(cap in agent.capabilities for cap in required_capabilities):
                candidates.append(name)

        # Return first match (could be enhanced with scoring)
        return candidates[0] if candidates else None

    def aggregate_results(
        self,
        results: List[AgentResult]
    ) -> Dict[str, Any]:
        """
        Aggregate results from multiple agents.

        Args:
            results: List of agent results

        Returns:
            Aggregated statistics
        """
        total = len(results)
        successful = sum(1 for r in results if r.success)
        failed = total - successful

        return {
            "total_tasks": total,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total if total > 0 else 0.0,
            "results": results
        }

    def create_workflow_from_plan(
        self,
        plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a workflow from a work plan.

        Args:
            plan: Work plan specification

        Returns:
            Workflow specification
        """
        steps = []

        for task in plan.get("tasks", []):
            task_type = task.get("type", "")
            task_id = task.get("id", "")
            description = task.get("description", "")
            depends_on = task.get("depends_on", [])

            # Map task type to agent
            agent_name = self._map_task_type_to_agent(task_type)

            steps.append({
                "id": task_id,
                "agent": agent_name,
                "task": description,
                "depends_on": depends_on
            })

        return {
            "goal": plan.get("goal", ""),
            "steps": steps
        }

    def _map_task_type_to_agent(self, task_type: str) -> str:
        """Map task type to agent name."""
        mapping = {
            "design": "design",
            "test": "test",
            "implement": "implement",
            "review": "review",
            "code_review": "review",
            "implementation": "implement"
        }
        return mapping.get(task_type.lower(), "default")

    def start_workflow(self, workflow_id: str, total_steps: int) -> None:
        """
        Start tracking a workflow.

        Args:
            workflow_id: Workflow identifier
            total_steps: Total number of steps
        """
        self._workflows[workflow_id] = {
            "workflow_id": workflow_id,
            "total_steps": total_steps,
            "completed": 0,
            "started_at": datetime.now().isoformat(),
            "status": "running"
        }

    def update_progress(self, workflow_id: str, completed: int) -> None:
        """
        Update workflow progress.

        Args:
            workflow_id: Workflow identifier
            completed: Number of completed steps
        """
        if workflow_id in self._workflows:
            self._workflows[workflow_id]["completed"] = completed
            self._workflows[workflow_id]["updated_at"] = datetime.now().isoformat()

    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """
        Get workflow status.

        Args:
            workflow_id: Workflow identifier

        Returns:
            Workflow status
        """
        if workflow_id not in self._workflows:
            return {"error": "Workflow not found"}

        status = self._workflows[workflow_id].copy()
        total = status["total_steps"]
        completed = status["completed"]

        status["progress_percentage"] = (completed / total * 100) if total > 0 else 0.0

        return status

    def execute(self, prompt: str, context: AgentContext) -> AgentResult:
        """
        Execute orchestration task.

        Args:
            prompt: Orchestration task description
            context: Execution context

        Returns:
            AgentResult with orchestration outcome
        """
        try:
            # Parse prompt to identify workflow steps
            steps = []

            # Look for numbered steps
            step_pattern = r'\d+\.\s*(.+?)(?=\n\d+\.|\Z)'
            matches = re.findall(step_pattern, prompt, re.DOTALL)

            for match in matches:
                step_text = match.strip()

                # Determine agent based on keywords
                agent_name = "default"
                if any(word in step_text.lower() for word in ["design", "architect"]):
                    agent_name = "design"
                elif any(word in step_text.lower() for word in ["test", "spec"]):
                    agent_name = "test"
                elif any(word in step_text.lower() for word in ["implement", "code", "write"]):
                    agent_name = "implement"
                elif any(word in step_text.lower() for word in ["review", "check", "analyze"]):
                    agent_name = "review"

                if agent_name in self._subagents:
                    steps.append({
                        "agent": agent_name,
                        "task": step_text
                    })

            # Execute workflow
            if steps:
                workflow_result = self.coordinate_workflow(
                    {"steps": steps},
                    context
                )

                return AgentResult(
                    success=workflow_result["success"],
                    output=f"Workflow completed: {workflow_result['steps_completed']}/{workflow_result['total_steps']} steps",
                    metadata={
                        "workflow_completed": workflow_result["success"],
                        "steps_completed": workflow_result["steps_completed"],
                        "total_steps": workflow_result["total_steps"]
                    }
                )
            else:
                return AgentResult(
                    success=False,
                    output="",
                    error="No valid workflow steps identified"
                )

        except Exception as e:
            return AgentResult(
                success=False,
                output="",
                error=str(e)
            )