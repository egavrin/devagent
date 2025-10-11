"""
Integration test demonstrating Work Planning Agent functionality

This test shows a complete workflow of using the Work Planning Agent.
"""

import pytest
import tempfile
from pathlib import Path
from ai_dev_agent.agents.work_planner import (
    Task,
    WorkPlan,
    WorkPlanningAgent,
    WorkPlanStorage,
    TaskStatus,
    Priority,
)


def test_complete_work_planning_workflow():
    """
    End-to-end test demonstrating the Work Planning Agent in action.

    This test simulates a realistic scenario: planning and executing
    the implementation of a new feature.
    """

    # Setup: Create temporary storage
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = WorkPlanStorage(storage_dir=Path(tmpdir))
        agent = WorkPlanningAgent(storage=storage)

        # ===== PHASE 1: Create a Work Plan =====
        print("\n=== Phase 1: Creating Work Plan ===")

        plan = agent.create_plan(
            goal="Implement user authentication system",
            context={
                "description": "Add JWT-based authentication with login, logout, and token refresh"
            }
        )

        # The agent creates a basic plan with placeholder task
        assert plan is not None
        assert plan.goal == "Implement user authentication system"
        assert len(plan.tasks) >= 1
        print(f"✓ Created plan: {plan.id}")
        print(f"✓ Initial tasks: {len(plan.tasks)}")

        # ===== PHASE 2: Add Detailed Tasks =====
        print("\n=== Phase 2: Adding Detailed Tasks ===")

        # Clear placeholder and add real tasks with dependencies
        plan.tasks = []

        # Task 1: Design database schema (no dependencies)
        task1 = Task(
            title="Design authentication database schema",
            description="Create tables for users, tokens, and sessions",
            priority=Priority.HIGH,
            effort_estimate="2h",
            acceptance_criteria=[
                "User table with email, password_hash, created_at",
                "Token table with token, user_id, expires_at",
                "Session table for tracking active sessions"
            ],
            tags=["database", "design"]
        )
        plan.tasks.append(task1)
        print(f"✓ Added task 1: {task1.title}")

        # Task 2: Implement user model (depends on schema)
        task2 = Task(
            title="Implement User model",
            description="Create User class with password hashing and validation",
            priority=Priority.HIGH,
            effort_estimate="1h",
            dependencies=[task1.id],
            acceptance_criteria=[
                "User class with email, password properties",
                "Password hashing using bcrypt",
                "Email validation"
            ],
            tags=["backend", "model"]
        )
        plan.tasks.append(task2)
        print(f"✓ Added task 2: {task2.title} (depends on task 1)")

        # Task 3: Implement JWT service (depends on User model)
        task3 = Task(
            title="Implement JWT token service",
            description="Create service for generating and validating JWT tokens",
            priority=Priority.CRITICAL,
            effort_estimate="3h",
            dependencies=[task2.id],
            acceptance_criteria=[
                "Generate JWT tokens with user claims",
                "Validate and decode tokens",
                "Handle token expiration"
            ],
            tags=["backend", "security"]
        )
        plan.tasks.append(task3)
        print(f"✓ Added task 3: {task3.title} (depends on task 2)")

        # Task 4: Create login endpoint (depends on JWT service)
        task4 = Task(
            title="Create login API endpoint",
            description="POST /api/auth/login endpoint",
            priority=Priority.HIGH,
            effort_estimate="2h",
            dependencies=[task3.id],
            acceptance_criteria=[
                "Accept email and password",
                "Return JWT token on success",
                "Return 401 on invalid credentials"
            ],
            tags=["backend", "api"]
        )
        plan.tasks.append(task4)
        print(f"✓ Added task 4: {task4.title} (depends on task 3)")

        # Task 5: Write integration tests (depends on login endpoint)
        task5 = Task(
            title="Write authentication integration tests",
            description="Test complete auth flow end-to-end",
            priority=Priority.MEDIUM,
            effort_estimate="2h",
            dependencies=[task4.id],
            acceptance_criteria=[
                "Test successful login",
                "Test invalid credentials",
                "Test token validation",
                "Achieve 90% coverage"
            ],
            tags=["testing"]
        )
        plan.tasks.append(task5)
        print(f"✓ Added task 5: {task5.title} (depends on task 4)")

        # Save the enhanced plan
        agent.storage.save_plan(plan)
        print(f"\n✓ Saved plan with {len(plan.tasks)} tasks")

        # ===== PHASE 3: Execute Tasks in Correct Order =====
        print("\n=== Phase 3: Executing Tasks ===")

        completed_tasks = []
        iteration = 0
        max_iterations = 10  # Safety limit

        while iteration < max_iterations:
            iteration += 1

            # Get next task
            next_task = agent.get_next_task(plan.id)
            if next_task is None:
                print("\n✓ No more tasks available - plan complete!")
                break

            print(f"\nIteration {iteration}: Next task is '{next_task.title}'")
            print(f"  Priority: {next_task.priority.value}")
            print(f"  Effort: {next_task.effort_estimate}")
            print(f"  Dependencies met: ✓")

            # Start the task
            agent.mark_task_started(plan.id, next_task.id)
            print(f"  → Started task")

            # Simulate work being done...
            # (In real usage, actual implementation would happen here)

            # Complete the task
            agent.mark_task_complete(plan.id, next_task.id)
            print(f"  → Completed task")

            completed_tasks.append(next_task.title)

            # Check progress
            updated_plan = agent.storage.load_plan(plan.id)
            progress = updated_plan.get_completion_percentage()
            print(f"  → Overall progress: {progress:.1f}%")

        # Verify all tasks completed
        assert len(completed_tasks) == 5
        print(f"\n✓ Completed all {len(completed_tasks)} tasks")

        # Verify completion order respects dependencies
        assert completed_tasks[0] == "Design authentication database schema"
        assert completed_tasks[1] == "Implement User model"
        assert completed_tasks[2] == "Implement JWT token service"
        assert completed_tasks[3] == "Create login API endpoint"
        assert completed_tasks[4] == "Write authentication integration tests"
        print("✓ Tasks completed in correct dependency order")

        # ===== PHASE 4: Generate Summary =====
        print("\n=== Phase 4: Generating Summary ===")

        summary = agent.get_plan_summary(plan.id)
        print("\n" + "="*60)
        print(summary)
        print("="*60)

        # Verify summary contains expected elements
        assert "Implement user authentication system" in summary
        assert "✅" in summary  # All tasks should be completed
        assert "100.0%" in summary  # Should show 100% complete
        print("\n✓ Generated markdown summary successfully")

        # ===== PHASE 5: Verify Persistence =====
        print("\n=== Phase 5: Testing Persistence ===")

        # Create new agent instance (simulating new session)
        agent2 = WorkPlanningAgent(storage=storage)

        # Load the plan
        loaded_plan = agent2.storage.load_plan(plan.id)
        assert loaded_plan is not None
        assert loaded_plan.id == plan.id
        assert len(loaded_plan.tasks) == 5

        # Verify all tasks are marked complete
        for task in loaded_plan.tasks:
            assert task.status == TaskStatus.COMPLETED
            assert task.completed_at is not None

        print("✓ Plan persisted correctly across sessions")
        print(f"✓ All {len(loaded_plan.tasks)} tasks remain completed")

        # ===== PHASE 6: Verify Dependency Logic =====
        print("\n=== Phase 6: Verifying Dependency Logic ===")

        # Create a new plan to test dependency blocking
        test_plan = agent.create_plan(goal="Test dependency blocking")
        test_plan.tasks = []

        dep_task1 = Task(title="Task A", priority=Priority.HIGH)
        dep_task2 = Task(
            title="Task B (blocked)",
            priority=Priority.CRITICAL,  # Higher priority but blocked
            dependencies=[dep_task1.id]
        )
        test_plan.tasks = [dep_task1, dep_task2]
        agent.storage.save_plan(test_plan)

        # Next task should be Task A, not Task B (even though B has higher priority)
        next_task = agent.get_next_task(test_plan.id)
        assert next_task.title == "Task A"
        print("✓ Dependency blocking works (high-priority task blocked correctly)")

        # Complete Task A
        agent.mark_task_complete(test_plan.id, dep_task1.id)

        # Now Task B should be available
        next_task = agent.get_next_task(test_plan.id)
        assert next_task.title == "Task B (blocked)"
        print("✓ Task becomes available after dependency completed")

        # ===== SUCCESS =====
        print("\n" + "="*60)
        print("✅ ALL WORK PLANNING FEATURES VERIFIED SUCCESSFULLY!")
        print("="*60)
        print("\nDemonstrated capabilities:")
        print("  ✓ Create work plans with goals and context")
        print("  ✓ Add tasks with priorities, estimates, and acceptance criteria")
        print("  ✓ Handle task dependencies correctly")
        print("  ✓ Get next task respecting dependencies and priorities")
        print("  ✓ Track task lifecycle (pending → in progress → completed)")
        print("  ✓ Calculate completion percentage")
        print("  ✓ Generate markdown summaries")
        print("  ✓ Persist plans across sessions")
        print("  ✓ Verify dependency blocking logic")


if __name__ == "__main__":
    # Allow running directly for demonstration
    test_complete_work_planning_workflow()
