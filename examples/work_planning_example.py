#!/usr/bin/env python3
"""
Work Planning Agent - Usage Example

This script demonstrates how to use the Work Planning Agent to:
- Create structured work plans
- Add tasks with dependencies and priorities
- Track progress through task execution
- Generate markdown summaries
- Persist plans across sessions

Usage:
    python3 examples/work_planning_example.py
"""

from ai_dev_agent.agents.work_planner import (
    Task,
    WorkPlan,
    WorkPlanningAgent,
    WorkPlanStorage,
    TaskStatus,
    Priority,
)


def main():
    print("\n" + "="*70)
    print(" Work Planning Agent - Interactive Example")
    print("="*70 + "\n")

    # Initialize the agent
    agent = WorkPlanningAgent()
    print("✓ Work Planning Agent initialized\n")

    # Create a work plan
    print("Creating a work plan for implementing a REST API...")
    plan = agent.create_plan(
        goal="Implement RESTful API for blog platform",
        context={
            "description": "Create CRUD endpoints for posts, comments, and users with authentication"
        }
    )
    print(f"✓ Created plan: {plan.id}\n")

    # Clear the placeholder task and add real tasks
    plan.tasks = []

    # Define tasks with dependencies
    print("Adding tasks with dependencies and priorities...\n")

    # Task 1: Set up project structure
    task1 = Task(
        title="Set up project structure",
        description="Initialize FastAPI project with proper folder structure",
        priority=Priority.HIGH,
        effort_estimate="30m",
        acceptance_criteria=[
            "FastAPI installed and configured",
            "Folder structure created (routes/, models/, schemas/)",
            "Basic config file for environment variables"
        ],
        tags=["setup", "infrastructure"]
    )
    plan.tasks.append(task1)
    print(f"  1. {task1.title} (Priority: {task1.priority.value}, Effort: {task1.effort_estimate})")

    # Task 2: Define data models (depends on task1)
    task2 = Task(
        title="Define database models",
        description="Create SQLAlchemy models for Post, Comment, and User",
        priority=Priority.HIGH,
        effort_estimate="1h",
        dependencies=[task1.id],
        acceptance_criteria=[
            "User model with authentication fields",
            "Post model with foreign key to User",
            "Comment model with foreign keys to Post and User",
            "All relationships properly defined"
        ],
        tags=["database", "models"]
    )
    plan.tasks.append(task2)
    print(f"  2. {task2.title} (Priority: {task2.priority.value}, Effort: {task2.effort_estimate})")
    print(f"     Dependencies: Task 1")

    # Task 3: Implement authentication (depends on task2)
    task3 = Task(
        title="Implement JWT authentication",
        description="Create authentication endpoints and JWT token management",
        priority=Priority.CRITICAL,
        effort_estimate="2h",
        dependencies=[task2.id],
        acceptance_criteria=[
            "POST /auth/register endpoint",
            "POST /auth/login endpoint with JWT token",
            "JWT validation middleware",
            "Password hashing with bcrypt"
        ],
        tags=["auth", "security"]
    )
    plan.tasks.append(task3)
    print(f"  3. {task3.title} (Priority: {task3.priority.value}, Effort: {task3.effort_estimate})")
    print(f"     Dependencies: Task 2")

    # Task 4: Create CRUD endpoints (depends on task3)
    task4 = Task(
        title="Implement CRUD endpoints for posts",
        description="Create all REST endpoints for post management",
        priority=Priority.HIGH,
        effort_estimate="1.5h",
        dependencies=[task3.id],
        acceptance_criteria=[
            "GET /posts (list all posts)",
            "GET /posts/{id} (get single post)",
            "POST /posts (create post, requires auth)",
            "PUT /posts/{id} (update post, requires auth)",
            "DELETE /posts/{id} (delete post, requires auth)"
        ],
        tags=["api", "crud"]
    )
    plan.tasks.append(task4)
    print(f"  4. {task4.title} (Priority: {task4.priority.value}, Effort: {task4.effort_estimate})")
    print(f"     Dependencies: Task 3")

    # Task 5: Add tests (depends on task4)
    task5 = Task(
        title="Write integration tests",
        description="Create comprehensive test suite for all endpoints",
        priority=Priority.MEDIUM,
        effort_estimate="2h",
        dependencies=[task4.id],
        acceptance_criteria=[
            "Test all CRUD operations",
            "Test authentication flow",
            "Test error cases",
            "Achieve 90% code coverage"
        ],
        tags=["testing", "quality"]
    )
    plan.tasks.append(task5)
    print(f"  5. {task5.title} (Priority: {task5.priority.value}, Effort: {task5.effort_estimate})")
    print(f"     Dependencies: Task 4\n")

    # Save the plan
    agent.storage.save_plan(plan)
    print(f"✓ Plan saved with {len(plan.tasks)} tasks\n")

    # Display the initial plan summary
    print("="*70)
    print(" Initial Plan Summary")
    print("="*70)
    summary = agent.get_plan_summary(plan.id)
    print(summary)
    print("="*70 + "\n")

    # Simulate executing tasks
    print("Simulating task execution...\n")

    task_count = 0
    max_tasks = len(plan.tasks)

    while task_count < max_tasks:
        # Get next task
        next_task = agent.get_next_task(plan.id)
        if next_task is None:
            break

        task_count += 1
        print(f"[{task_count}/{max_tasks}] Starting: {next_task.title}")
        print(f"      Priority: {next_task.priority.value}")
        print(f"      Estimated effort: {next_task.effort_estimate}")

        # Start the task
        agent.mark_task_started(plan.id, next_task.id)

        # Simulate work (in real usage, actual implementation happens here)
        print(f"      Status: In Progress...")

        # Complete the task
        agent.mark_task_complete(plan.id, next_task.id)
        print(f"      Status: ✓ Completed")

        # Show progress
        updated_plan = agent.storage.load_plan(plan.id)
        progress = updated_plan.get_completion_percentage()
        print(f"      Overall progress: {progress:.0f}%\n")

    # Display the final plan summary
    print("="*70)
    print(" Final Plan Summary (All Tasks Completed)")
    print("="*70)
    final_summary = agent.get_plan_summary(plan.id)
    print(final_summary)
    print("="*70 + "\n")

    # Show where the plan is stored
    plan_file = agent.storage.storage_dir / f"{plan.id}.json"
    print(f"✓ Plan persisted to: {plan_file}")
    print(f"✓ You can reload this plan in future sessions using the plan ID: {plan.id}\n")

    # Example of loading a plan
    print("Demonstrating plan persistence...")
    print(f"Loading plan {plan.id}...\n")

    # Create a new agent instance (simulating a new session)
    agent2 = WorkPlanningAgent()
    loaded_plan = agent2.storage.load_plan(plan.id)

    if loaded_plan:
        print(f"✓ Successfully loaded plan: {loaded_plan.goal}")
        print(f"✓ Tasks: {len(loaded_plan.tasks)}")
        print(f"✓ Completion: {loaded_plan.get_completion_percentage():.0f}%")

        # Show task statuses
        print("\nTask Statuses:")
        for i, task in enumerate(loaded_plan.tasks, 1):
            status_icon = "✅" if task.status == TaskStatus.COMPLETED else "⏳"
            print(f"  {i}. {status_icon} {task.title} ({task.status.value})")
    else:
        print("❌ Failed to load plan")

    print("\n" + "="*70)
    print(" Example Complete!")
    print("="*70)
    print("\nKey Features Demonstrated:")
    print("  ✓ Create work plans with goals and context")
    print("  ✓ Add tasks with priorities, estimates, and acceptance criteria")
    print("  ✓ Define task dependencies")
    print("  ✓ Get next task respecting dependencies and priorities")
    print("  ✓ Track task lifecycle (pending → in progress → completed)")
    print("  ✓ Calculate completion percentage")
    print("  ✓ Generate markdown summaries")
    print("  ✓ Persist plans across sessions")
    print("\nFor more information, see:")
    print("  - docs/design/work_planning_design.md")
    print("  - docs/reference_analysis/work_planning_analysis.md")
    print("  - tests/test_work_planner_integration.py")
    print()


if __name__ == "__main__":
    main()
