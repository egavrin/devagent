"""CLI commands for multi-agent system."""
import click
from pathlib import Path
from typing import Optional

from ..agents.specialized import (
    DesignAgent, TestingAgent, ImplementationAgent, ReviewAgent, OrchestratorAgent
)
from ..agents.integration import PlanningIntegration, AutomatedWorkflow
from ..agents.base import AgentContext, AgentResult
from ..agents.work_planner.storage import WorkPlanStorage


@click.group(name="agent")
def agent_group():
    """Multi-agent system commands."""
    pass


@agent_group.command(name="design")
@click.argument("feature", required=True)
@click.option("--output", "-o", help="Output path for design document")
@click.option("--context", "-c", help="Additional context")
def agent_design(feature: str, output: Optional[str], context: Optional[str]):
    """Run Design Agent to create technical design."""
    try:
        agent = DesignAgent()
        agent_context = AgentContext(session_id=f"design-{feature}")

        prompt = f"Design {feature}"
        if context:
            prompt += f"\nContext: {context}"

        click.echo(f"🎨 Design Agent: Creating design for '{feature}'...")

        result = agent.execute(prompt, agent_context)

        if result.success:
            click.echo(click.style("✓ Design completed successfully", fg="green"))
            click.echo(f"\nOutput: {result.output}")

            if "design_document" in result.metadata:
                click.echo(f"Design document: {result.metadata['design_document']}")
        else:
            click.echo(click.style(f"✗ Design failed: {result.error}", fg="red"))
            raise click.Abort()

    except Exception as e:
        click.echo(click.style(f"Error: {str(e)}", fg="red"))
        raise click.Abort()


@agent_group.command(name="test")
@click.argument("feature", required=True)
@click.option("--coverage", "-c", default=90, help="Target coverage percentage")
@click.option("--type", "-t", type=click.Choice(["unit", "integration", "all"]), default="all")
def agent_test(feature: str, coverage: int, type: str):
    """Run Test Agent to generate tests."""
    try:
        agent = TestingAgent()
        agent_context = AgentContext(session_id=f"test-{feature}")

        prompt = f"Create {type} tests for {feature} with {coverage}% coverage"

        click.echo(f"🧪 Test Agent: Generating tests for '{feature}'...")

        result = agent.execute(prompt, agent_context)

        if result.success:
            click.echo(click.style("✓ Tests generated successfully", fg="green"))
            click.echo(f"\nOutput: {result.output}")

            if "test_files_created" in result.metadata:
                files = result.metadata["test_files_created"]
                click.echo(f"\nTest files created:")
                for f in files:
                    click.echo(f"  - {f}")
        else:
            click.echo(click.style(f"✗ Test generation failed: {result.error}", fg="red"))
            raise click.Abort()

    except Exception as e:
        click.echo(click.style(f"Error: {str(e)}", fg="red"))
        raise click.Abort()


@agent_group.command(name="implement")
@click.argument("design_file", required=True)
@click.option("--test-file", "-t", help="Path to test file")
def agent_implement(design_file: str, test_file: Optional[str]):
    """Run Implementation Agent to implement design."""
    try:
        agent = ImplementationAgent()
        agent_context = AgentContext(session_id=f"implement-{Path(design_file).stem}")

        prompt = f"Implement the design at {design_file}"
        if test_file:
            prompt += f" with tests at {test_file}"

        click.echo(f"⚙️  Implementation Agent: Implementing design from '{design_file}'...")

        result = agent.execute(prompt, agent_context)

        if result.success:
            click.echo(click.style("✓ Implementation completed", fg="green"))
            click.echo(f"\nOutput: {result.output}")

            if "files_created" in result.metadata:
                click.echo(f"\nFiles created: {len(result.metadata['files_created'])}")
        else:
            click.echo(click.style(f"✗ Implementation failed: {result.error}", fg="red"))
            raise click.Abort()

    except Exception as e:
        click.echo(click.style(f"Error: {str(e)}", fg="red"))
        raise click.Abort()


@agent_group.command(name="review")
@click.argument("file_path", required=True)
@click.option("--report", "-r", help="Output path for review report")
def agent_review(file_path: str, report: Optional[str]):
    """Run Review Agent to analyze code quality."""
    try:
        agent = ReviewAgent()
        agent_context = AgentContext(session_id=f"review-{Path(file_path).stem}")

        prompt = f"Review the code at {file_path} for security and performance issues"

        click.echo(f"🔍 Review Agent: Reviewing '{file_path}'...")

        result = agent.execute(prompt, agent_context)

        if result.success:
            issues = result.metadata.get("issues_found", 0)
            score = result.metadata.get("quality_score", 0.0)

            if issues == 0:
                click.echo(click.style("✓ No issues found - code looks good!", fg="green"))
            else:
                click.echo(click.style(f"⚠ Found {issues} issue(s)", fg="yellow"))

            click.echo(f"Quality score: {score:.2f}/1.00")

            if report:
                click.echo(f"\nReport saved to: {report}")
        else:
            click.echo(click.style(f"✗ Review failed: {result.error}", fg="red"))
            raise click.Abort()

    except Exception as e:
        click.echo(click.style(f"Error: {str(e)}", fg="red"))
        raise click.Abort()


@agent_group.command(name="orchestrate")
@click.argument("plan_id", required=True)
@click.option("--auto", is_flag=True, help="Run automatically without confirmation")
@click.option("--stop-on-failure", is_flag=True, default=True, help="Stop if any task fails")
def agent_orchestrate(plan_id: str, auto: bool, stop_on_failure: bool):
    """Run Orchestrator to execute a work plan with agents."""
    try:
        # Load the plan
        storage = WorkPlanStorage()
        plan = storage.load_plan(plan_id)

        if not plan:
            click.echo(click.style(f"Plan '{plan_id}' not found", fg="red"))
            raise click.Abort()

        click.echo(f"📋 Plan: {plan.goal}")
        click.echo(f"   Tasks: {len(plan.tasks)}")

        if not auto:
            if not click.confirm("Execute this plan with agents?"):
                return

        # Create workflow
        workflow = AutomatedWorkflow()
        context = AgentContext(session_id=f"orchestrate-{plan_id}")

        click.echo("\n🤖 Orchestrator: Executing plan with multi-agent system...\n")

        # Progress callback
        def progress_callback(task_id, status, message):
            if status == "started":
                click.echo(f"  → Starting: {message}")
            elif status == "completed":
                click.echo(click.style(f"  ✓ Completed: {message}", fg="green"))
            elif status == "failed":
                click.echo(click.style(f"  ✗ Failed: {message}", fg="red"))

        # Execute
        result = workflow.execute_plan_automatically(
            plan,
            context,
            stop_on_failure=stop_on_failure,
            progress_callback=progress_callback
        )

        # Display results
        click.echo("\n" + "=" * 50)

        if result["success"]:
            click.echo(click.style("✓ Plan completed successfully!", fg="green"))
        else:
            click.echo(click.style("⚠ Plan completed with issues", fg="yellow"))

        click.echo(f"\nTasks completed: {result['tasks_completed']}/{result['total_tasks']}")
        click.echo(f"Completion: {result['completion_percentage']:.1f}%")

        if result["tasks_failed"] > 0:
            click.echo(click.style(f"Failed tasks: {result['tasks_failed']}", fg="red"))

    except Exception as e:
        click.echo(click.style(f"Error: {str(e)}", fg="red"))
        raise click.Abort()


@agent_group.command(name="list")
def agent_list():
    """List available agents."""
    agents = [
        ("design", "DesignAgent", "Creates technical designs and architecture"),
        ("test", "TestAgent", "Generates tests following TDD workflow"),
        ("implement", "ImplementationAgent", "Implements code from designs"),
        ("review", "ReviewAgent", "Reviews code for quality and security"),
        ("orchestrator", "OrchestratorAgent", "Coordinates multiple agents"),
    ]

    click.echo("Available Agents:\n")

    for name, class_name, description in agents:
        click.echo(f"  {click.style(name, fg='cyan', bold=True)}")
        click.echo(f"    Class: {class_name}")
        click.echo(f"    {description}\n")


@agent_group.command(name="workflow")
@click.argument("goal", required=True)
@click.option("--steps", "-s", multiple=True, help="Workflow steps (agent:task)")
def agent_workflow(goal: str, steps: tuple):
    """Execute a custom agent workflow."""
    try:
        if not steps:
            click.echo("No workflow steps provided. Use -s agent:task")
            return

        # Parse steps
        workflow_steps = []
        for step in steps:
            if ":" not in step:
                click.echo(f"Invalid step format: {step}. Use agent:task")
                continue

            agent_name, task = step.split(":", 1)
            workflow_steps.append({
                "agent": agent_name.strip(),
                "task": task.strip()
            })

        if not workflow_steps:
            click.echo("No valid steps to execute")
            return

        # Create orchestrator
        orchestrator = OrchestratorAgent()

        # Register agents
        orchestrator.register_subagent("design", DesignAgent())
        orchestrator.register_subagent("test", TestingAgent())
        orchestrator.register_subagent("implement", ImplementationAgent())
        orchestrator.register_subagent("review", ReviewAgent())

        click.echo(f"🎯 Goal: {goal}")
        click.echo(f"📝 Steps: {len(workflow_steps)}\n")

        # Execute workflow
        context = AgentContext(session_id=f"workflow-{goal}")

        result = orchestrator.coordinate_workflow(
            {"goal": goal, "steps": workflow_steps},
            context
        )

        if result["success"]:
            click.echo(click.style("\n✓ Workflow completed successfully!", fg="green"))
        else:
            click.echo(click.style("\n⚠ Workflow completed with issues", fg="yellow"))

        click.echo(f"Steps completed: {result['steps_completed']}/{result['total_steps']}")

    except Exception as e:
        click.echo(click.style(f"Error: {str(e)}", fg="red"))
        raise click.Abort()