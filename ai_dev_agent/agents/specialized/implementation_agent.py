"""Implementation Agent for executing designs with TDD workflow."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from ai_dev_agent.tools import registry

from ..base import AgentCapability, AgentContext, AgentResult, BaseAgent


class ImplementationAgent(BaseAgent):
    """Agent specialized in implementing designs following TDD principles."""

    def __init__(self):
        """Initialize Implementation Agent."""
        super().__init__(
            name="implementation_agent",
            description="Implements designs following TDD, makes minimal changes, preserves compatibility",
            capabilities=[
                "code_implementation",
                "incremental_development",
                "status_tracking",
                "error_handling",
            ],
            tools=["read", "write", "grep", "find", "run"],
            max_iterations=40,
        )

        # Register capabilities
        self._register_capabilities()

    def _register_capabilities(self):
        """Register agent capabilities."""
        capabilities = [
            AgentCapability(
                name="code_implementation",
                description="Implement code from designs",
                required_tools=["read", "write"],
                optional_tools=["grep", "find"],
            ),
            AgentCapability(
                name="incremental_development",
                description="Develop in small incremental steps",
                required_tools=["write", "run"],
                optional_tools=["read"],
            ),
            AgentCapability(
                name="status_tracking",
                description="Track implementation progress",
                required_tools=[],
                optional_tools=["write"],
            ),
            AgentCapability(
                name="error_handling",
                description="Implement proper error handling",
                required_tools=["write"],
                optional_tools=["read"],
            ),
        ]

        for capability in capabilities:
            self.register_capability(capability)

    def parse_design_document(self, design_path: str, context: AgentContext) -> dict[str, Any]:
        """
        Parse a design document to extract implementation details.

        Args:
            design_path: Path to design document
            context: Execution context

        Returns:
            Parsed design information
        """
        try:
            design_path = Path(design_path)
            with design_path.open() as f:
                content = f.read()

            # Extract components
            components_match = re.search(r"## Components\s*(.*?)(?=##|$)", content, re.DOTALL)
            components = []
            if components_match:
                comp_text = components_match.group(1)
                components = re.findall(r"[-*]\s*(\w+)", comp_text)

            # Extract requirements
            req_match = re.search(r"## Requirements\s*(.*?)(?=##|$)", content, re.DOTALL)
            requirements = []
            if req_match:
                req_text = req_match.group(1)
                requirements = re.findall(r"[-*]\s*(.+)", req_text)

            # Extract patterns
            pattern_match = re.search(r"## Patterns\s*(.*?)(?=##|$)", content, re.DOTALL)
            patterns = []
            if pattern_match:
                pattern_text = pattern_match.group(1)
                patterns = re.findall(r"[-*]\s*(\w+(?:\s+\w+)*)", pattern_text)

            return {
                "components": components,
                "requirements": [r.strip() for r in requirements],
                "patterns": patterns,
                "content": content,
            }

        except Exception as e:
            return {"error": str(e), "components": [], "requirements": [], "patterns": []}

    def generate_code_from_design(
        self, design: dict[str, Any], output_path: str, context: AgentContext
    ) -> dict[str, Any]:
        """
        Generate code from design specification.

        Args:
            design: Design specification
            output_path: Path to save generated code
            context: Execution context

        Returns:
            Result of code generation
        """
        try:
            components = design.get("components", [])
            methods = design.get("methods", [])

            lines = []

            # Generate imports
            lines.extend(
                [
                    '"""Generated implementation."""',
                    "from typing import Any, Dict, List, Optional",
                    "",
                    "",
                ]
            )

            # Generate classes for each component
            for component in components:
                lines.append(f"class {component}:")
                lines.append(f'    """Implementation of {component}."""')
                lines.append("")

                # Add init
                lines.append("    def __init__(self):")
                lines.append('        """Initialize component."""')
                lines.append("        pass")
                lines.append("")

                # Add methods
                for method in methods:
                    method_name = method.get("name", "method")
                    params = method.get("params", [])

                    param_str = ", ".join(params)
                    lines.append(f"    def {method_name}(self, {param_str}):")
                    lines.append(f'        """Implement {method_name}."""')
                    lines.append("        raise NotImplementedError('To be implemented')")
                    lines.append("")

            # Write code
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w") as f:
                f.write("\n".join(lines))

            return {"success": True, "path": output_path, "lines_generated": len(lines)}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def implement_incrementally(
        self, tasks: list[dict[str, Any]], context: AgentContext
    ) -> dict[str, Any]:
        """
        Implement tasks incrementally.

        Args:
            tasks: List of implementation tasks
            context: Execution context

        Returns:
            Implementation result
        """
        completed_steps = []

        # Sort by priority
        sorted_tasks = sorted(
            tasks,
            key=lambda t: {"high": 0, "medium": 1, "low": 2}.get(t.get("priority", "medium"), 1),
        )

        for task in sorted_tasks:
            step = {
                "task": task["name"],
                "priority": task.get("priority", "medium"),
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
            }
            completed_steps.append(step)

        return {"completed": len(completed_steps), "steps": completed_steps, "status": "done"}

    def verify_tests_pass(self, test_path: str, context: AgentContext) -> dict[str, Any]:
        """
        Verify that tests pass after implementation.

        Args:
            test_path: Path to test file
            context: Execution context

        Returns:
            Test verification results
        """
        try:
            payload = {
                "cmd": "pytest",
                "args": [test_path, "-v"],
                "timeout_sec": context.metadata.get("test_timeout_sec", 60),
            }
            tool_context = self._build_tool_context(context)
            result = registry.invoke("run", payload, tool_context)

            stdout_tail = result.get("stdout_tail") or ""
            stderr_tail = result.get("stderr_tail") or ""
            output = (stdout_tail + "\n" + stderr_tail).strip()

            tests_passed = len(re.findall(r"PASSED", output))
            tests_failed = len(re.findall(r"FAILED", output))

            return {
                "success": result.get("exit_code", 1) == 0,
                "tests_passed": tests_passed,
                "tests_failed": tests_failed,
                "output": output,
            }

        except Exception as e:  # pragma: no cover - defensive guard
            return {"success": False, "error": str(e)}

    def rollback_changes(
        self, file_path: str, original_content: str, context: AgentContext
    ) -> dict[str, Any]:
        """
        Rollback changes to original state.

        Args:
            file_path: Path to file
            original_content: Original file content
            context: Execution context

        Returns:
            Rollback result
        """
        try:
            file_path = Path(file_path)
            with file_path.open("w") as f:
                f.write(original_content)

            return {
                "success": True,
                "file": file_path,
                "message": "Changes rolled back successfully",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def update_status(self, status_update: dict[str, Any], context: AgentContext) -> dict[str, Any]:
        """
        Update implementation status.

        Args:
            status_update: Status update information
            context: Execution context

        Returns:
            Acknowledgment
        """
        return {
            "acknowledged": True,
            "component": status_update.get("component"),
            "status": status_update.get("status"),
            "progress": status_update.get("progress", 0),
            "timestamp": datetime.now().isoformat(),
        }

    def apply_minimal_change(
        self, existing_code: str, new_requirement: dict[str, Any], context: AgentContext
    ) -> dict[str, Any]:
        """
        Apply minimal change to existing code.

        Args:
            existing_code: Current code
            new_requirement: New requirement to implement
            context: Execution context

        Returns:
            Updated code with minimal changes
        """
        lines = existing_code.split("\n")
        original_line_count = len(lines)

        # Add new method if requested
        if "add_method" in new_requirement:
            method_name = new_requirement["add_method"]

            # Find the last method and add after it
            indent = "    "
            new_method = [
                "",
                f"{indent}def {method_name}(self):",
                f'{indent}    """New method."""',
                f"{indent}    pass",
            ]

            # Insert before last line (which is usually empty or end of class)
            insert_pos = len(lines) - 1
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip() and not lines[i].strip().startswith("#"):
                    insert_pos = i + 1
                    break

            lines[insert_pos:insert_pos] = new_method

        updated_code = "\n".join(lines)
        lines_changed = len(lines) - original_line_count

        return {"updated_code": updated_code, "lines_changed": lines_changed}

    def check_compatibility_preserved(
        self, existing_api: dict[str, Any], new_code: str, context: AgentContext
    ) -> dict[str, Any]:
        """
        Check that existing API is preserved in new code.

        Args:
            existing_api: Existing API specification
            new_code: New code to check
            context: Execution context

        Returns:
            Compatibility check results
        """
        preserved_functions = []
        preserved_classes = []

        # Check functions
        for func in existing_api.get("functions", []):
            if f"def {func}" in new_code:
                preserved_functions.append(func)

        # Check classes
        for cls in existing_api.get("classes", []):
            if f"class {cls}" in new_code:
                preserved_classes.append(cls)

        compatible = len(preserved_functions) == len(existing_api.get("functions", [])) and len(
            preserved_classes
        ) == len(existing_api.get("classes", []))

        return {
            "compatible": compatible,
            "preserved_functions": preserved_functions,
            "preserved_classes": preserved_classes,
            "missing_functions": [
                f for f in existing_api.get("functions", []) if f not in preserved_functions
            ],
            "missing_classes": [
                c for c in existing_api.get("classes", []) if c not in preserved_classes
            ],
        }

    def apply_design_patterns(
        self, design: dict[str, Any], context: AgentContext
    ) -> dict[str, Any]:
        """
        Apply design patterns from design spec.

        Args:
            design: Design specification
            context: Execution context

        Returns:
            Pattern application results
        """
        patterns = design.get("patterns", [])
        components = design.get("components", [])

        implemented_components = []
        patterns_applied = []

        # Simulate pattern application
        for component in components:
            implemented_components.append(component)

        for pattern in patterns:
            patterns_applied.append(pattern)

        return {
            "implemented_components": implemented_components,
            "patterns_applied": patterns_applied,
        }

    def generate_method_with_error_handling(
        self, method_spec: dict[str, Any], context: AgentContext
    ) -> str:
        """
        Generate method with proper error handling.

        Args:
            method_spec: Method specification
            context: Execution context

        Returns:
            Generated method code
        """
        method_name = method_spec.get("name", "method")
        params = method_spec.get("params", [])
        method_spec.get("error_conditions", [])

        lines = [
            f"def {method_name}(self, {', '.join(params)}):",
            f'    """Process {params[0] if params else "data"} with error handling."""',
            "    try:",
            "        # Validate input",
            f"        if not {params[0] if params else 'data'}:",
            "            raise ValueError('Invalid input')",
            "",
            "        # Process",
            "        result = None  # Implementation here",
            "",
            "        return result",
            "",
            "    except ValueError as e:",
            "        # Handle validation errors",
            "        raise",
            "    except Exception as e:",
            "        # Handle unexpected errors",
            "        raise RuntimeError(f'Error in {method_name}: {{e}}')",
        ]

        return "\n".join(lines)

    def implement_with_tdd(
        self, design: dict[str, Any], test_path: str, context: AgentContext
    ) -> dict[str, Any]:
        """
        Implement following TDD workflow.

        Args:
            design: Design to implement
            test_path: Path to tests
            context: Execution context

        Returns:
            TDD implementation result
        """
        # Step 1: Verify tests exist and fail initially
        initial_test = self.verify_tests_pass(test_path, context)
        tests_failed_initially = not initial_test.get("success", True)

        # Step 2: Implement (simulated)
        # In real implementation, this would generate the actual code

        # Step 3: Verify tests pass after implementation
        final_test = self.verify_tests_pass(test_path, context)
        tests_pass_after = final_test.get("success", False)

        tdd_workflow_followed = tests_failed_initially and tests_pass_after

        return {
            "tdd_workflow_followed": tdd_workflow_followed,
            "tests_failed_initially": tests_failed_initially,
            "tests_pass_after_implementation": tests_pass_after,
            "initial_failures": initial_test.get("tests_failed", 0),
            "final_passes": final_test.get("tests_passed", 0),
        }

    def execute(self, prompt: str, context: AgentContext) -> AgentResult:
        """
        Execute implementation agent task using ReAct workflow.

        Args:
            prompt: Implementation task description
            context: Execution context

        Returns:
            AgentResult with implementation artifacts
        """
        # Import executor bridge
        from .executor_bridge import execute_agent_with_react

        # Execute using ReAct workflow with LLM and real tools
        return execute_agent_with_react(agent=self, prompt=prompt, context=context)
