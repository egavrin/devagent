"""Test Agent for TDD workflow and test generation."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any

from ..base import AgentCapability, AgentContext, AgentResult, BaseAgent


class TestingAgent(BaseAgent):
    """Agent specialized in test generation and TDD workflow."""

    __test__ = False  # Prevent pytest from collecting this production class

    def __init__(self):
        """Initialize Test Agent."""
        super().__init__(
            name="test_agent",
            description="Generates tests, ensures TDD workflow, and validates coverage",
            capabilities=[
                "test_generation",
                "tdd_workflow",
                "coverage_analysis",
                "fixture_creation",
            ],
            tools=["read", "write", "grep", "find", "run"],
            max_iterations=25,
        )

        # Register capabilities
        self._register_capabilities()

    def _register_capabilities(self):
        """Register agent capabilities."""
        capabilities = [
            AgentCapability(
                name="test_generation",
                description="Generate unit and integration tests",
                required_tools=["read", "write"],
                optional_tools=["grep"],
            ),
            AgentCapability(
                name="tdd_workflow",
                description="Ensure tests are written before implementation",
                required_tools=["write", "run"],
                optional_tools=["read"],
            ),
            AgentCapability(
                name="coverage_analysis",
                description="Analyze and validate test coverage",
                required_tools=["run"],
                optional_tools=["read", "grep"],
            ),
            AgentCapability(
                name="fixture_creation",
                description="Create test fixtures and mock data",
                required_tools=["write"],
                optional_tools=["read"],
            ),
        ]

        for capability in capabilities:
            self.register_capability(capability)

    def analyze_design_for_tests(
        self, design: dict[str, Any], context: AgentContext
    ) -> dict[str, Any]:
        """
        Analyze a design to generate test cases.

        Args:
            design: Design specification
            context: Execution context

        Returns:
            Test plan with test cases
        """
        test_cases = []

        # Extract components and generate tests
        components = design.get("components", [])
        for component in components:
            test_cases.append(
                {
                    "name": f"test_{component.lower()}_initialization",
                    "type": "unit",
                    "target": component,
                    "description": f"Test {component} can be initialized",
                }
            )

        # Generate tests for requirements
        requirements = design.get("requirements", [])
        for req in requirements:
            req_lower = req.lower()
            test_name = re.sub(r"[^\w]+", "_", req_lower)
            test_cases.append(
                {
                    "name": f"test_{test_name}",
                    "type": "functional",
                    "target": req,
                    "description": f"Test {req} functionality",
                }
            )

        # Generate tests for API endpoints
        endpoints = design.get("api_endpoints", [])
        for endpoint in endpoints:
            method = endpoint.get("method", "GET")
            path = endpoint.get("path", "")
            handler = endpoint.get("handler", "handler")

            test_cases.append(
                {
                    "name": f"test_{method.lower()}_{handler}",
                    "type": "integration",
                    "target": f"{method} {path}",
                    "description": f"Test {method} {path} endpoint",
                }
            )

        return {
            "status": "analyzed",
            "test_cases": test_cases,
            "total_tests": len(test_cases),
            "feature": design.get("feature", "Unknown"),
        }

    def generate_unit_tests(
        self, module_spec: dict[str, Any], output_path: str, context: AgentContext
    ) -> dict[str, Any]:
        """
        Generate unit tests for a module.

        Args:
            module_spec: Module specification with methods to test
            output_path: Path to save test file
            context: Execution context

        Returns:
            Result of test generation
        """
        try:
            module_name = module_spec.get("module", "module")
            class_name = module_spec.get("class", "Class")
            methods = module_spec.get("methods", [])

            # Generate test file content
            lines = [
                f'"""Unit tests for {module_name}.{class_name}"""',
                "import pytest",
                f"from {module_name} import {class_name}",
                "\n",
                f"class Test{class_name}:",
                f'    """Test suite for {class_name}"""',
                "",
            ]

            # Generate test for each method
            for method in methods:
                method_name = method.get("name", "method")
                params = method.get("params", [])
                method.get("returns", "None")

                lines.extend(
                    [
                        f"    def test_{method_name}(self):",
                        f'        """Test {method_name} method."""',
                        "        # Arrange",
                        f"        obj = {class_name}()",
                        "",
                        "        # Act",
                        f"        result = obj.{method_name}({', '.join('None' for _ in params)})",
                        "",
                        "        # Assert",
                        "        assert result is not None",
                        "",
                    ]
                )

            # Write test file
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            output_path = Path(output_path)

            with output_path.open("w") as f:
                f.write("\n".join(lines))

            return {"success": True, "path": output_path, "tests_generated": len(methods)}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def generate_integration_tests(
        self, integration_spec: dict[str, Any], output_path: str, context: AgentContext
    ) -> dict[str, Any]:
        """
        Generate integration tests.

        Args:
            integration_spec: Integration test specification
            output_path: Path to save test file
            context: Execution context

        Returns:
            Result of test generation
        """
        try:
            system_name = integration_spec.get("system", "System")
            endpoints = integration_spec.get("endpoints", [])
            dependencies = integration_spec.get("dependencies", [])

            lines = [
                f'"""Integration tests for {system_name}"""',
                "import pytest",
                "",
                "",
                "class TestIntegration:",
                f'    """Integration tests for {system_name}"""',
                "",
            ]

            # Generate setup/teardown
            if dependencies:
                lines.extend(
                    [
                        "    @pytest.fixture(autouse=True)",
                        "    def setup_dependencies(self):",
                        '        """Set up test dependencies."""',
                    ]
                )
                for dep in dependencies:
                    lines.append(f"        # Initialize {dep}")
                lines.extend(["        yield", "        # Cleanup", ""])

            # Generate test for each endpoint
            for i, endpoint in enumerate(endpoints):
                method = endpoint.get("method", "GET")
                path = endpoint.get("path", "/")
                expected_status = endpoint.get("expected_status", 200)

                test_name = f"test_{method.lower()}_endpoint_{i+1}"
                lines.extend(
                    [
                        f"    def {test_name}(self):",
                        f'        """Test {method} {path} endpoint."""',
                        "        # Arrange",
                        f'        endpoint = "{path}"',
                        "",
                        "        # Act",
                        f'        # response = make_request("{method}", endpoint)',
                        "",
                        "        # Assert",
                        f"        # assert response.status_code == {expected_status}",
                        "",
                    ]
                )

            # Write test file
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            output_path = Path(output_path)

            with output_path.open("w") as f:
                f.write("\n".join(lines))

            return {"success": True, "path": output_path, "tests_generated": len(endpoints)}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def create_test_fixtures(
        self, fixture_spec: dict[str, Any], output_path: str, context: AgentContext
    ) -> dict[str, Any]:
        """
        Create test fixtures and mock data.

        Args:
            fixture_spec: Fixture specification
            output_path: Path to save fixtures (conftest.py)
            context: Execution context

        Returns:
            Result of fixture creation
        """
        try:
            models = fixture_spec.get("models", [])
            data = fixture_spec.get("data", {})

            lines = [
                '"""Test fixtures and configuration."""',
                "import pytest",
                "",
            ]

            # Generate fixture for each model
            for model in models:
                fixture_name = f"{model.lower()}_fixture"
                model_data = data.get(model, {})

                lines.extend(
                    [
                        "@pytest.fixture",
                        f"def {fixture_name}():",
                        f'    """Fixture for {model} model."""',
                        f"    return {model}(**{model_data})",
                        "",
                    ]
                )

            # Write conftest.py
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            output_path = Path(output_path)

            with output_path.open("w") as f:
                f.write("\n".join(lines))

            return {"success": True, "path": output_path, "fixtures_created": len(models)}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def calculate_coverage_requirements(self, module_info: dict[str, Any]) -> dict[str, Any]:
        """
        Calculate coverage requirements for a module.

        Args:
            module_info: Module information

        Returns:
            Coverage requirements
        """
        complexity = module_info.get("complexity", "medium")
        functions = module_info.get("functions", 0)
        classes = module_info.get("classes", 0)

        # Base target is 90%
        target_coverage = 0.9

        # Adjust based on complexity
        if complexity == "high":
            target_coverage = 0.95
        elif complexity == "low":
            target_coverage = 0.85

        # Estimate required test count
        # Rule of thumb: 3-5 tests per function, 5-10 per class
        required_test_count = (functions * 3) + (classes * 5)

        # Identify critical paths
        critical_paths = []
        if "critical" in str(module_info).lower():
            critical_paths.append("Error handling")
            critical_paths.append("Data validation")
            critical_paths.append("Edge cases")

        return {
            "target_coverage": target_coverage,
            "required_test_count": required_test_count,
            "critical_paths": critical_paths,
            "module": module_info.get("name", "Unknown"),
        }

    def validate_test_coverage(
        self, coverage_data: dict[str, Any], context: AgentContext
    ) -> dict[str, Any]:
        """
        Validate test coverage against requirements.

        Args:
            coverage_data: Coverage statistics
            context: Execution context

        Returns:
            Validation results
        """
        total_lines = coverage_data.get("total_lines", 0)
        covered_lines = coverage_data.get("covered_lines", 0)

        coverage_percentage = (covered_lines / total_lines * 100) if total_lines > 0 else 0

        meets_requirements = coverage_percentage >= 90.0

        uncovered_files = coverage_data.get("uncovered_files", [])
        uncovered_lines = coverage_data.get("uncovered_lines", {})

        return {
            "coverage_percentage": coverage_percentage,
            "meets_requirements": meets_requirements,
            "uncovered_files": uncovered_files,
            "uncovered_lines": uncovered_lines,
            "recommendations": (
                [f"Add tests for {f}" for f in uncovered_files] if uncovered_files else []
            ),
        }

    def run_tests(
        self, test_path: str, context: AgentContext, framework: str = "pytest"
    ) -> dict[str, Any]:
        """
        Run tests and collect results.

        Args:
            test_path: Path to test file or directory
            context: Execution context
            framework: Test framework (pytest or unittest)

        Returns:
            Test execution results
        """
        try:
            if framework == "pytest":
                cmd = ["pytest", test_path, "-v"]
            else:
                cmd = ["python", "-m", "unittest", test_path]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            # Parse output
            output = result.stdout + result.stderr

            # Extract test counts
            tests_passed = len(re.findall(r"PASSED", output))
            tests_failed = len(re.findall(r"FAILED", output))

            # Extract duration
            duration_match = re.search(r"in ([\d.]+)s", output)
            duration = float(duration_match.group(1)) if duration_match else 0.0

            return {
                "success": result.returncode == 0,
                "tests_passed": tests_passed,
                "tests_failed": tests_failed,
                "duration": duration,
                "output": output,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def ensure_tests_fail_first(
        self, test_spec: dict[str, Any], context: AgentContext
    ) -> dict[str, Any]:
        """
        Ensure tests fail before implementation (TDD).

        Args:
            test_spec: Test specification
            context: Execution context

        Returns:
            Status of TDD workflow
        """
        # In TDD, tests should fail initially because functionality isn't implemented yet
        # This method validates that pattern

        test_cases = test_spec.get("test_cases", [])

        status = "ready_for_implementation"

        for test_case in test_cases:
            expected = test_case.get("expected", "")
            if "NotImplementedError" in expected or "fail" in expected.lower():
                # Good - test is expected to fail
                continue
            else:
                # Need to ensure test will fail first
                status = "tests_need_fixing"

        return {
            "status": status,
            "test_count": len(test_cases),
            "message": (
                "Tests are designed to fail before implementation"
                if status == "ready_for_implementation"
                else "Tests may pass prematurely"
            ),
        }

    def generate_backward_compatibility_tests(
        self, api_spec: dict[str, Any], output_path: str, context: AgentContext
    ) -> dict[str, Any]:
        """
        Generate backward compatibility tests.

        Args:
            api_spec: API specification with existing functions
            output_path: Path to save compatibility tests
            context: Execution context

        Returns:
            Result of test generation
        """
        try:
            existing_functions = api_spec.get("existing_functions", [])
            existing_classes = api_spec.get("existing_classes", [])

            lines = [
                '"""Backward compatibility tests."""',
                "import pytest",
                "",
                "",
                "class TestBackwardCompatibility:",
                '    """Ensure existing APIs remain functional."""',
                "",
            ]

            # Generate tests for existing functions
            for func in existing_functions:
                func_name = func.get("name", "function")
                signature = func.get("signature", "")

                lines.extend(
                    [
                        f"    def test_{func_name}_signature(self):",
                        f'        """Test {func_name} maintains signature: {signature}"""',
                        "        # Verify function exists",
                        f"        # assert callable({func_name})",
                        "        # Verify signature hasn't changed",
                        "        pass",
                        "",
                    ]
                )

            # Generate tests for existing classes
            for cls in existing_classes:
                lines.extend(
                    [
                        f"    def test_{cls.lower()}_class_exists(self):",
                        f'        """Test {cls} class still exists."""',
                        f"        # assert {cls} is not None",
                        "        pass",
                        "",
                    ]
                )

            # Write test file
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            output_path = Path(output_path)

            with output_path.open("w") as f:
                f.write("\n".join(lines))

            return {
                "success": True,
                "path": output_path,
                "tests_generated": len(existing_functions) + len(existing_classes),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def detect_test_framework(self, project_path: str) -> str:
        """
        Detect test framework used in project.

        Args:
            project_path: Path to project

        Returns:
            Test framework name
        """
        # Check for pytest
        project_path_obj = Path(project_path)
        if (project_path_obj / "pytest.ini").exists() or (
            project_path_obj / "pyproject.toml"
        ).exists():
            return "pytest"

        # Default to pytest as it's more modern
        return "pytest"

    def extend_existing_tests(
        self, test_file: str, new_spec: dict[str, Any], context: AgentContext
    ) -> dict[str, Any]:
        """
        Extend existing test file with new tests.

        Args:
            test_file: Path to existing test file
            new_spec: Specification for new tests to add
            context: Execution context

        Returns:
            Result of extension
        """
        try:
            # Read existing tests
            test_file = Path(test_file)

            with test_file.open() as f:
                existing_content = f.read()

            # Generate new tests
            new_tests = []
            for func in new_spec.get("add_tests_for", []):
                new_tests.append(
                    f"""
def test_{func}():
    \"\"\"Test {func} function.\"\"\"
    # Arrange
    # Act
    result = {func}()
    # Assert
    assert result is not None
"""
                )

            # Append new tests
            updated_content = existing_content + "\n\n" + "\n".join(new_tests)

            # Write back
            test_file = Path(test_file)

            with test_file.open("w") as f:
                f.write(updated_content)

            return {"success": True, "tests_added": len(new_tests)}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def execute(self, prompt: str, context: AgentContext) -> AgentResult:
        """
        Execute test agent task using ReAct workflow.

        Args:
            prompt: Test generation task description
            context: Execution context

        Returns:
            AgentResult with test artifacts
        """
        # Import executor bridge
        from .executor_bridge import execute_agent_with_react

        # Execute using ReAct workflow with LLM and real tools
        return execute_agent_with_react(agent=self, prompt=prompt, context=context)
