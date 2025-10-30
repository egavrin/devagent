"""Tests for Test Agent."""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import pytest

from ai_dev_agent.agents.base import AgentContext
from ai_dev_agent.agents.specialized.testing_agent import TestingAgent


class TestTestAgent:
    """Test Test Agent functionality."""

    def test_test_agent_initialization(self):
        """Test creating a test agent."""
        agent = TestingAgent()

        assert agent.name == "test_agent"
        assert "read" in agent.tools
        assert "write" in agent.tools
        assert "grep" in agent.tools
        assert "run" in agent.tools  # For running tests
        assert agent.max_iterations == 25

    def test_test_agent_capabilities(self):
        """Test test agent has correct capabilities."""
        agent = TestingAgent()

        assert "test_generation" in agent.capabilities
        assert "tdd_workflow" in agent.capabilities
        assert "coverage_analysis" in agent.capabilities
        assert "fixture_creation" in agent.capabilities

    def test_analyze_design_for_tests(self):
        """Test analyzing a design to generate test cases."""
        agent = TestingAgent()
        context = AgentContext(session_id="test-analyze")

        design = {
            "feature": "User Authentication",
            "components": ["AuthController", "AuthService", "TokenManager"],
            "requirements": ["JWT tokens", "Password hashing", "Refresh tokens"],
            "api_endpoints": [
                {"method": "POST", "path": "/login", "handler": "login"},
                {"method": "POST", "path": "/refresh", "handler": "refresh_token"},
            ],
        }

        test_plan = agent.analyze_design_for_tests(design, context)

        assert test_plan["status"] == "analyzed"
        assert "test_cases" in test_plan
        assert len(test_plan["test_cases"]) > 0
        assert any("login" in tc["name"].lower() for tc in test_plan["test_cases"])
        assert any("token" in tc["name"].lower() for tc in test_plan["test_cases"])

    def test_generate_unit_tests(self):
        """Test generating unit tests for a module."""
        agent = TestingAgent()
        context = AgentContext(session_id="test-gen-unit")

        module_spec = {
            "module": "auth_service",
            "class": "AuthService",
            "methods": [
                {"name": "create_user", "params": ["username", "password"], "returns": "User"},
                {"name": "authenticate", "params": ["username", "password"], "returns": "bool"},
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = str(Path(tmpdir) / "test_auth_service.py")

            result = agent.generate_unit_tests(module_spec, test_file, context)

            assert result["success"] is True
            assert Path(test_file).exists()

            # Check test content
            with Path(test_file).open() as f:
                content = f.read()
                assert "def test_create_user" in content
                assert "def test_authenticate" in content
                assert "pytest" in content or "unittest" in content

    def test_generate_integration_tests(self):
        """Test generating integration tests."""
        agent = TestingAgent()
        context = AgentContext(session_id="test-gen-int")

        integration_spec = {
            "system": "User Management API",
            "endpoints": [
                {"method": "POST", "path": "/users", "expected_status": 201},
                {"method": "GET", "path": "/users/1", "expected_status": 200},
            ],
            "dependencies": ["database", "auth_service"],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = str(Path(tmpdir) / "test_integration.py")

            result = agent.generate_integration_tests(integration_spec, test_file, context)

            assert result["success"] is True
            assert Path(test_file).exists()

            with Path(test_file).open() as f:
                content = f.read()
                assert "integration" in content.lower() or "test" in content.lower()
                assert "/users" in content

    def test_create_test_fixtures(self):
        """Test creating test fixtures and mock data."""
        agent = TestingAgent()
        context = AgentContext(session_id="test-fixtures")

        fixture_spec = {
            "models": ["User", "Post", "Comment"],
            "data": {
                "User": {"username": "testuser", "email": "test@example.com"},
                "Post": {"title": "Test Post", "content": "Content"},
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            fixture_file = str(Path(tmpdir) / "conftest.py")

            result = agent.create_test_fixtures(fixture_spec, fixture_file, context)

            assert result["success"] is True
            assert Path(fixture_file).exists()

            with Path(fixture_file).open() as f:
                content = f.read()
                assert "@pytest.fixture" in content
                assert "User" in content

    def test_calculate_coverage_requirements(self):
        """Test calculating coverage requirements."""
        agent = TestingAgent()

        module_info = {
            "name": "auth_service",
            "functions": 10,
            "classes": 2,
            "lines": 150,
            "complexity": "medium",
        }

        requirements = agent.calculate_coverage_requirements(module_info)

        assert requirements["target_coverage"] >= 0.9  # 90% minimum
        assert requirements["required_test_count"] > 0
        assert "critical_paths" in requirements

    def test_validate_test_coverage(self):
        """Test validating test coverage."""
        agent = TestingAgent()
        context = AgentContext(session_id="test-coverage")

        coverage_data = {
            "total_lines": 200,
            "covered_lines": 185,
            "total_branches": 50,
            "covered_branches": 45,
            "uncovered_files": ["util.py"],
            "uncovered_lines": {"main.py": [10, 25, 30]},
        }

        validation = agent.validate_test_coverage(coverage_data, context)

        assert "coverage_percentage" in validation
        assert validation["coverage_percentage"] == 92.5  # 185/200
        assert "meets_requirements" in validation
        assert validation["meets_requirements"] is True  # >= 90%

    def test_run_tests_and_report(self):
        """Test running tests and generating report."""
        agent = TestingAgent()
        context = AgentContext(session_id="test-run")

        # Mock test execution
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="10 passed in 2.5s", stderr="")

            result = agent.run_tests(test_path="tests/test_auth.py", context=context)

            assert result["success"] is True
            assert result["tests_passed"] >= 0
            assert "duration" in result

    def test_ensure_tests_fail_before_implementation(self):
        """Test that TDD workflow ensures tests fail first."""
        agent = TestingAgent()
        context = AgentContext(session_id="test-tdd")

        test_spec = {
            "module": "new_feature",
            "test_cases": [{"name": "test_new_functionality", "expected": "NotImplementedError"}],
        }

        result = agent.ensure_tests_fail_first(test_spec, context)

        assert "status" in result
        assert result["status"] in ["ready_for_implementation", "tests_need_fixing"]

    def test_generate_backward_compatibility_tests(self):
        """Test generating backward compatibility tests."""
        agent = TestingAgent()
        context = AgentContext(session_id="test-compat")

        api_spec = {
            "existing_functions": [
                {"name": "get_user", "signature": "get_user(id: int) -> User"},
                {"name": "create_user", "signature": "create_user(data: dict) -> User"},
            ],
            "existing_classes": ["User", "Session"],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            compat_file = str(Path(tmpdir) / "test_backward_compatibility.py")

            result = agent.generate_backward_compatibility_tests(api_spec, compat_file, context)

            assert result["success"] is True
            assert Path(compat_file).exists()

            with Path(compat_file).open() as f:
                content = f.read()
                assert "compatibility" in content.lower() or "backward" in content.lower()
                assert "get_user" in content

    @pytest.mark.llm
    def test_test_agent_execute(self):
        """Test full test agent execution."""
        agent = TestingAgent()
        context = AgentContext(session_id="test-execute")

        prompt = """
        Create tests for User Authentication feature with:
        - Login endpoint tests
        - JWT token validation tests
        - Password hashing tests
        Ensure 90% coverage and TDD workflow.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            context.working_directory = tmpdir

            with patch("os.makedirs"):
                with patch("builtins.open", mock_open()):
                    result = agent.execute(prompt, context)

            assert result.success is True
            assert "test_files_created" in result.metadata
            assert "coverage_target" in result.metadata

    def test_pytest_vs_unittest_detection(self):
        """Test detection of pytest vs unittest framework."""
        agent = TestingAgent()

        # Should prefer pytest
        framework = agent.detect_test_framework("/path/to/project")
        assert framework in ["pytest", "unittest"]

    def test_test_agent_with_existing_tests(self):
        """Test agent can extend existing test suites."""
        agent = TestingAgent()
        context = AgentContext(session_id="test-extend")

        existing_tests = """
def test_existing_function():
    assert existing_function() == True
"""

        new_spec = {"add_tests_for": ["new_function", "another_function"]}

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = str(Path(tmpdir) / "test_module.py")

            with Path(test_file).open("w") as f:
                f.write(existing_tests)

            result = agent.extend_existing_tests(test_file, new_spec, context)

            assert result["success"] is True
            assert result["tests_added"] > 0
