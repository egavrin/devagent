"""Tests for Implementation Agent."""
import pytest
from unittest.mock import Mock, patch, mock_open
import tempfile
import os

from ai_dev_agent.agents.specialized.implementation_agent import ImplementationAgent
from ai_dev_agent.agents.base import AgentContext, AgentResult


class TestImplementationAgent:
    """Test Implementation Agent functionality."""

    def test_implementation_agent_initialization(self):
        """Test creating an implementation agent."""
        agent = ImplementationAgent()

        assert agent.name == "implementation_agent"
        assert "read" in agent.tools
        assert "write" in agent.tools
        assert "grep" in agent.tools
        assert "run" in agent.tools
        assert agent.max_iterations == 40

    def test_implementation_agent_capabilities(self):
        """Test implementation agent has correct capabilities."""
        agent = ImplementationAgent()

        assert "code_implementation" in agent.capabilities
        assert "incremental_development" in agent.capabilities
        assert "status_tracking" in agent.capabilities
        assert "error_handling" in agent.capabilities

    def test_parse_design_document(self):
        """Test parsing a design document."""
        agent = ImplementationAgent()
        context = AgentContext(session_id="test-parse")

        design_content = """
# Design: User Authentication

## Components
- AuthController
- AuthService
- TokenManager

## Requirements
- JWT token generation
- Password hashing
- Refresh token support

## Implementation Notes
Use bcrypt for password hashing.
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(design_content)
            design_path = f.name

        try:
            result = agent.parse_design_document(design_path, context)

            assert result["components"] is not None
            assert "AuthController" in str(result["components"])
            assert "requirements" in result

        finally:
            os.unlink(design_path)

    def test_generate_code_from_design(self):
        """Test generating code from design."""
        agent = ImplementationAgent()
        context = AgentContext(session_id="test-gen-code")

        design = {
            "components": ["UserService"],
            "methods": [
                {"name": "create_user", "params": ["username", "email"]},
                {"name": "get_user", "params": ["user_id"]}
            ]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "user_service.py")

            result = agent.generate_code_from_design(
                design,
                output_file,
                context
            )

            assert result["success"] is True
            assert os.path.exists(output_file)

            with open(output_file, 'r') as f:
                content = f.read()
                assert "class UserService" in content
                assert "def create_user" in content
                assert "def get_user" in content

    def test_implement_incrementally(self):
        """Test incremental implementation."""
        agent = ImplementationAgent()
        context = AgentContext(session_id="test-incremental")

        tasks = [
            {"name": "Create base class", "priority": "high"},
            {"name": "Add methods", "priority": "medium"},
            {"name": "Add error handling", "priority": "low"}
        ]

        result = agent.implement_incrementally(tasks, context)

        assert "completed" in result
        assert "steps" in result
        assert len(result["steps"]) > 0

    def test_verify_tests_pass(self):
        """Test verifying tests pass after implementation."""
        agent = ImplementationAgent()
        context = AgentContext(session_id="test-verify")

        # Mock test execution
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout="test_module.py::test_1 PASSED\ntest_module.py::test_2 PASSED\n5 PASSED in 1.2s",
                stderr=""
            )

            result = agent.verify_tests_pass("tests/test_module.py", context)

            assert result["success"] is True
            assert result["tests_passed"] >= 2

    def test_rollback_on_failure(self):
        """Test rollback when implementation breaks tests."""
        agent = ImplementationAgent()
        context = AgentContext(session_id="test-rollback")

        original_code = "def original_function():\n    return True"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(original_code)
            file_path = f.name

        try:
            # Simulate breaking change
            with open(file_path, 'w') as f:
                f.write("def original_function():\n    raise Exception('broken')")

            # Rollback
            result = agent.rollback_changes(file_path, original_code, context)

            assert result["success"] is True

            # Verify code was restored
            with open(file_path, 'r') as f:
                restored = f.read()
                assert restored == original_code

        finally:
            os.unlink(file_path)

    def test_update_status_tracking(self):
        """Test updating implementation status."""
        agent = ImplementationAgent()
        context = AgentContext(session_id="test-status")

        status_update = {
            "component": "AuthService",
            "status": "completed",
            "progress": 75,
            "notes": "Implemented JWT generation"
        }

        result = agent.update_status(status_update, context)

        assert result["acknowledged"] is True
        assert result["progress"] == 75

    def test_minimal_changes_principle(self):
        """Test that implementation makes minimal changes."""
        agent = ImplementationAgent()
        context = AgentContext(session_id="test-minimal")

        existing_code = """
class ExistingClass:
    def method1(self):
        pass

    def method2(self):
        pass
"""

        new_requirement = {
            "add_method": "method3"
        }

        result = agent.apply_minimal_change(
            existing_code,
            new_requirement,
            context
        )

        # Should only add method3, not change existing methods
        assert "method3" in result["updated_code"]
        assert "def method1(self):" in result["updated_code"]
        assert "def method2(self):" in result["updated_code"]
        assert result["lines_changed"] < 10  # Minimal change

    def test_compatibility_preservation(self):
        """Test that implementation preserves compatibility."""
        agent = ImplementationAgent()
        context = AgentContext(session_id="test-compat-pres")

        existing_api = {
            "functions": ["get_user", "create_user"],
            "classes": ["User", "Session"]
        }

        new_code = """
class User:
    pass

class Session:
    pass

def get_user(user_id):
    return None

def create_user(data):
    return None

def new_helper_function():
    return None
"""

        result = agent.check_compatibility_preserved(
            existing_api,
            new_code,
            context
        )

        assert result["compatible"] is True
        assert all(f in result["preserved_functions"] for f in existing_api["functions"])
        assert all(c in result["preserved_classes"] for c in existing_api["classes"])

    def test_follow_design_patterns(self):
        """Test following design patterns from design doc."""
        agent = ImplementationAgent()
        context = AgentContext(session_id="test-patterns")

        design = {
            "patterns": ["MVC", "Repository Pattern"],
            "components": ["UserController", "UserRepository"]
        }

        result = agent.apply_design_patterns(design, context)

        assert "UserController" in result["implemented_components"]
        assert "UserRepository" in result["implemented_components"]
        assert "patterns_applied" in result

    def test_error_handling_implementation(self):
        """Test proper error handling in generated code."""
        agent = ImplementationAgent()
        context = AgentContext(session_id="test-errors")

        method_spec = {
            "name": "process_data",
            "params": ["data"],
            "error_conditions": ["invalid_data", "network_error"]
        }

        code = agent.generate_method_with_error_handling(method_spec, context)

        assert "try:" in code or "except" in code
        assert "raise" in code or "Exception" in code

    @pytest.mark.llm
    def test_implementation_agent_execute(self):
        """Test full implementation agent execution."""
        agent = ImplementationAgent()
        context = AgentContext(session_id="test-execute")

        prompt = """
        Implement the User Authentication design at docs/design/auth_design.md.
        Follow TDD - tests should already exist at tests/test_auth.py.
        Make minimal changes and preserve backward compatibility.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            context.working_directory = tmpdir

            # Create mock design file
            design_path = os.path.join(tmpdir, "auth_design.md")
            os.makedirs(os.path.dirname(design_path), exist_ok=True)
            with open(design_path, 'w') as f:
                f.write("# Design: Auth\n## Components\n- AuthService")

            with patch('os.makedirs'):
                with patch('builtins.open', mock_open()):
                    with patch('subprocess.run') as mock_run:
                        mock_run.return_value = Mock(returncode=0, stdout="tests passed")

                        result = agent.execute(prompt, context)

            assert result.success is True
            assert "files_created" in result.metadata or "files_modified" in result.metadata

    def test_implementation_follows_tdd(self):
        """Test that implementation follows TDD workflow."""
        agent = ImplementationAgent()
        context = AgentContext(session_id="test-tdd")

        # Mock test execution showing tests exist and fail initially
        with patch('subprocess.run') as mock_run:
            # First call: tests fail (expected in TDD)
            # Second call: tests pass after implementation
            mock_run.side_effect = [
                Mock(returncode=1, stdout="test_1 FAILED\ntest_2 FAILED\n5 FAILED", stderr=""),  # Before implementation
                Mock(returncode=0, stdout="test_1 PASSED\ntest_2 PASSED\n5 PASSED", stderr="")   # After implementation
            ]

            result = agent.implement_with_tdd(
                design={"component": "TestComponent"},
                test_path="tests/test_component.py",
                context=context
            )

            assert result["tdd_workflow_followed"] is True
            assert result["tests_failed_initially"] is True
            assert result["tests_pass_after_implementation"] is True