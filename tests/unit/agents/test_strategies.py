"""Tests for agent strategies."""

from unittest.mock import Mock, patch

import pytest

from ai_dev_agent.agents.strategies import (
    AgentStrategy,
    DesignAgentStrategy,
    ImplementationAgentStrategy,
    ReviewAgentStrategy,
)
from ai_dev_agent.agents.strategies import TestAgentStrategy as TestGenerationStrategy


class TestAgentStrategy:
    """Test the base AgentStrategy class."""

    def test_abstract_class_cannot_instantiate(self):
        """Test that abstract base class cannot be instantiated."""
        with pytest.raises(TypeError):
            AgentStrategy()

    def test_context_management(self):
        """Test context management methods."""
        # Use a concrete implementation for testing
        strategy = DesignAgentStrategy()

        # Test set_context
        context = {"workspace": "/test", "key": "value"}
        strategy.set_context(context)
        assert strategy.get_context() == context

        # Test update_context
        strategy.update_context({"key": "new_value", "new_key": "test"})
        expected = {"workspace": "/test", "key": "new_value", "new_key": "test"}
        assert strategy.get_context() == expected

        # Test get_context returns copy
        retrieved = strategy.get_context()
        retrieved["modified"] = True
        assert "modified" not in strategy._context


class TestDesignAgentStrategy:
    """Test the DesignAgentStrategy class."""

    def test_properties(self):
        """Test agent properties."""
        strategy = DesignAgentStrategy()
        assert strategy.name == "design"
        assert "design" in strategy.description.lower()

    def test_build_prompt(self):
        """Test building design prompt."""
        strategy = DesignAgentStrategy()
        task = "Design a user authentication system"
        context = {"workspace": "/project", "existing_patterns": "MVC"}

        # Test without mocking to verify the actual behavior
        prompt = strategy.build_prompt(task, context)
        assert task in prompt
        assert "## Task" in prompt

    def test_validate_input_valid(self):
        """Test input validation with valid input."""
        strategy = DesignAgentStrategy()
        assert strategy.validate_input("Design a feature") is True
        assert strategy.validate_input("A" * 100, {"workspace": "/test"}) is True

    def test_validate_input_invalid(self):
        """Test input validation with invalid input."""
        strategy = DesignAgentStrategy()
        assert strategy.validate_input("") is False
        assert strategy.validate_input("   ") is False

    def test_process_output(self):
        """Test processing design output."""
        strategy = DesignAgentStrategy()
        output = """
        ## Requirements
        - REQ-1: User login
        - REQ-2: Password reset

        ## Components
        - AuthController: Handles authentication
        - UserService: User management

        ## Data Models
        ```python
        class User:
            id: int
            username: str
        ```

        ## API Specifications
        POST /api/login
        GET /api/user/{id}

        ## Implementation Notes
        - Use JWT tokens
        - Hash passwords with bcrypt
        """

        result = strategy.process_output(output)

        assert len(result["requirements"]) == 2
        assert "REQ-1" in result["requirements"][0]
        assert len(result["components"]) == 2
        assert "AuthController" in result["components"]
        assert len(result["data_models"]) == 1
        assert len(result["api_specs"]) == 2
        assert len(result["implementation_notes"]) == 2


class TestTestAgentStrategy:
    """Test the TestAgentStrategy class."""

    def test_properties(self):
        """Test agent properties."""
        strategy = TestGenerationStrategy()
        assert strategy.name == "test"
        assert "test" in strategy.description.lower()

    def test_build_prompt_with_defaults(self):
        """Test building test prompt with default context."""
        strategy = TestGenerationStrategy()
        task = "authentication module"

        prompt = strategy.build_prompt(task)
        assert "Generate tests for:" in prompt
        assert task in prompt

    def test_validate_input_coverage_target(self):
        """Test validation of coverage target."""
        strategy = TestGenerationStrategy()
        assert strategy.validate_input("task", {"coverage_target": 80}) is True
        assert strategy.validate_input("task", {"coverage_target": 0}) is True
        assert strategy.validate_input("task", {"coverage_target": 100}) is True
        assert strategy.validate_input("task", {"coverage_target": -1}) is False
        assert strategy.validate_input("task", {"coverage_target": 101}) is False
        assert strategy.validate_input("task", {"coverage_target": "invalid"}) is False

    def test_validate_input_test_type(self):
        """Test validation of test type."""
        strategy = TestGenerationStrategy()
        assert strategy.validate_input("task", {"test_type": "unit"}) is True
        assert strategy.validate_input("task", {"test_type": "integration"}) is True
        assert strategy.validate_input("task", {"test_type": "all"}) is True
        assert strategy.validate_input("task", {"test_type": "invalid"}) is False

    def test_process_output(self):
        """Test processing test output."""
        strategy = TestGenerationStrategy()
        output = """
        Here are the tests:

        ```python
        import pytest

        def test_login_success():
            assert login("user", "pass") == True

        def test_login_invalid_user():
            with pytest.raises(ValueError):
                login("", "pass")

        def test_edge_case_empty_password():
            assert login("user", "") == False

        def test_integration_database():
            # Integration test
            pass
        ```

        Run with: pytest --cov=module --cov-report=term
        """

        result = strategy.process_output(output)

        assert result["test_count"] == 4
        assert "test_login_success" in result["test_names"]
        assert len(result["test_types"]["unit"]) == 1
        assert len(result["test_types"]["error"]) == 1
        assert len(result["test_types"]["edge_case"]) == 1
        assert len(result["test_types"]["integration"]) == 1
        assert result["assertions_count"] >= 3
        assert len(result["coverage_commands"]) == 1


class TestImplementationAgentStrategy:
    """Test the ImplementationAgentStrategy class."""

    def test_properties(self):
        """Test agent properties."""
        strategy = ImplementationAgentStrategy()
        assert strategy.name == "implementation"
        assert "implement" in strategy.description.lower()

    def test_build_prompt_with_files(self):
        """Test building prompt with design and test files."""
        strategy = ImplementationAgentStrategy()
        task = "Implement authentication"
        context = {"design_file": "docs/auth_design.md", "test_file": "tests/test_auth.py"}

        # Don't mock compose_prompt since build_prompt appends to its result
        prompt = strategy.build_prompt(task, context)

        # Check that task is included
        assert task in prompt
        # Check that file references are added
        assert "docs/auth_design.md" in prompt
        assert "tests/test_auth.py" in prompt

    def test_validate_input_no_tests(self):
        """Test validation with no test file provided."""
        strategy = ImplementationAgentStrategy()

        # Should still pass validation even without test file (just warning)
        assert strategy.validate_input("implement feature", context={}) is True

        # Should pass with test file
        assert (
            strategy.validate_input("implement feature", context={"test_file": "test.py"}) is True
        )

    def test_process_output(self):
        """Test processing implementation output."""
        strategy = ImplementationAgentStrategy()
        output = """
        FILE: src/auth.py
        LANGUAGE: python

        ```python
        from typing import Optional
        import bcrypt

        class AuthService:
            def login(self, username: str, password: str) -> bool:
                # Implementation
                pass

            def logout(self):
                pass
        ```

        FILE: src/utils.py
        OPERATION: REPLACE

        Running tests...
        3 passed, 0 failed

        All tests pass! GREEN phase achieved.
        """

        result = strategy.process_output(output)

        assert "src/auth.py" in result["files_created"]
        assert "src/utils.py" in result["files_modified"]
        assert "AuthService" in result["classes_created"]
        assert "login" in result["methods_implemented"]
        assert "logout" in result["methods_implemented"]
        # Import extraction works on the code blocks
        assert len(result["code_blocks"]) == 1
        assert "import bcrypt" in result["code_blocks"][0]
        assert result["test_results"]["passed"] == 3
        assert result["test_results"]["failed"] == 0
        assert result["test_results"]["status"] == "GREEN"


class TestReviewAgentStrategy:
    """Test the ReviewAgentStrategy class."""

    def test_properties(self):
        """Test agent properties."""
        strategy = ReviewAgentStrategy()
        assert strategy.name == "review"
        assert "code" in strategy.description.lower() or "review" in strategy.description.lower()

    def test_build_prompt_with_rule_and_patch(self):
        """Test building prompt with rule file and patch data."""
        strategy = ReviewAgentStrategy()
        task = "Review authentication changes"
        context = {"rule_file": "rules/security.md", "patch_data": "diff data here"}

        prompt = strategy.build_prompt(task, context)
        assert "Review Task" in prompt
        assert task in prompt
        assert "rules/security.md" in prompt
        assert "diff data here" in prompt

    def test_validate_input(self):
        """Test input validation."""
        strategy = ReviewAgentStrategy()
        assert strategy.validate_input("review code") is True
        assert strategy.validate_input("") is False

    def test_process_output_json(self):
        """Test processing JSON output."""
        strategy = ReviewAgentStrategy()
        output = """
        Found the following violations:

        ```json
        {
            "violations": [
                {
                    "file": "src/auth.py",
                    "line": 42,
                    "severity": "error",
                    "message": "Missing input validation"
                },
                {
                    "file": "src/auth.py",
                    "line": 55,
                    "severity": "warning",
                    "message": "Password not hashed"
                }
            ],
            "summary": {
                "total_violations": 2,
                "files_reviewed": 1,
                "rule_name": "security"
            }
        }
        ```
        """

        result = strategy.process_output(output)

        assert len(result["violations"]) == 2
        assert result["violations"][0]["file"] == "src/auth.py"
        assert result["violations"][0]["line"] == 42
        assert result["severity_counts"]["error"] == 1
        assert result["severity_counts"]["warning"] == 1
        assert result["summary"]["total_violations"] == 2

    def test_process_output_text(self):
        """Test processing text output when JSON not available."""
        strategy = ReviewAgentStrategy()
        output = """
        FILE: src/auth.py
        LINE: 42
        SEVERITY: error
        MESSAGE: Missing input validation

        File: src/utils.py
        Line: 15
        Severity: warning
        Issue: Hardcoded credentials

        Suggestion: Use environment variables for credentials
        """

        result = strategy.process_output(output)

        assert len(result["violations"]) == 2
        assert result["violations"][0]["file"] == "src/auth.py"
        assert result["violations"][0]["line"] == 42
        assert result["severity_counts"]["error"] == 1
        assert result["severity_counts"]["warning"] == 1
        assert len(result["suggestions"]) == 1
