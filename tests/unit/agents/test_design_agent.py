"""Tests for Design Agent."""

import os
import tempfile
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from ai_dev_agent.agents.base import AgentContext
from ai_dev_agent.agents.specialized.design_agent import DesignAgent


class TestDesignAgent:
    """Test Design Agent functionality."""

    def test_design_agent_initialization(self):
        """Test creating a design agent."""
        agent = DesignAgent()

        assert agent.name == "design_agent"
        assert "read" in agent.tools
        assert "grep" in agent.tools
        assert "find" in agent.tools
        assert "symbols" in agent.tools
        assert "write" in agent.tools  # For creating design docs
        assert agent.max_iterations == 30

    def test_design_agent_capabilities(self):
        """Test design agent has correct capabilities."""
        agent = DesignAgent()

        assert "technical_design" in agent.capabilities
        assert "reference_analysis" in agent.capabilities
        assert "architecture_design" in agent.capabilities
        assert "pattern_extraction" in agent.capabilities

    def test_analyze_requirements(self):
        """Test analyzing requirements for a feature."""
        agent = DesignAgent()
        context = AgentContext(session_id="test-design")

        requirements = """
        Build a REST API for user management with:
        - User registration and login
        - JWT authentication
        - CRUD operations for user profiles
        - Password reset functionality
        """

        result = agent.analyze_requirements(requirements, context)

        assert result["status"] == "analyzed"
        assert "features" in result
        assert "user_registration" in result["features"]
        assert "authentication" in result["features"]
        assert "crud_operations" in result["features"]

    def test_extract_patterns_from_reference(self):
        """Test extracting patterns from reference implementations."""
        agent = DesignAgent()
        context = AgentContext(session_id="test-pattern")

        # Mock file content from reference
        reference_code = """
        class UserController:
            def __init__(self, user_service):
                self.user_service = user_service

            def register(self, request):
                data = validate_request(request)
                user = self.user_service.create_user(data)
                return success_response(user)
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            ref_path = Path(tmpdir) / "user_controller.py"
            ref_path.write_text(reference_code, encoding="utf-8")
            patterns = agent.extract_patterns(str(ref_path), context)

        assert "patterns" in patterns
        assert any("controller" in p.lower() for p in patterns["patterns"])
        assert any(
            "dependency" in p.lower() or "injection" in p.lower() for p in patterns["patterns"]
        )
        assert "architecture" in patterns

    def test_create_design_document(self):
        """Test creating a design document."""
        agent = DesignAgent()
        context = AgentContext(session_id="test-doc")

        design_data = {
            "feature": "User Authentication",
            "requirements": ["JWT tokens", "Refresh tokens", "Password hashing"],
            "architecture": {
                "components": ["AuthController", "AuthService", "TokenManager"],
                "flow": "Request → Controller → Service → Database",
            },
            "patterns": ["MVC", "Service Layer", "Repository Pattern"],
            "references": ["/ref/auth_system.py"],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            doc_path = str(Path(tmpdir) / "auth_design.md")

            result = agent.create_design_document(design_data, doc_path, context)

            assert result["success"] is True
            assert Path(doc_path).exists()

            # Check document content
            with Path(doc_path).open() as f:
                content = f.read()
                assert "# Design: User Authentication" in content
                assert "## Requirements" in content
                assert "JWT tokens" in content
                assert "## Architecture" in content
                assert "## Patterns" in content
                assert "MVC" in content

    def test_reference_analysis(self):
        """Test analyzing reference implementations."""
        agent = DesignAgent()
        context = AgentContext(session_id="test-ref")

        reference_paths = [
            "/Users/eg/Documents/aider",
            "/Users/eg/Documents/cline-1",
            "/Users/eg/Documents/opencode",
        ]

        # Mock the file system operations
        with patch("os.path.exists", return_value=True):
            with patch("os.listdir", return_value=["agent.py", "core.py"]):
                analysis = agent.analyze_references(
                    feature="multi-agent system", reference_paths=reference_paths, context=context
                )

        assert "references" in analysis
        assert len(analysis["references"]) > 0
        assert "patterns" in analysis
        assert "recommendations" in analysis

    def test_compatibility_assessment(self):
        """Test assessing compatibility impact."""
        agent = DesignAgent()
        context = AgentContext(session_id="test-compat")

        proposed_changes = {
            "new_modules": ["agents/orchestrator.py"],
            "modified_modules": ["agents/registry.py", "cli/runtime/main.py"],
            "new_dependencies": ["asyncio", "threading"],
            "api_changes": [{"module": "registry", "change": "added create_agent method"}],
        }

        assessment = agent.assess_compatibility(proposed_changes, context)

        assert "risk_level" in assessment
        assert assessment["risk_level"] in ["low", "medium", "high"]
        assert "impacts" in assessment
        assert "recommendations" in assessment

    @pytest.mark.llm
    def test_design_agent_execute(self):
        """Test full design agent execution."""
        agent = DesignAgent()
        context = AgentContext(session_id="test-execute")

        prompt = """
        Design a multi-agent coordination system with:
        - Design, Test, Implementation, and Review agents
        - Orchestrator for coordination
        - Event bus for communication
        Reference: /Users/eg/Documents/cline-1 and /Users/eg/Documents/opencode
        """

        # Mock file operations
        with patch("os.path.exists", return_value=True):
            with patch("os.makedirs"):
                with patch("builtins.open", mock_open()):
                    result = agent.execute(prompt, context)

        assert result.success is True
        assert "design_document" in result.metadata
        assert "patterns_found" in result.metadata
        assert "compatibility_assessment" in result.metadata

    def test_design_validation(self):
        """Test design validation against best practices."""
        agent = DesignAgent()

        design = {
            "architecture": {
                "layers": ["presentation", "business", "data"],
                "components": ["UserController", "UserService", "UserRepository"],
            },
            "patterns": ["MVC", "Repository", "Service Layer"],
            "principles": ["SOLID", "DRY", "KISS"],
        }

        validation = agent.validate_design(design)

        assert validation["valid"] is True
        assert "suggestions" in validation
        assert validation["score"] >= 0.7  # Good design score

    @pytest.mark.llm
    def test_design_agent_with_errors(self):
        """Test design agent error handling."""
        agent = DesignAgent()
        context = AgentContext(session_id="test-error")

        # Test with invalid reference path
        with patch("os.path.exists", return_value=False):
            result = agent.execute("Design feature with reference /nonexistent/path", context)

        assert result.success is False
        assert "error" in result.error.lower() or "not found" in result.error.lower()
