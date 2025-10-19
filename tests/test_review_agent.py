"""Tests for Review Agent."""
import pytest
from unittest.mock import Mock, patch, mock_open
import tempfile
import os

from ai_dev_agent.agents.specialized.review_agent import ReviewAgent
from ai_dev_agent.agents.base import AgentContext, AgentResult


class TestReviewAgent:
    """Test Review Agent functionality."""

    def test_review_agent_initialization(self):
        """Test creating a review agent."""
        agent = ReviewAgent()

        assert agent.name == "review_agent"
        assert "read" in agent.tools
        assert "grep" in agent.tools
        assert "find" in agent.tools
        # Review agent should NOT have write or run permissions
        assert "write" not in agent.tools
        assert "run" not in agent.tools
        assert agent.max_iterations == 30

    def test_review_agent_capabilities(self):
        """Test review agent has correct capabilities."""
        agent = ReviewAgent()

        assert "code_quality" in agent.capabilities
        assert "security_analysis" in agent.capabilities
        assert "performance_review" in agent.capabilities
        assert "best_practices" in agent.capabilities

    def test_analyze_code_quality(self):
        """Test analyzing code quality."""
        agent = ReviewAgent()
        context = AgentContext(session_id="test-quality")

        code = """
def process_data(data):
    x = data
    y = x + 1
    z = y * 2
    return z
"""

        result = agent.analyze_code_quality(code, context)

        assert "issues" in result
        assert "score" in result
        assert result["score"] >= 0 and result["score"] <= 1.0

    def test_check_security_vulnerabilities(self):
        """Test checking for security vulnerabilities."""
        agent = ReviewAgent()
        context = AgentContext(session_id="test-security")

        vulnerable_code = """
import os
user_input = input("Enter command: ")
os.system(user_input)  # Security vulnerability!
"""

        result = agent.check_security_vulnerabilities(vulnerable_code, context)

        assert "vulnerabilities" in result
        assert len(result["vulnerabilities"]) > 0
        assert result["severity"] in ["low", "medium", "high", "critical"]

    def test_review_performance(self):
        """Test performance review."""
        agent = ReviewAgent()
        context = AgentContext(session_id="test-perf")

        inefficient_code = """
def find_duplicates(items):
    duplicates = []
    for i in range(len(items)):
        for j in range(len(items)):
            if i != j and items[i] == items[j]:
                duplicates.append(items[i])
    return duplicates
"""

        result = agent.review_performance(inefficient_code, context)

        assert "issues" in result
        assert "complexity" in result
        assert any("O(n^2)" in str(issue) or "nested loop" in str(issue).lower()
                   for issue in result["issues"])

    def test_check_best_practices(self):
        """Test checking adherence to best practices."""
        agent = ReviewAgent()
        context = AgentContext(session_id="test-practices")

        bad_code = """
def f(x, y):
    return x + y

class c:
    def m(self):
        pass
"""

        result = agent.check_best_practices(bad_code, context)

        assert "violations" in result
        assert len(result["violations"]) > 0
        # Should flag single-letter names
        assert any("name" in str(v).lower() for v in result["violations"])

    def test_review_file(self):
        """Test reviewing a complete file."""
        agent = ReviewAgent()
        context = AgentContext(session_id="test-file")

        code = """
def calculate_total(items):
    total = 0
    for item in items:
        total += item.price
    return total
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            file_path = f.name

        try:
            result = agent.review_file(file_path, context)

            assert result["success"] is True
            assert "quality_score" in result
            assert "issues_found" in result

        finally:
            os.unlink(file_path)

    def test_generate_review_report(self):
        """Test generating a review report."""
        agent = ReviewAgent()
        context = AgentContext(session_id="test-report")

        review_data = {
            "files_reviewed": ["module1.py", "module2.py"],
            "total_issues": 5,
            "critical_issues": 1,
            "quality_scores": {"module1.py": 0.8, "module2.py": 0.9},
            "issues": [
                {"file": "module1.py", "line": 10, "severity": "high", "message": "SQL injection risk"},
                {"file": "module2.py", "line": 25, "severity": "low", "message": "Unused import"}
            ]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = os.path.join(tmpdir, "review_report.md")

            result = agent.generate_review_report(review_data, report_path, context)

            assert result["success"] is True
            assert os.path.exists(report_path)

            with open(report_path, 'r') as f:
                content = f.read()
                assert "Code Review Report" in content
                assert "SQL injection" in content
                assert "module1.py" in content

    def test_check_complexity(self):
        """Test checking code complexity."""
        agent = ReviewAgent()
        context = AgentContext(session_id="test-complexity")

        complex_code = """
def complex_function(a, b, c, d):
    if a > 0:
        if b > 0:
            if c > 0:
                if d > 0:
                    return a + b + c + d
                else:
                    return 0
            else:
                return 0
        else:
            return 0
    else:
        return 0
"""

        result = agent.check_complexity(complex_code, context)

        assert "complexity_score" in result
        assert "cyclomatic_complexity" in result
        assert result["cyclomatic_complexity"] > 5  # High complexity

    def test_review_with_rules(self):
        """Test reviewing code against custom rules."""
        agent = ReviewAgent()
        context = AgentContext(session_id="test-rules")

        code = "print('Hello')"
        rules = [
            {"name": "no_print_statements", "pattern": r"print\(", "message": "Use logging instead of print"}
        ]

        result = agent.review_with_rules(code, rules, context)

        assert "violations" in result
        assert len(result["violations"]) > 0
        assert any("logging" in v["message"] for v in result["violations"])

    def test_suggest_improvements(self):
        """Test suggesting code improvements."""
        agent = ReviewAgent()
        context = AgentContext(session_id="test-improvements")

        code = """
def get_items():
    items = []
    items.append(1)
    items.append(2)
    items.append(3)
    return items
"""

        suggestions = agent.suggest_improvements(code, context)

        assert "suggestions" in suggestions
        assert len(suggestions["suggestions"]) > 0

    def test_review_agent_read_only(self):
        """Test that review agent cannot modify files."""
        agent = ReviewAgent()

        # Ensure write is not in tools
        assert not agent.has_tool("write")
        assert not agent.has_tool("run")

        # Only read-only tools
        assert agent.has_tool("read")
        assert agent.has_tool("grep")

    @pytest.mark.llm
    def test_review_agent_execute(self):
        """Test full review agent execution."""
        agent = ReviewAgent()
        context = AgentContext(session_id="test-execute")

        code = """
def vulnerable_function(user_input):
    eval(user_input)  # Security issue

def slow_function(items):
    result = []
    for i in items:
        for j in items:
            result.append(i * j)
    return result
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "code.py")
            with open(file_path, 'w') as f:
                f.write(code)

            prompt = f"Review the code at {file_path} for security and performance issues"

            result = agent.execute(prompt, context)

            assert result.success is True
            assert "issues_found" in result.metadata
            assert result.metadata["issues_found"] > 0

    def test_aggregate_review_scores(self):
        """Test aggregating review scores across multiple files."""
        agent = ReviewAgent()

        file_scores = {
            "module1.py": 0.9,
            "module2.py": 0.8,
            "module3.py": 0.7
        }

        aggregate = agent.aggregate_scores(file_scores)

        assert aggregate["average"] == pytest.approx(0.8, 0.01)
        assert aggregate["min"] == 0.7
        assert aggregate["max"] == 0.9