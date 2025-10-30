"""Review Agent for code quality, security, and performance analysis."""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from ..base import AgentCapability, AgentContext, AgentResult, BaseAgent


class ReviewAgent(BaseAgent):
    """Agent specialized in code review without modification permissions."""

    def __init__(self):
        """Initialize Review Agent."""
        super().__init__(
            name="review_agent",
            description="Reviews code for quality, security, and performance (read-only)",
            capabilities=[
                "code_quality",
                "security_analysis",
                "performance_review",
                "best_practices",
            ],
            tools=["read", "grep", "find", "symbols"],  # No write or run!
            max_iterations=30,
            permissions={
                "write": "deny",  # Explicitly deny write
                "run": "deny",  # Explicitly deny execution
            },
        )

        # Register capabilities
        self._register_capabilities()

    def _register_capabilities(self):
        """Register agent capabilities."""
        capabilities = [
            AgentCapability(
                name="code_quality",
                description="Analyze code quality and style",
                required_tools=["read"],
                optional_tools=["grep"],
            ),
            AgentCapability(
                name="security_analysis",
                description="Check for security vulnerabilities",
                required_tools=["read"],
                optional_tools=["grep", "symbols"],
            ),
            AgentCapability(
                name="performance_review",
                description="Review performance and complexity",
                required_tools=["read"],
                optional_tools=["symbols"],
            ),
            AgentCapability(
                name="best_practices",
                description="Check adherence to best practices",
                required_tools=["read"],
                optional_tools=["grep"],
            ),
        ]

        for capability in capabilities:
            self.register_capability(capability)

    def analyze_code_quality(self, code: str, context: AgentContext) -> dict[str, Any]:
        """
        Analyze code quality.

        Args:
            code: Source code to analyze
            context: Execution context

        Returns:
            Quality analysis results
        """
        issues = []
        score = 1.0

        # Check for common quality issues
        lines = code.split("\n")

        # Long lines
        for i, line in enumerate(lines):
            if len(line) > 100:
                issues.append(
                    {
                        "line": i + 1,
                        "severity": "low",
                        "message": f"Line too long ({len(line)} characters)",
                    }
                )
                score -= 0.05

        # Single-letter variable names (except common ones)
        single_letter_pattern = r"\b([a-z])\s*="
        matches = re.finditer(single_letter_pattern, code)
        common_single_letters = {"i", "j", "k", "x", "y", "z"}

        for match in matches:
            var_name = match.group(1)
            if var_name not in common_single_letters:
                issues.append(
                    {"severity": "medium", "message": f"Single-letter variable name: {var_name}"}
                )
                score -= 0.1

        # Missing docstrings
        if "def " in code and '"""' not in code and "'''" not in code:
            issues.append({"severity": "medium", "message": "Functions missing docstrings"})
            score -= 0.1

        # Ensure score is between 0 and 1
        score = max(0.0, min(1.0, score))

        return {"issues": issues, "score": score, "total_issues": len(issues)}

    def check_security_vulnerabilities(self, code: str, context: AgentContext) -> dict[str, Any]:
        """
        Check for security vulnerabilities.

        Args:
            code: Source code to check
            context: Execution context

        Returns:
            Security vulnerability report
        """
        vulnerabilities = []
        severity = "low"

        # Check for dangerous patterns
        dangerous_patterns = {
            r"eval\(": {"severity": "critical", "message": "Use of eval() is dangerous"},
            r"exec\(": {"severity": "critical", "message": "Use of exec() is dangerous"},
            r"os\.system\(": {
                "severity": "high",
                "message": "Command injection risk with os.system()",
            },
            r"subprocess\..*shell=True": {"severity": "high", "message": "Shell injection risk"},
            r"pickle\.loads?\(": {
                "severity": "medium",
                "message": "Pickle deserialization can be unsafe",
            },
            r"input\(.*\).*eval": {"severity": "critical", "message": "eval() on user input"},
            r"__import__\(": {"severity": "medium", "message": "Dynamic imports can be risky"},
        }

        for pattern, vuln_info in dangerous_patterns.items():
            if re.search(pattern, code):
                vulnerabilities.append(
                    {
                        "pattern": pattern,
                        "severity": vuln_info["severity"],
                        "message": vuln_info["message"],
                    }
                )

                # Update overall severity
                severity_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
                if severity_order[vuln_info["severity"]] > severity_order[severity]:
                    severity = vuln_info["severity"]

        return {
            "vulnerabilities": vulnerabilities,
            "severity": severity,
            "total_vulnerabilities": len(vulnerabilities),
        }

    def review_performance(self, code: str, context: AgentContext) -> dict[str, Any]:
        """
        Review code for performance issues.

        Args:
            code: Source code to review
            context: Execution context

        Returns:
            Performance review results
        """
        issues = []
        complexity = "low"

        # Check for nested loops (O(n^2) or worse)
        nested_loop_pattern = r"for\s+\w+.*:\s*\n\s+for\s+\w+"
        if re.search(nested_loop_pattern, code, re.MULTILINE):
            issues.append(
                {
                    "severity": "medium",
                    "message": "Nested loops detected - O(n^2) complexity or worse",
                }
            )
            complexity = "high"

        # Check for inefficient string concatenation in loops
        if re.search(r'for\s+.*:\s*\n\s+.*\+=\s*["\']', code):
            issues.append(
                {
                    "severity": "low",
                    "message": "String concatenation in loop - consider using join()",
                }
            )

        # Check for global variables
        if re.search(r"^global\s+\w+", code, re.MULTILINE):
            issues.append(
                {
                    "severity": "low",
                    "message": "Global variables can impact performance and maintainability",
                }
            )

        return {"issues": issues, "complexity": complexity, "total_issues": len(issues)}

    def check_best_practices(self, code: str, context: AgentContext) -> dict[str, Any]:
        """
        Check adherence to best practices.

        Args:
            code: Source code to check
            context: Execution context

        Returns:
            Best practices violations
        """
        violations = []

        # Check function naming (should be snake_case and descriptive)
        func_pattern = r"def\s+([A-Z]\w+)\("
        camel_case_funcs = re.findall(func_pattern, code)
        for func in camel_case_funcs:
            violations.append(
                {"severity": "low", "message": f"Function '{func}' should use snake_case naming"}
            )

        # Check for single-letter function names
        single_letter_func_pattern = r"def\s+([a-z])\("
        single_letter_funcs = re.findall(single_letter_func_pattern, code)
        for func in single_letter_funcs:
            violations.append(
                {
                    "severity": "medium",
                    "message": f"Function name '{func}' is too short - use descriptive names",
                }
            )

        # Check class naming (should be PascalCase)
        class_pattern = r"class\s+([a-z]\w+)"
        lowercase_classes = re.findall(class_pattern, code)
        for cls in lowercase_classes:
            violations.append(
                {"severity": "low", "message": f"Class '{cls}' should use PascalCase naming"}
            )

        # Check for single-letter class names
        single_letter_class_pattern = r"class\s+([a-z])\s*:"
        single_letter_classes = re.findall(single_letter_class_pattern, code)
        for cls in single_letter_classes:
            violations.append(
                {
                    "severity": "medium",
                    "message": f"Class name '{cls}' is too short - use descriptive names",
                }
            )

        # Check for bare except clauses
        if re.search(r"except\s*:", code):
            violations.append(
                {"severity": "medium", "message": "Bare except clause - specify exception types"}
            )

        # Check for mutable default arguments
        mutable_default_pattern = r"def\s+\w+\([^)]*=\s*(\[\]|\{\})"
        if re.search(mutable_default_pattern, code):
            violations.append(
                {"severity": "high", "message": "Mutable default argument - use None instead"}
            )

        return {"violations": violations, "total_violations": len(violations)}

    def review_file(self, file_path: str, context: AgentContext) -> dict[str, Any]:
        """
        Review a complete file.

        Args:
            file_path: Path to file to review
            context: Execution context

        Returns:
            Complete file review
        """
        try:
            file_path = Path(file_path)

            with file_path.open() as f:
                code = f.read()

            # Run all checks
            quality = self.analyze_code_quality(code, context)
            security = self.check_security_vulnerabilities(code, context)
            performance = self.review_performance(code, context)
            practices = self.check_best_practices(code, context)

            total_issues = (
                quality["total_issues"]
                + security["total_vulnerabilities"]
                + performance["total_issues"]
                + practices["total_violations"]
            )

            return {
                "success": True,
                "file": file_path,
                "quality_score": quality["score"],
                "issues_found": total_issues,
                "quality": quality,
                "security": security,
                "performance": performance,
                "best_practices": practices,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def generate_review_report(
        self, review_data: dict[str, Any], output_path: str, context: AgentContext
    ) -> dict[str, Any]:
        """
        Generate a review report.

        Args:
            review_data: Review results
            output_path: Path to save report
            context: Execution context

        Returns:
            Report generation result
        """
        try:
            lines = [
                "# Code Review Report",
                f"\n**Generated**: {datetime.now().isoformat()}",
                f"\n**Reviewed by**: {self.name}",
                "\n## Summary\n",
                f"- **Files Reviewed**: {len(review_data.get('files_reviewed', []))}",
                f"- **Total Issues**: {review_data.get('total_issues', 0)}",
                f"- **Critical Issues**: {review_data.get('critical_issues', 0)}",
                "\n## Issues by File\n",
            ]

            # Add issues
            for issue in review_data.get("issues", []):
                file_name = issue.get("file", "unknown")
                line = issue.get("line", 0)
                severity = issue.get("severity", "low").upper()
                message = issue.get("message", "No message")

                lines.append(f"### {file_name}:{line} - [{severity}]")
                lines.append(f"{message}\n")

            # Add quality scores
            lines.append("\n## Quality Scores\n")
            for file_name, score in review_data.get("quality_scores", {}).items():
                lines.append(f"- **{file_name}**: {score:.2f}/1.00")

            # Write report
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            output_path = Path(output_path)

            with output_path.open("w") as f:
                f.write("\n".join(lines))

            return {"success": True, "path": output_path}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def check_complexity(self, code: str, context: AgentContext) -> dict[str, Any]:
        """
        Check code complexity.

        Args:
            code: Source code to analyze
            context: Execution context

        Returns:
            Complexity analysis
        """
        # Calculate cyclomatic complexity (simplified)
        complexity = 1  # Base complexity

        # Count decision points
        decision_keywords = ["if", "elif", "else", "for", "while", "and", "or", "except"]

        for keyword in decision_keywords:
            # Count occurrences
            pattern = r"\b" + keyword + r"\b"
            count = len(re.findall(pattern, code))
            complexity += count

        # Complexity score (inverse - lower is better)
        if complexity <= 10:
            complexity_score = 1.0
        elif complexity <= 20:
            complexity_score = 0.7
        else:
            complexity_score = 0.4

        return {
            "cyclomatic_complexity": complexity,
            "complexity_score": complexity_score,
            "rating": "low" if complexity <= 10 else ("medium" if complexity <= 20 else "high"),
        }

    def review_with_rules(
        self, code: str, rules: list[dict[str, Any]], context: AgentContext
    ) -> dict[str, Any]:
        """
        Review code against custom rules.

        Args:
            code: Source code to review
            rules: List of custom rules
            context: Execution context

        Returns:
            Rule violation results
        """
        violations = []

        for rule in rules:
            pattern = rule.get("pattern", "")
            message = rule.get("message", "Rule violation")
            name = rule.get("name", "unknown")

            if re.search(pattern, code):
                violations.append(
                    {"rule": name, "message": message, "severity": rule.get("severity", "medium")}
                )

        return {"violations": violations, "total_violations": len(violations)}

    def suggest_improvements(self, code: str, context: AgentContext) -> dict[str, Any]:
        """
        Suggest code improvements.

        Args:
            code: Source code to analyze
            context: Execution context

        Returns:
            Improvement suggestions
        """
        suggestions = []

        # Suggest list comprehensions (multiple appends to same list)
        if re.search(r"\.append\(.*\).*\n.*\.append\(", code, re.MULTILINE):
            suggestions.append(
                {
                    "type": "optimization",
                    "message": "Consider using list comprehension instead of multiple append calls",
                }
            )
        elif re.search(r"for\s+\w+\s+in\s+.*:", code) and ".append(" in code:
            suggestions.append(
                {
                    "type": "optimization",
                    "message": "Consider using list comprehension instead of append in loop",
                }
            )

        # Suggest context managers
        if "open(" in code and "close()" in code:
            suggestions.append(
                {
                    "type": "best_practice",
                    "message": "Use context manager (with statement) for file handling",
                }
            )

        # Suggest f-strings over format
        if ".format(" in code or "%s" in code:
            suggestions.append(
                {
                    "type": "modernization",
                    "message": "Consider using f-strings for string formatting",
                }
            )

        return {"suggestions": suggestions, "total_suggestions": len(suggestions)}

    def aggregate_scores(self, file_scores: dict[str, float]) -> dict[str, Any]:
        """
        Aggregate review scores across multiple files.

        Args:
            file_scores: Dictionary of file paths to scores

        Returns:
            Aggregated statistics
        """
        if not file_scores:
            return {"average": 0.0, "min": 0.0, "max": 0.0}

        scores = list(file_scores.values())

        return {
            "average": sum(scores) / len(scores),
            "min": min(scores),
            "max": max(scores),
            "total_files": len(scores),
        }

    def execute(self, prompt: str, context: AgentContext) -> AgentResult:
        """
        Execute review agent task using ReAct workflow.

        Args:
            prompt: Review task description
            context: Execution context

        Returns:
            AgentResult with review findings
        """
        # Import executor bridge
        from .executor_bridge import execute_agent_with_react

        # Execute using ReAct workflow with LLM and real tools
        return execute_agent_with_react(agent=self, prompt=prompt, context=context)
