"""Test generation agent strategy implementation."""

import logging
import re
from typing import Any, Dict, Optional

from .base import AgentStrategy

logger = logging.getLogger(__name__)


class TestGenerationAgentStrategy(AgentStrategy):
    """Strategy for the test generation agent."""

    @property
    def name(self) -> str:
        """Get the agent name."""
        return "test"

    @property
    def description(self) -> str:
        """Get the agent description."""
        return "Generates comprehensive test suites following TDD principles"

    def build_prompt(self, task: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Build the complete prompt for the test agent.

        Args:
            task: The test generation task
            context: Additional context (e.g., feature, coverage_target, test_type)

        Returns:
            The complete prompt string.
        """
        # Merge provided context with agent context
        full_context = self.get_context()
        if context:
            full_context.update(context)

        # Set defaults for test context
        full_context.setdefault("coverage_target", 90)
        full_context.setdefault("test_type", "all")
        full_context["feature"] = task

        # Compose the full prompt
        components = [
            # System prompts
            ("system/base_context", full_context),
            "system/error_handling",
            # Agent-specific prompt with TDD workflow
            ("agents/test", full_context),
            # Format guidelines for test files
            "formats/whole_file",
        ]

        # Build the composed prompt
        prompt = self.prompt_loader.compose_prompt(components)

        # Add the actual task
        prompt += f"\n\n## Task\n\nGenerate tests for: {task}"

        return prompt

    def validate_input(self, task: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """Validate input before processing.

        Args:
            task: The task to validate
            context: Additional context

        Returns:
            True if input is valid, False otherwise.
        """
        if not task or not task.strip():
            logger.error("Test generation task is empty")
            return False

        # Validate context parameters
        if context:
            if "coverage_target" in context:
                try:
                    coverage = float(context["coverage_target"])
                    if coverage < 0 or coverage > 100:
                        logger.error(f"Invalid coverage target: {coverage}")
                        return False
                except (TypeError, ValueError):
                    logger.error(f"Coverage target must be a number: {context['coverage_target']}")
                    return False

            if "test_type" in context:
                valid_types = ["unit", "integration", "all"]
                if context["test_type"] not in valid_types:
                    logger.error(
                        f"Invalid test type: {context['test_type']}. Must be one of {valid_types}"
                    )
                    return False

        return True

    def process_output(self, output: str) -> Dict[str, Any]:
        """Process the test agent's output.

        Args:
            output: Raw output from the LLM

        Returns:
            Processed output with extracted test information.
        """
        result = {
            "raw_output": output,
            "test_file_content": None,
            "test_count": 0,
            "test_names": [],
            "test_types": {"unit": [], "integration": [], "edge_case": [], "error": []},
            "coverage_commands": [],
            "assertions_count": 0,
        }

        # Extract test file content (between ```python markers)
        python_blocks = re.findall(r"```python\n(.*?)```", output, re.DOTALL)
        if python_blocks:
            # Use the longest block as the main test file
            result["test_file_content"] = max(python_blocks, key=len)

            # Count tests and extract names
            test_pattern = re.compile(r"def (test_\w+)\(")
            matches = test_pattern.findall(result["test_file_content"])
            result["test_count"] = len(matches)
            result["test_names"] = matches

            # Categorize tests by type
            for test_name in matches:
                test_lower = test_name.lower()
                if "integration" in test_lower:
                    result["test_types"]["integration"].append(test_name)
                elif any(
                    word in test_lower
                    for word in ["edge", "boundary", "limit", "empty", "null", "max", "min"]
                ):
                    result["test_types"]["edge_case"].append(test_name)
                elif any(word in test_lower for word in ["error", "exception", "fail", "invalid"]):
                    result["test_types"]["error"].append(test_name)
                else:
                    result["test_types"]["unit"].append(test_name)

            # Count assertions
            assertion_pattern = re.compile(
                r"assert\s+|self\.assert|pytest\.raises|with\s+pytest\.raises"
            )
            result["assertions_count"] = len(assertion_pattern.findall(result["test_file_content"]))

        # Extract coverage commands
        coverage_pattern = re.compile(r"(pytest.*--cov|coverage\s+run|coverage\s+report)")
        result["coverage_commands"] = coverage_pattern.findall(output)

        logger.info(
            f"Processed {result['test_count']} tests with {result['assertions_count']} assertions"
        )
        return result
