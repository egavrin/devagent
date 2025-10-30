"""Implementation agent strategy implementation."""

import logging
import re
from typing import Any, Dict, Optional

from .base import AgentStrategy

logger = logging.getLogger(__name__)


class ImplementationAgentStrategy(AgentStrategy):
    """Strategy for the code implementation agent."""

    @property
    def name(self) -> str:
        """Get the agent name."""
        return "implementation"

    @property
    def description(self) -> str:
        """Get the agent description."""
        return "Implements code following TDD principles and making tests pass"

    def build_prompt(self, task: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Build the complete prompt for the implementation agent.

        Args:
            task: The implementation task
            context: Additional context (e.g., design_file, test_file, workspace)

        Returns:
            The complete prompt string.
        """
        # Merge provided context with agent context
        full_context = self.get_context()
        if context:
            full_context.update(context)

        # Add task to context
        full_context["task"] = task

        # Compose the full prompt
        components = [
            # System prompts
            ("system/base_context", full_context),
            "system/error_handling",
            "system/react_loop",
            # Agent-specific prompt with TDD workflow
            ("agents/implementation", full_context),
            # Format guidelines - implementation may use different formats
            "formats/whole_file",
            "formats/edit_block",
        ]

        # Build the composed prompt
        prompt = self.prompt_loader.compose_prompt(components)

        # Add task section
        prompt += f"\n\n## Task\n\n{task}"

        # Add design file reference if provided
        if "design_file" in full_context:
            prompt += f"\n\n## Design Document\n\nRefer to: {full_context['design_file']}"

        # Add test file reference if provided
        if "test_file" in full_context:
            prompt += f"\n\n## Test File\n\nMake tests pass in: {full_context['test_file']}"

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
            logger.error("Implementation task is empty")
            return False

        # Warn if no test file is provided (violates TDD)
        if context and "test_file" not in context:
            logger.warning("No test file provided - TDD workflow requires tests first")

        return True

    def process_output(self, output: str) -> Dict[str, Any]:
        """Process the implementation agent's output.

        Args:
            output: Raw output from the LLM

        Returns:
            Processed output with extracted implementation details.
        """
        result = {
            "raw_output": output,
            "files_created": [],
            "files_modified": [],
            "code_blocks": [],
            "test_results": {"passed": 0, "failed": 0, "errors": 0, "status": "unknown"},
            "methods_implemented": [],
            "classes_created": [],
            "imports_added": [],
        }

        # Extract file operations
        file_create_pattern = re.compile(r"FILE:\s*([^\n]+)\n.*?LANGUAGE:\s*([^\n]+)", re.DOTALL)
        for match in file_create_pattern.finditer(output):
            result["files_created"].append(match.group(1).strip())

        # Extract edit operations (must not have LANGUAGE line between FILE and OPERATION)
        edit_pattern = re.compile(
            r"FILE:\s*([^\n]+)\n(?!.*?LANGUAGE).*?OPERATION:\s*(REPLACE|INSERT|DELETE)", re.DOTALL
        )
        for match in edit_pattern.finditer(output):
            file_path = match.group(1).strip()
            if (
                file_path not in result["files_modified"]
                and file_path not in result["files_created"]
            ):
                result["files_modified"].append(file_path)

        # Extract code blocks
        code_blocks = re.findall(r"```(?:python|py)?\n(.*?)```", output, re.DOTALL)
        result["code_blocks"] = code_blocks

        # Analyze code blocks for implementation details
        for code in code_blocks:
            # Extract class definitions
            class_pattern = re.compile(r"class\s+(\w+)\s*[:\(]")
            result["classes_created"].extend(class_pattern.findall(code))

            # Extract method/function definitions
            method_pattern = re.compile(r"def\s+(\w+)\s*\(")
            result["methods_implemented"].extend(method_pattern.findall(code))

            # Extract imports
            import_pattern = re.compile(r"^(?:from\s+[\w.]+\s+)?import\s+.+", re.MULTILINE)
            result["imports_added"].extend(import_pattern.findall(code))

        # Extract test results if present
        test_output_pattern = re.compile(
            r"(\d+)\s+passed.*?(?:(\d+)\s+failed)?.*?(?:(\d+)\s+error)?"
        )
        test_match = test_output_pattern.search(output)
        if test_match:
            result["test_results"]["passed"] = int(test_match.group(1) or 0)
            result["test_results"]["failed"] = int(test_match.group(2) or 0)
            result["test_results"]["errors"] = int(test_match.group(3) or 0)

            # Determine overall status
            if result["test_results"]["failed"] == 0 and result["test_results"]["errors"] == 0:
                result["test_results"]["status"] = "GREEN"
            else:
                result["test_results"]["status"] = "RED"

        # Check for explicit test status mentions
        if "all tests pass" in output.lower() or "green phase" in output.lower():
            result["test_results"]["status"] = "GREEN"
        elif "tests fail" in output.lower() or "red phase" in output.lower():
            result["test_results"]["status"] = "RED"

        logger.info(
            f"Processed implementation: {len(result['files_created'])} files created, "
            f"{len(result['files_modified'])} files modified, "
            f"{len(result['methods_implemented'])} methods implemented"
        )

        return result
