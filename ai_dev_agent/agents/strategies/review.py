"""Code review agent strategy implementation."""

import json
import logging
import re
from typing import Any, Dict, Optional

from .base import AgentStrategy

logger = logging.getLogger(__name__)


class ReviewAgentStrategy(AgentStrategy):
    """Strategy for the code review agent."""

    @property
    def name(self) -> str:
        """Get the agent name."""
        return "review"

    @property
    def description(self) -> str:
        """Get the agent description."""
        return "Analyzes code patches against coding rules and reports violations"

    def build_prompt(self, task: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Build the complete prompt for the review agent.

        Args:
            task: The review task (e.g., file path or patch to review)
            context: Additional context (e.g., rule_file, patch_data)

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
            # Agent-specific prompt
            ("agents/review", full_context),
        ]

        # Build the composed prompt
        prompt = self.prompt_loader.compose_prompt(components)

        # Add the actual review task
        prompt += f"\n\n## Review Task\n\n{task}"

        # Add rule file if provided
        if "rule_file" in full_context:
            prompt += f"\n\n## Rule File\n\n{full_context['rule_file']}"

        # Add patch data if provided
        if "patch_data" in full_context:
            prompt += f"\n\n## Patch Dataset\n\n{full_context['patch_data']}"

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
            logger.error("Review task is empty")
            return False

        # Validate context if provided
        if context:
            if "patch_data" in context and not context["patch_data"]:
                logger.warning("Patch data is empty")

            if "rule_file" in context and not context["rule_file"]:
                logger.warning("Rule file is empty")

        return True

    def process_output(self, output: str) -> Dict[str, Any]:
        """Process the review agent's output.

        Args:
            output: Raw output from the LLM

        Returns:
            Processed output with extracted violations.
        """
        result = {
            "raw_output": output,
            "violations": [],
            "summary": {"total_violations": 0, "files_reviewed": 0, "rule_name": ""},
            "suggestions": [],
            "severity_counts": {"error": 0, "warning": 0, "info": 0},
        }

        # Try to extract JSON output
        json_pattern = re.compile(r"```json\n(.*?)```", re.DOTALL)
        json_matches = json_pattern.findall(output)

        if json_matches:
            try:
                # Parse the JSON output
                json_data = json.loads(json_matches[0])

                # Extract violations
                if "violations" in json_data:
                    result["violations"] = json_data["violations"]

                    # Count severities
                    for violation in result["violations"]:
                        severity = violation.get("severity", "info").lower()
                        if severity in result["severity_counts"]:
                            result["severity_counts"][severity] += 1

                # Extract summary
                if "summary" in json_data:
                    result["summary"] = json_data["summary"]
                else:
                    # Calculate summary from violations
                    result["summary"]["total_violations"] = len(result["violations"])
                    if result["violations"]:
                        files = {v.get("file", "") for v in result["violations"]}
                        result["summary"]["files_reviewed"] = len(files)

                logger.info(f"Parsed JSON output with {len(result['violations'])} violations")

            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON output: {e}")
                # Fall back to text parsing
                self._parse_text_output(output, result)
        else:
            # Parse text output for violations
            self._parse_text_output(output, result)

        # Extract suggestions (non-JSON format)
        suggestion_pattern = re.compile(
            r"(?:Suggestion|Recommendation|Fix):\s*(.+?)(?:\n|$)", re.IGNORECASE
        )
        result["suggestions"] = suggestion_pattern.findall(output)

        return result

    def _parse_text_output(self, output: str, result: Dict[str, Any]):
        """Parse text output for violations when JSON is not available.

        Args:
            output: Raw text output
            result: Result dictionary to update
        """
        # Look for violation patterns in text
        violation_pattern = re.compile(
            r"(?:File|FILE):\s*([^\n]+)\n.*?"
            r"(?:Line|LINE):\s*(\d+).*?\n.*?"
            r"(?:Severity|SEVERITY):\s*(\w+).*?\n.*?"
            r"(?:Message|MESSAGE|Issue|ISSUE):\s*([^\n]+)",
            re.DOTALL | re.IGNORECASE,
        )

        for match in violation_pattern.finditer(output):
            violation = {
                "file": match.group(1).strip(),
                "line": int(match.group(2)),
                "severity": match.group(3).lower(),
                "message": match.group(4).strip(),
            }
            result["violations"].append(violation)

            # Update severity counts
            if violation["severity"] in result["severity_counts"]:
                result["severity_counts"][violation["severity"]] += 1

        # Update summary
        result["summary"]["total_violations"] = len(result["violations"])
        if result["violations"]:
            files = {v["file"] for v in result["violations"]}
            result["summary"]["files_reviewed"] = len(files)

        logger.info(f"Parsed text output with {len(result['violations'])} violations")
