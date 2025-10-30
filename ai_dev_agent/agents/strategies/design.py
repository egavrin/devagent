"""Design agent strategy implementation."""

import json
import logging
from typing import Any, Dict, Optional

from .base import AgentStrategy

logger = logging.getLogger(__name__)


class DesignAgentStrategy(AgentStrategy):
    """Strategy for the technical design agent."""

    @property
    def name(self) -> str:
        """Get the agent name."""
        return "design"

    @property
    def description(self) -> str:
        """Get the agent description."""
        return "Creates comprehensive technical designs and architecture documents"

    def build_prompt(self, task: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Build the complete prompt for the design agent.

        Args:
            task: The design task to perform
            context: Additional context (e.g., workspace, existing_patterns)

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
            # Agent-specific prompt
            ("agents/design", full_context),
            # Format guidelines for markdown output
            "formats/whole_file",
        ]

        # Build the composed prompt
        prompt = self.prompt_loader.compose_prompt(components)

        # Add the actual task
        prompt += f"\n\n## Task\n\n{task}"

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
            logger.error("Design task is empty")
            return False

        if len(task) < 10:
            logger.warning("Design task seems too short")

        # Validate context if provided
        if context:
            if "workspace" in context and not context["workspace"]:
                logger.warning("Workspace context is empty")

        return True

    def process_output(self, output: str) -> Dict[str, Any]:
        """Process the design agent's output.

        Args:
            output: Raw output from the LLM

        Returns:
            Processed output with extracted design elements.
        """
        result = {
            "raw_output": output,
            "design_sections": {},
            "requirements": [],
            "components": [],
            "data_models": [],
            "api_specs": [],
            "implementation_notes": [],
        }

        # Parse markdown sections
        current_section = None
        current_content = []

        for line in output.split("\n"):
            # Check for markdown headers (handle indented sections too)
            stripped = line.strip()
            if stripped.startswith("##"):
                # Save previous section
                if current_section:
                    section_content = "\n".join(current_content).strip()
                    result["design_sections"][current_section] = section_content
                    self._extract_section_data(current_section, section_content, result)

                # Start new section
                current_section = stripped.lstrip("#").strip()
                current_content = []
            elif current_section:
                current_content.append(line)

        # Save last section
        if current_section:
            section_content = "\n".join(current_content).strip()
            result["design_sections"][current_section] = section_content
            self._extract_section_data(current_section, section_content, result)

        logger.info(f"Processed design with {len(result['design_sections'])} sections")
        return result

    def _extract_section_data(self, section_name: str, content: str, result: Dict[str, Any]):
        """Extract structured data from design sections.

        Args:
            section_name: Name of the section
            content: Section content
            result: Result dictionary to update
        """
        section_lower = section_name.lower()

        if "requirement" in section_lower:
            # Extract requirements (REQ-1, REQ-2, etc.)
            for line in content.split("\n"):
                if line.strip().startswith("REQ-") or line.strip().startswith("- REQ-"):
                    result["requirements"].append(line.strip())

        elif "component" in section_lower or "architecture" in section_lower:
            # Extract component names
            for line in content.split("\n"):
                if line.strip().startswith("- ") and len(line.strip()) > 2:
                    component = line.strip()[2:].split(":")[0].strip()
                    if component:
                        result["components"].append(component)

        elif "data model" in section_lower or "model" in section_lower:
            # Extract data model definitions
            in_code_block = False
            model_lines = []
            for line in content.split("\n"):
                if line.strip().startswith("```"):
                    if in_code_block and model_lines:
                        result["data_models"].append("\n".join(model_lines))
                        model_lines = []
                    in_code_block = not in_code_block
                elif in_code_block:
                    model_lines.append(line)

        elif "api" in section_lower or "endpoint" in section_lower:
            # Extract API endpoints
            for line in content.split("\n"):
                if any(
                    method in line.upper() for method in ["GET", "POST", "PUT", "DELETE", "PATCH"]
                ):
                    result["api_specs"].append(line.strip())

        elif "implementation" in section_lower or "note" in section_lower:
            # Extract implementation notes
            for line in content.split("\n"):
                if line.strip().startswith("- ") and len(line.strip()) > 2:
                    result["implementation_notes"].append(line.strip()[2:])
