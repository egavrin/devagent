"""Proposal Generator for Dynamic Instruction System.

Generates instruction update proposals using LLM analysis of query patterns.
Uses the configured model from settings, not a hardcoded model.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from ai_dev_agent.core.utils.config import Settings

from .manager import InstructionUpdate, UpdateConfidence, UpdateSource, UpdateType

if TYPE_CHECKING:
    from .pattern_tracker import PatternSignal

logger = logging.getLogger(__name__)


class ProposalGenerator:
    """Generates instruction update proposals from query patterns using LLM analysis."""

    def __init__(self, settings: Settings | None = None):
        """Initialize the proposal generator.

        Args:
            settings: Settings instance (uses configured model)
        """
        self.settings = settings or Settings()

    def generate_proposals(
        self, patterns: list[PatternSignal], max_proposals: int = 3
    ) -> list[InstructionUpdate]:
        """Generate instruction update proposals from detected patterns.

        Args:
            patterns: List of detected patterns with confidence scores
            max_proposals: Maximum number of proposals to generate

        Returns:
            List of instruction update proposals
        """
        if not patterns:
            logger.debug("No patterns provided for proposal generation")
            return []

        # Build prompt for LLM
        prompt = self._build_proposal_prompt(patterns, max_proposals)

        # Call LLM to generate proposals
        try:
            proposals_data = self._call_llm_for_proposals(prompt)

            # Parse and validate proposals
            proposals = self._parse_proposals(proposals_data, patterns)

            logger.info(f"Generated {len(proposals)} instruction proposals")
            return proposals

        except Exception as e:
            logger.warning(f"Failed to generate proposals: {e}")
            return []

    def _build_proposal_prompt(self, patterns: list[PatternSignal], max_proposals: int) -> str:
        """Build prompt for LLM proposal generation.

        Args:
            patterns: Detected patterns
            max_proposals: Maximum proposals to request

        Returns:
            Prompt string
        """
        # Format patterns for the prompt
        pattern_descriptions = []

        success_patterns = [
            p for p in patterns if p.pattern_type in ("tool_sequence", "success_strategy")
        ]
        failure_patterns = [p for p in patterns if p.pattern_type == "failure_pattern"]
        recovery_patterns = [p for p in patterns if p.pattern_type == "error_recovery"]

        if success_patterns:
            pattern_descriptions.append("**Success Patterns** (high success rate):")
            for pattern in success_patterns:
                pattern_descriptions.append(
                    f"  - {pattern.description}: {pattern.query_count} queries, "
                    f"{pattern.success_rate*100:.0f}% success (confidence: {pattern.confidence:.2f})"
                )

        if failure_patterns:
            pattern_descriptions.append("\n**Failure Patterns** (low success rate):")
            for pattern in failure_patterns:
                pattern_descriptions.append(
                    f"  - {pattern.description}: {pattern.query_count} queries, "
                    f"{pattern.success_rate*100:.0f}% success (confidence: {pattern.confidence:.2f})"
                )

        if recovery_patterns:
            pattern_descriptions.append("\n**Error Recovery Patterns**:")
            for pattern in recovery_patterns:
                pattern_descriptions.append(
                    f"  - {pattern.description}: {pattern.query_count} cases, "
                    f"{pattern.success_rate*100:.0f}% recovered (confidence: {pattern.confidence:.2f})"
                )

        patterns_text = "\n".join(pattern_descriptions)

        prompt = f"""You are an expert at analyzing software development patterns and proposing improvements.

I have detected the following patterns from recent query executions:

{patterns_text}

Based on these patterns, propose {max_proposals} instruction updates that would:
1. Reinforce successful patterns (high success rate strategies)
2. Prevent common failures (low success rate patterns)
3. Improve efficiency and reliability

For each proposal, provide:
- **type**: One of: ADD, MODIFY, REMOVE, WEIGHT_INCREASE, WEIGHT_DECREASE
- **content**: The instruction text (be specific and actionable)
- **reasoning**: Why this instruction would help (reference the patterns)
- **confidence**: Your confidence score (0.0-1.0) based on:
  - Pattern sample size (more samples = higher confidence)
  - Pattern consistency (higher success rate = higher confidence)
  - Clarity of benefit (clearer benefit = higher confidence)

Output ONLY valid JSON in this exact format:
{{
  "proposals": [
    {{
      "type": "ADD",
      "content": "Before writing to a file, always read it first to understand the current content and context",
      "reasoning": "Tool sequence 'find → read → write' shows 93% success rate across 15 queries, while direct writes have only 20% success",
      "confidence": 0.85
    }}
  ]
}}

IMPORTANT:
- Output ONLY the JSON, no other text
- Ensure the JSON is valid and parseable
- Focus on actionable, specific instructions
- Confidence should reflect the strength of evidence from patterns
"""

        return prompt

    def _call_llm_for_proposals(self, prompt: str) -> dict[str, Any]:
        """Call the configured LLM to generate proposals.

        Args:
            prompt: Proposal generation prompt

        Returns:
            Parsed JSON response from LLM
        """
        # Import LLM provider
        from ai_dev_agent.providers.llm import create_client
        from ai_dev_agent.providers.llm.base import Message

        # Create LLM client using configured settings
        try:
            client = create_client(
                provider=self.settings.provider,
                model=self.settings.model,
                api_key=self.settings.api_key,
                base_url=self.settings.base_url,
                provider_config=self.settings.provider_config,
            )

            # Call LLM with the prompt
            messages = [Message(role="user", content=prompt)]

            response = client.complete(
                messages=messages,
                temperature=0.3,  # Lower temperature for more consistent output
                max_tokens=2000,
            )

            # Parse JSON response
            # Try to extract JSON from response (handle cases where LLM adds extra text)
            response_clean = response.strip()

            # Remove markdown code fences if present
            if response_clean.startswith("```json"):
                response_clean = response_clean[7:]
            if response_clean.startswith("```"):
                response_clean = response_clean[3:]
            if response_clean.endswith("```"):
                response_clean = response_clean[:-3]

            response_clean = response_clean.strip()

            # Parse JSON
            try:
                data = json.loads(response_clean)
                return data
            except json.JSONDecodeError as e:
                logger.warning(f"LLM response is not valid JSON: {e}")
                logger.debug(f"Response was: {response_clean[:200]}")
                # Try to extract JSON from within the response
                import re

                json_match = re.search(r"\{.*\}", response_clean, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group(0))
                    return data
                raise

        except Exception as e:
            logger.error(f"Failed to call LLM for proposals: {e}")
            raise

    def _parse_proposals(
        self, proposals_data: dict[str, Any], patterns: list[PatternSignal]
    ) -> list[InstructionUpdate]:
        """Parse and validate LLM-generated proposals.

        Args:
            proposals_data: Raw proposal data from LLM
            patterns: Original patterns (for validation)

        Returns:
            List of validated InstructionUpdate objects
        """
        proposals: list[InstructionUpdate] = []

        try:
            proposals_raw = proposals_data.get("proposals", [])

            for proposal_data in proposals_raw:
                # Extract fields
                update_type_str = proposal_data.get("type", "ADD")
                content = proposal_data.get("content", "")
                reasoning = proposal_data.get("reasoning", "")
                confidence = float(proposal_data.get("confidence", 0.5))

                # Validate
                if not content or not reasoning:
                    logger.warning("Skipping proposal with missing content or reasoning")
                    continue

                if confidence < 0.0 or confidence > 1.0:
                    logger.warning(f"Invalid confidence {confidence}, clamping to [0, 1]")
                    confidence = max(0.0, min(1.0, confidence))

                # Map confidence to confidence level
                if confidence > 0.9:
                    conf_level = UpdateConfidence.VERY_HIGH
                elif confidence > 0.7:
                    conf_level = UpdateConfidence.HIGH
                elif confidence > 0.5:
                    conf_level = UpdateConfidence.MEDIUM
                elif confidence > 0.3:
                    conf_level = UpdateConfidence.LOW
                else:
                    conf_level = UpdateConfidence.VERY_LOW

                # Parse update type
                try:
                    update_type = UpdateType(update_type_str.lower())
                except ValueError:
                    logger.warning(f"Unknown update type '{update_type_str}', defaulting to ADD")
                    update_type = UpdateType.ADD

                # Create InstructionUpdate
                update = InstructionUpdate(
                    instruction_id="",  # Will be assigned when applied
                    update_type=update_type,
                    update_source=UpdateSource.AUTOMATIC,
                    new_content=content,
                    confidence=confidence,
                    confidence_level=conf_level,
                    reasoning=reasoning,
                )

                proposals.append(update)

        except Exception as e:
            logger.error(f"Failed to parse proposals: {e}")

        return proposals

    def validate_proposal_quality(self, proposal: InstructionUpdate) -> bool:
        """Validate that a proposal meets quality standards.

        Args:
            proposal: Instruction update proposal

        Returns:
            True if proposal is high quality
        """
        # Check content length (should be substantive)
        if len(proposal.new_content or "") < 20:
            logger.debug(f"Proposal content too short: {len(proposal.new_content or '')} chars")
            return False

        # Check reasoning exists
        if not proposal.reasoning or len(proposal.reasoning) < 10:
            logger.debug("Proposal reasoning missing or too short")
            return False

        # Check confidence is reasonable
        if proposal.confidence < 0.3:
            logger.debug(f"Proposal confidence too low: {proposal.confidence}")
            return False

        return True
