"""Comprehensive tests for the ProposalGenerator module."""

import json
from unittest.mock import MagicMock, patch

import pytest

from ai_dev_agent.core.utils.config import Settings
from ai_dev_agent.dynamic_instructions.manager import (
    InstructionUpdate,
    UpdateConfidence,
    UpdateSource,
    UpdateType,
)
from ai_dev_agent.dynamic_instructions.pattern_tracker import PatternSignal
from ai_dev_agent.dynamic_instructions.proposal_generator import ProposalGenerator


class TestProposalGenerator:
    """Test suite for ProposalGenerator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.settings = Settings()
        self.generator = ProposalGenerator(settings=self.settings)

    def test_init_default_settings(self):
        """Test initialization with default settings."""
        generator = ProposalGenerator()
        assert generator.settings is not None
        assert isinstance(generator.settings, Settings)

    def test_init_custom_settings(self):
        """Test initialization with custom settings."""
        custom_settings = Settings()
        generator = ProposalGenerator(settings=custom_settings)
        assert generator.settings is custom_settings

    def test_generate_proposals_empty_patterns(self):
        """Test generate_proposals with empty patterns list."""
        proposals = self.generator.generate_proposals([])
        assert proposals == []

    def test_generate_proposals_success(self):
        """Test successful proposal generation."""
        # Create test patterns
        patterns = [
            PatternSignal(
                pattern_type="tool_sequence",
                description="find → read → write sequence",
                query_count=15,
                success_rate=0.93,
                confidence=0.85,
                examples=["session1", "session2"],
            ),
            PatternSignal(
                pattern_type="failure_pattern",
                description="direct write without read",
                query_count=10,
                success_rate=0.20,
                confidence=0.75,
                examples=["session3"],
            ),
        ]

        # Mock LLM response
        mock_response = {
            "proposals": [
                {
                    "type": "ADD",
                    "content": "Always read a file before writing to understand its current content",
                    "reasoning": "Tool sequence 'find → read → write' shows 93% success rate",
                    "confidence": 0.85,
                },
                {
                    "type": "MODIFY",
                    "content": "Check file existence before attempting direct writes",
                    "reasoning": "Direct writes have only 20% success rate",
                    "confidence": 0.70,
                },
            ]
        }

        with patch.object(self.generator, "_call_llm_for_proposals", return_value=mock_response):
            proposals = self.generator.generate_proposals(patterns, max_proposals=2)

        assert len(proposals) == 2
        assert proposals[0].update_type == UpdateType.ADD
        assert (
            proposals[0].new_content
            == "Always read a file before writing to understand its current content"
        )
        assert proposals[0].confidence == 0.85
        assert proposals[0].confidence_level == UpdateConfidence.HIGH

        assert proposals[1].update_type == UpdateType.MODIFY
        assert proposals[1].confidence == 0.70
        assert proposals[1].confidence_level == UpdateConfidence.MEDIUM

    def test_generate_proposals_exception_handling(self):
        """Test exception handling during proposal generation."""
        patterns = [
            PatternSignal(
                pattern_type="tool_sequence",
                description="test pattern",
                query_count=5,
                success_rate=0.8,
                confidence=0.6,
                examples=[],
            )
        ]

        with patch.object(
            self.generator, "_call_llm_for_proposals", side_effect=Exception("LLM error")
        ):
            proposals = self.generator.generate_proposals(patterns)
            assert proposals == []

    def test_build_proposal_prompt_all_pattern_types(self):
        """Test prompt building with all pattern types."""
        patterns = [
            PatternSignal(
                pattern_type="tool_sequence",
                description="successful sequence",
                query_count=20,
                success_rate=0.95,
                confidence=0.90,
                examples=[],
            ),
            PatternSignal(
                pattern_type="failure_pattern",
                description="common failure",
                query_count=15,
                success_rate=0.10,
                confidence=0.80,
                examples=[],
            ),
            PatternSignal(
                pattern_type="error_recovery",
                description="recovery strategy",
                query_count=8,
                success_rate=0.75,
                confidence=0.65,
                examples=[],
            ),
            PatternSignal(
                pattern_type="success_strategy",
                description="effective approach",
                query_count=12,
                success_rate=0.88,
                confidence=0.77,
                examples=[],
            ),
        ]

        prompt = self.generator._build_proposal_prompt(patterns, max_proposals=3)

        # Check that all pattern types are included
        assert "Success Patterns" in prompt
        assert "Failure Patterns" in prompt
        assert "Error Recovery Patterns" in prompt
        assert "successful sequence" in prompt
        assert "common failure" in prompt
        assert "recovery strategy" in prompt
        assert "effective approach" in prompt
        assert "95% success" in prompt
        assert "10% success" in prompt
        assert "75% recovered" in prompt

    def test_build_proposal_prompt_only_success_patterns(self):
        """Test prompt building with only success patterns."""
        patterns = [
            PatternSignal(
                pattern_type="tool_sequence",
                description="good sequence",
                query_count=10,
                success_rate=0.9,
                confidence=0.8,
                examples=[],
            )
        ]

        prompt = self.generator._build_proposal_prompt(patterns, max_proposals=1)

        assert "Success Patterns" in prompt
        assert "Failure Patterns" not in prompt
        assert "Error Recovery" not in prompt
        assert "good sequence" in prompt

    @patch("ai_dev_agent.providers.llm.create_client")
    def test_call_llm_for_proposals_success(self, mock_create_client):
        """Test successful LLM call."""
        # Mock LLM client and response
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client

        expected_response = {
            "proposals": [
                {
                    "type": "ADD",
                    "content": "Test instruction",
                    "reasoning": "Test reasoning",
                    "confidence": 0.75,
                }
            ]
        }

        mock_client.complete.return_value = json.dumps(expected_response)

        result = self.generator._call_llm_for_proposals("test prompt")

        assert result == expected_response
        mock_create_client.assert_called_once()
        mock_client.complete.assert_called_once()

    @patch("ai_dev_agent.providers.llm.create_client")
    def test_call_llm_for_proposals_json_with_markdown(self, mock_create_client):
        """Test LLM response with markdown code fences."""
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client

        expected_response = {
            "proposals": [
                {"type": "ADD", "content": "Test", "reasoning": "Test", "confidence": 0.5}
            ]
        }

        # Response wrapped in markdown code fences
        mock_client.complete.return_value = f"```json\n{json.dumps(expected_response)}\n```"

        result = self.generator._call_llm_for_proposals("test prompt")

        assert result == expected_response

    @patch("ai_dev_agent.providers.llm.create_client")
    def test_call_llm_for_proposals_json_extraction(self, mock_create_client):
        """Test JSON extraction from mixed text response."""
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client

        expected_response = {
            "proposals": [
                {"type": "ADD", "content": "Test", "reasoning": "Test", "confidence": 0.5}
            ]
        }

        # Response with extra text
        mock_client.complete.return_value = (
            f"Here is the JSON:\n{json.dumps(expected_response)}\nEnd of response"
        )

        result = self.generator._call_llm_for_proposals("test prompt")

        assert result == expected_response

    @patch("ai_dev_agent.providers.llm.create_client")
    def test_call_llm_for_proposals_invalid_json(self, mock_create_client):
        """Test handling of invalid JSON response."""
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client

        # Invalid JSON response
        mock_client.complete.return_value = "This is not valid JSON"

        with pytest.raises(json.JSONDecodeError):
            self.generator._call_llm_for_proposals("test prompt")

    @patch("ai_dev_agent.providers.llm.create_client")
    def test_call_llm_for_proposals_exception(self, mock_create_client):
        """Test LLM call exception handling."""
        mock_create_client.side_effect = Exception("Connection error")

        with pytest.raises(Exception) as exc_info:
            self.generator._call_llm_for_proposals("test prompt")

        assert "Connection error" in str(exc_info.value)

    def test_parse_proposals_valid_data(self):
        """Test parsing valid proposal data."""
        patterns = []  # Not used in this test

        proposals_data = {
            "proposals": [
                {
                    "type": "ADD",
                    "content": "First instruction",
                    "reasoning": "Good reason",
                    "confidence": 0.95,
                },
                {
                    "type": "MODIFY",
                    "content": "Modified instruction",
                    "reasoning": "Another reason",
                    "confidence": 0.60,
                },
                {
                    "type": "REMOVE",
                    "content": "Remove this",
                    "reasoning": "Not needed",
                    "confidence": 0.45,
                },
            ]
        }

        proposals = self.generator._parse_proposals(proposals_data, patterns)

        assert len(proposals) == 3

        # Check first proposal
        assert proposals[0].update_type == UpdateType.ADD
        assert proposals[0].new_content == "First instruction"
        assert proposals[0].confidence == 0.95
        assert proposals[0].confidence_level == UpdateConfidence.VERY_HIGH
        assert proposals[0].update_source == UpdateSource.AUTOMATIC

        # Check second proposal
        assert proposals[1].update_type == UpdateType.MODIFY
        assert proposals[1].confidence == 0.60
        assert proposals[1].confidence_level == UpdateConfidence.MEDIUM

        # Check third proposal
        assert proposals[2].update_type == UpdateType.REMOVE
        assert proposals[2].confidence == 0.45
        assert proposals[2].confidence_level == UpdateConfidence.LOW

    def test_parse_proposals_confidence_levels(self):
        """Test confidence level mapping."""
        patterns = []

        test_cases = [
            (0.95, UpdateConfidence.VERY_HIGH),
            (0.85, UpdateConfidence.HIGH),
            (0.65, UpdateConfidence.MEDIUM),
            (0.35, UpdateConfidence.LOW),
            (0.15, UpdateConfidence.VERY_LOW),
        ]

        for confidence, expected_level in test_cases:
            proposals_data = {
                "proposals": [
                    {
                        "type": "ADD",
                        "content": "Test",
                        "reasoning": "Test",
                        "confidence": confidence,
                    }
                ]
            }

            proposals = self.generator._parse_proposals(proposals_data, patterns)
            assert proposals[0].confidence_level == expected_level

    def test_parse_proposals_missing_fields(self):
        """Test parsing with missing required fields."""
        patterns = []

        proposals_data = {
            "proposals": [
                {
                    "type": "ADD",
                    # Missing content
                    "reasoning": "Test",
                    "confidence": 0.5,
                },
                {
                    "type": "ADD",
                    "content": "Test",
                    # Missing reasoning
                    "confidence": 0.5,
                },
                {
                    "type": "ADD",
                    "content": "Valid content",
                    "reasoning": "Valid reasoning",
                    "confidence": 0.7,
                },
            ]
        }

        proposals = self.generator._parse_proposals(proposals_data, patterns)

        # Only the valid proposal should be returned
        assert len(proposals) == 1
        assert proposals[0].new_content == "Valid content"

    def test_parse_proposals_invalid_confidence(self):
        """Test parsing with invalid confidence values."""
        patterns = []

        proposals_data = {
            "proposals": [
                {
                    "type": "ADD",
                    "content": "Test high",
                    "reasoning": "Test",
                    "confidence": 1.5,  # Too high
                },
                {
                    "type": "ADD",
                    "content": "Test low",
                    "reasoning": "Test",
                    "confidence": -0.5,  # Too low
                },
            ]
        }

        proposals = self.generator._parse_proposals(proposals_data, patterns)

        assert len(proposals) == 2
        assert proposals[0].confidence == 1.0  # Clamped to max
        assert proposals[1].confidence == 0.0  # Clamped to min

    def test_parse_proposals_invalid_update_type(self):
        """Test parsing with invalid update type."""
        patterns = []

        proposals_data = {
            "proposals": [
                {"type": "INVALID_TYPE", "content": "Test", "reasoning": "Test", "confidence": 0.5}
            ]
        }

        proposals = self.generator._parse_proposals(proposals_data, patterns)

        assert len(proposals) == 1
        assert proposals[0].update_type == UpdateType.ADD  # Defaults to ADD

    def test_parse_proposals_exception_handling(self):
        """Test exception handling during proposal parsing."""
        patterns = []

        # Invalid data structure
        proposals_data = "not a dict"

        proposals = self.generator._parse_proposals(proposals_data, patterns)
        assert proposals == []

    def test_validate_proposal_quality_valid(self):
        """Test validation of high-quality proposal."""
        proposal = InstructionUpdate(
            instruction_id="test",
            update_type=UpdateType.ADD,
            update_source=UpdateSource.AUTOMATIC,
            new_content="This is a substantive instruction with clear guidance",
            confidence=0.75,
            confidence_level=UpdateConfidence.HIGH,
            reasoning="This proposal is well-reasoned and based on strong evidence",
        )

        assert self.generator.validate_proposal_quality(proposal) is True

    def test_validate_proposal_quality_short_content(self):
        """Test validation rejects short content."""
        proposal = InstructionUpdate(
            instruction_id="test",
            update_type=UpdateType.ADD,
            update_source=UpdateSource.AUTOMATIC,
            new_content="Too short",  # Less than 20 chars
            confidence=0.75,
            confidence_level=UpdateConfidence.HIGH,
            reasoning="Good reasoning here",
        )

        assert self.generator.validate_proposal_quality(proposal) is False

    def test_validate_proposal_quality_missing_reasoning(self):
        """Test validation rejects missing reasoning."""
        proposal = InstructionUpdate(
            instruction_id="test",
            update_type=UpdateType.ADD,
            update_source=UpdateSource.AUTOMATIC,
            new_content="This is a good instruction with enough content",
            confidence=0.75,
            confidence_level=UpdateConfidence.HIGH,
            reasoning="",  # Empty reasoning
        )

        assert self.generator.validate_proposal_quality(proposal) is False

    def test_validate_proposal_quality_short_reasoning(self):
        """Test validation rejects short reasoning."""
        proposal = InstructionUpdate(
            instruction_id="test",
            update_type=UpdateType.ADD,
            update_source=UpdateSource.AUTOMATIC,
            new_content="This is a good instruction with enough content",
            confidence=0.75,
            confidence_level=UpdateConfidence.HIGH,
            reasoning="Too short",  # Less than 10 chars
        )

        assert self.generator.validate_proposal_quality(proposal) is False

    def test_validate_proposal_quality_low_confidence(self):
        """Test validation rejects low confidence."""
        proposal = InstructionUpdate(
            instruction_id="test",
            update_type=UpdateType.ADD,
            update_source=UpdateSource.AUTOMATIC,
            new_content="This is a substantive instruction with clear guidance",
            confidence=0.25,  # Below 0.3 threshold
            confidence_level=UpdateConfidence.VERY_LOW,
            reasoning="This proposal has good reasoning but low confidence",
        )

        assert self.generator.validate_proposal_quality(proposal) is False

    def test_validate_proposal_quality_none_content(self):
        """Test validation handles None content."""
        proposal = InstructionUpdate(
            instruction_id="test",
            update_type=UpdateType.REMOVE,
            update_source=UpdateSource.AUTOMATIC,
            new_content=None,  # None content for REMOVE type
            confidence=0.75,
            confidence_level=UpdateConfidence.HIGH,
            reasoning="Good reasoning for removal",
        )

        assert self.generator.validate_proposal_quality(proposal) is False

    def test_validate_proposal_quality_none_reasoning(self):
        """Test validation handles None reasoning."""
        proposal = InstructionUpdate(
            instruction_id="test",
            update_type=UpdateType.ADD,
            update_source=UpdateSource.AUTOMATIC,
            new_content="This is a substantive instruction with clear guidance",
            confidence=0.75,
            confidence_level=UpdateConfidence.HIGH,
            reasoning=None,  # None reasoning
        )

        assert self.generator.validate_proposal_quality(proposal) is False

    def test_generate_proposals_integration(self):
        """Integration test for full proposal generation flow."""
        # Create realistic patterns
        patterns = [
            PatternSignal(
                pattern_type="tool_sequence",
                description="read → analyze → write workflow",
                query_count=25,
                success_rate=0.92,
                confidence=0.88,
                examples=["session4", "session5", "session6"],
            ),
            PatternSignal(
                pattern_type="failure_pattern",
                description="skipping validation steps",
                query_count=18,
                success_rate=0.28,
                confidence=0.82,
                examples=["session7", "session8"],
            ),
            PatternSignal(
                pattern_type="error_recovery",
                description="retry with exponential backoff",
                query_count=12,
                success_rate=0.83,
                confidence=0.75,
                examples=["session9"],
            ),
        ]

        # Mock a realistic LLM response
        mock_llm_response = {
            "proposals": [
                {
                    "type": "ADD",
                    "content": "Always validate data before processing to prevent downstream errors",
                    "reasoning": "Skipping validation shows only 28% success rate in 18 queries, indicating validation is critical",
                    "confidence": 0.82,
                },
                {
                    "type": "WEIGHT_INCREASE",
                    "content": "Prioritize read-analyze-write workflow for complex operations",
                    "reasoning": "This workflow demonstrates 92% success rate across 25 queries with high confidence",
                    "confidence": 0.88,
                },
                {
                    "type": "ADD",
                    "content": "Implement exponential backoff for retry operations on transient failures",
                    "reasoning": "Recovery with exponential backoff shows 83% success in error recovery scenarios",
                    "confidence": 0.75,
                },
            ]
        }

        with patch.object(
            self.generator, "_call_llm_for_proposals", return_value=mock_llm_response
        ):
            proposals = self.generator.generate_proposals(patterns, max_proposals=3)

        # Verify all proposals are properly generated
        assert len(proposals) == 3

        # Check each proposal is well-formed
        for proposal in proposals:
            assert proposal.update_type in [UpdateType.ADD, UpdateType.WEIGHT_INCREASE]
            assert proposal.update_source == UpdateSource.AUTOMATIC
            assert len(proposal.new_content) > 20
            assert len(proposal.reasoning) > 10
            assert 0.0 <= proposal.confidence <= 1.0
            assert proposal.confidence_level in UpdateConfidence

        # Verify specific proposals
        assert proposals[0].confidence == 0.82
        assert proposals[1].update_type == UpdateType.WEIGHT_INCREASE
        assert "exponential backoff" in proposals[2].new_content.lower()
