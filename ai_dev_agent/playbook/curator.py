"""Playbook curator for semantic deduplication and quality management.

This module implements the curation layer for the ACE (Agentic Context Engineering)
pattern, ensuring high-quality, non-redundant instructions in the playbook.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from .manager import Instruction, InstructionCategory, PlaybookManager

logger = logging.getLogger(__name__)

# Try to import embedding generator from memory system
try:
    from ai_dev_agent.memory.embeddings import EmbeddingGenerator

    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logger.debug("Memory system embeddings not available, using basic similarity")


@dataclass
class DuplicateGroup:
    """Group of similar instructions that may need merging."""

    primary: Instruction
    duplicates: list[tuple[Instruction, float]]  # (instruction, similarity_score)
    merge_recommended: bool
    merge_strategy: str  # "keep_primary", "merge_content", "keep_both"
    reason: str


@dataclass
class QualityIssue:
    """Quality issue found in an instruction."""

    instruction: Instruction
    issue_type: str  # "too_vague", "too_specific", "outdated", "low_effectiveness"
    severity: str  # "critical", "major", "minor"
    recommendation: str
    auto_fixable: bool


class PlaybookCurator:
    """Manages playbook quality and deduplication."""

    def __init__(
        self,
        playbook_manager: PlaybookManager,
        similarity_threshold: float = 0.9,
        quality_threshold: float = 0.3,
    ):
        """Initialize the curator.

        Args:
            playbook_manager: The playbook manager to curate
            similarity_threshold: Threshold for considering instructions similar (0-1)
            quality_threshold: Minimum effectiveness score for quality checks
        """
        self.playbook_manager = playbook_manager
        self.similarity_threshold = similarity_threshold
        self.quality_threshold = quality_threshold

        # Initialize embedding generator if available
        self._embedding_generator = None
        if EMBEDDINGS_AVAILABLE:
            try:
                self._embedding_generator = EmbeddingGenerator(method="auto")
                logger.debug("Initialized curator with semantic similarity")
            except Exception as e:
                logger.warning(f"Failed to initialize embeddings: {e}")

    def find_duplicates(
        self, category: InstructionCategory | None = None, min_similarity: float | None = None
    ) -> list[DuplicateGroup]:
        """Find groups of similar instructions that may be duplicates.

        Args:
            category: Optional category to limit search
            min_similarity: Minimum similarity threshold (uses default if None)

        Returns:
            List of duplicate groups found
        """
        threshold = min_similarity or self.similarity_threshold
        duplicate_groups = []

        # Get instructions to check
        if category:
            instructions = self.playbook_manager.get_instructions_by_category(category)
        else:
            instructions = self.playbook_manager.get_all_instructions()

        if len(instructions) < 2:
            return []

        # Track which instructions have been grouped
        grouped_ids = set()

        # Compare each instruction with others
        for i, inst1 in enumerate(instructions):
            if inst1.instruction_id in grouped_ids:
                continue

            similar_instructions = []

            for _j, inst2 in enumerate(instructions[i + 1 :], start=i + 1):
                if inst2.instruction_id in grouped_ids:
                    continue

                similarity = self._compute_similarity(inst1, inst2)

                if similarity >= threshold:
                    similar_instructions.append((inst2, similarity))
                    grouped_ids.add(inst2.instruction_id)

            if similar_instructions:
                # Determine merge strategy
                merge_strategy, reason = self._determine_merge_strategy(inst1, similar_instructions)

                duplicate_group = DuplicateGroup(
                    primary=inst1,
                    duplicates=similar_instructions,
                    merge_recommended=(merge_strategy != "keep_both"),
                    merge_strategy=merge_strategy,
                    reason=reason,
                )
                duplicate_groups.append(duplicate_group)
                grouped_ids.add(inst1.instruction_id)

        return duplicate_groups

    def merge_duplicates(
        self, duplicate_group: DuplicateGroup, auto_merge: bool = False
    ) -> Instruction | None:
        """Merge a group of duplicate instructions.

        Args:
            duplicate_group: The group to merge
            auto_merge: If False, only merge if strategy is clear

        Returns:
            The merged instruction, or None if not merged
        """
        if not duplicate_group.merge_recommended and not auto_merge:
            logger.info(f"Merge not recommended for {duplicate_group.primary.instruction_id}")
            return None

        primary = duplicate_group.primary
        duplicates = duplicate_group.duplicates

        if duplicate_group.merge_strategy == "keep_primary":
            # Remove duplicates, keep primary
            for dup_inst, _ in duplicates:
                self.playbook_manager.remove_instruction(dup_inst.instruction_id)
            logger.info(
                f"Kept primary {primary.instruction_id}, removed {len(duplicates)} duplicates"
            )
            return primary

        elif duplicate_group.merge_strategy == "merge_content":
            # Merge content from all instructions
            merged = self._merge_instruction_content(primary, duplicates)

            # Update the primary instruction
            self.playbook_manager.update_instruction(
                instruction_id=primary.instruction_id,
                content=merged.content,
                examples=merged.examples,
                tags=merged.tags,
            )

            # Remove duplicates
            for dup_inst, _ in duplicates:
                self.playbook_manager.remove_instruction(dup_inst.instruction_id)

            # Get the updated instruction
            updated = self.playbook_manager.get_instruction(primary.instruction_id)
            logger.info(f"Merged {len(duplicates)} instructions into {primary.instruction_id}")
            return updated

        elif duplicate_group.merge_strategy == "keep_both":
            # No merge needed
            logger.info(f"Keeping both {primary.instruction_id} and duplicates as distinct")
            return None

        return None

    def check_quality(self, category: InstructionCategory | None = None) -> list[QualityIssue]:
        """Check for quality issues in instructions.

        Args:
            category: Optional category to limit checks

        Returns:
            List of quality issues found
        """
        issues = []

        # Get instructions to check
        if category:
            instructions = self.playbook_manager.get_instructions_by_category(category)
        else:
            instructions = self.playbook_manager.get_all_instructions()

        for instruction in instructions:
            # Check 1: Low effectiveness
            if (
                instruction.usage_count >= 5
                and instruction.effectiveness_score < self.quality_threshold
            ):
                issues.append(
                    QualityIssue(
                        instruction=instruction,
                        issue_type="low_effectiveness",
                        severity="major",
                        recommendation=f"Review or remove - only {instruction.effectiveness_score:.1%} effective",
                        auto_fixable=False,
                    )
                )

            # Check 2: Too vague (very short content)
            if len(instruction.content) < 20:
                issues.append(
                    QualityIssue(
                        instruction=instruction,
                        issue_type="too_vague",
                        severity="minor",
                        recommendation="Add more specific guidance or examples",
                        auto_fixable=False,
                    )
                )

            # Check 3: Too specific (overly long, detailed)
            if len(instruction.content) > 500:
                issues.append(
                    QualityIssue(
                        instruction=instruction,
                        issue_type="too_specific",
                        severity="minor",
                        recommendation="Consider splitting into multiple instructions",
                        auto_fixable=False,
                    )
                )

            # Check 4: Never used (but exists for a while)
            if instruction.usage_count == 0 and instruction.created_at:
                # Check if it's been more than 30 days (in real implementation)
                # For now, just flag unused instructions
                issues.append(
                    QualityIssue(
                        instruction=instruction,
                        issue_type="unused",
                        severity="minor",
                        recommendation="Consider testing or removing if not needed",
                        auto_fixable=False,
                    )
                )

            # Check 5: Missing examples for complex instructions
            if len(instruction.content) > 100 and not instruction.examples:
                issues.append(
                    QualityIssue(
                        instruction=instruction,
                        issue_type="missing_examples",
                        severity="minor",
                        recommendation="Add examples to clarify usage",
                        auto_fixable=False,
                    )
                )

        return issues

    def suggest_category(self, instruction: Instruction) -> InstructionCategory:
        """Suggest the best category for an instruction.

        Args:
            instruction: The instruction to categorize

        Returns:
            Suggested category
        """
        content_lower = instruction.content.lower()

        # Simple keyword-based categorization
        category_keywords = {
            InstructionCategory.DEBUGGING: ["debug", "error", "fix", "bug", "trace", "breakpoint"],
            InstructionCategory.TESTING: ["test", "assert", "mock", "coverage", "verify"],
            InstructionCategory.REFACTORING: [
                "refactor",
                "clean",
                "simplify",
                "restructure",
                "organize",
            ],
            InstructionCategory.OPTIMIZATION: [
                "optimize",
                "performance",
                "speed",
                "efficiency",
                "cache",
            ],
            InstructionCategory.SECURITY: ["security", "auth", "encrypt", "validate", "sanitize"],
            InstructionCategory.CODE_REVIEW: [
                "review",
                "feedback",
                "suggest",
                "critique",
                "analyze",
            ],
            InstructionCategory.DOCUMENTATION: [
                "document",
                "comment",
                "explain",
                "describe",
                "readme",
            ],
            InstructionCategory.ERROR_HANDLING: ["error", "exception", "try", "catch", "handle"],
            InstructionCategory.API_DESIGN: ["api", "endpoint", "rest", "graphql", "interface"],
            InstructionCategory.DATABASE: ["database", "sql", "query", "schema", "migration"],
            InstructionCategory.GENERAL: [],  # Default fallback
        }

        # Score each category
        category_scores = {}
        for cat, keywords in category_keywords.items():
            score = sum(1 for kw in keywords if kw in content_lower)
            if instruction.tags:
                score += sum(1 for tag in instruction.tags if tag.lower() in keywords)
            category_scores[cat] = score

        # Return category with highest score, or GENERAL if no matches
        best_category = max(category_scores, key=category_scores.get)
        if category_scores[best_category] == 0:
            return InstructionCategory.GENERAL

        return best_category

    def optimize_playbook(self, dry_run: bool = True) -> dict[str, any]:
        """Perform comprehensive optimization of the playbook.

        Args:
            dry_run: If True, only report what would be done without making changes

        Returns:
            Dictionary with optimization results
        """
        results = {
            "duplicates_found": 0,
            "duplicates_merged": 0,
            "quality_issues": 0,
            "recategorized": 0,
            "removed": 0,
            "actions": [],
        }

        # Step 1: Find and merge duplicates
        duplicate_groups = self.find_duplicates()
        results["duplicates_found"] = len(duplicate_groups)

        if not dry_run:
            for group in duplicate_groups:
                if group.merge_recommended:
                    merged = self.merge_duplicates(group, auto_merge=False)
                    if merged:
                        results["duplicates_merged"] += 1
                        results["actions"].append(
                            f"Merged {len(group.duplicates)} duplicates into {merged.instruction_id}"
                        )
        else:
            for group in duplicate_groups:
                results["actions"].append(
                    f"Would merge {len(group.duplicates)} duplicates with {group.primary.instruction_id}: {group.reason}"
                )

        # Step 2: Check quality issues
        quality_issues = self.check_quality()
        results["quality_issues"] = len(quality_issues)

        for issue in quality_issues:
            if issue.severity == "critical":
                if not dry_run and issue.auto_fixable:
                    # Auto-fix critical issues
                    if issue.issue_type == "low_effectiveness":
                        self.playbook_manager.remove_instruction(issue.instruction.instruction_id)
                        results["removed"] += 1
                        results["actions"].append(
                            f"Removed {issue.instruction.instruction_id}: {issue.recommendation}"
                        )
                else:
                    results["actions"].append(
                        f"Issue: {issue.instruction.instruction_id} - {issue.recommendation}"
                    )

        # Step 3: Suggest recategorization
        all_instructions = self.playbook_manager.get_all_instructions()
        for instruction in all_instructions:
            suggested_category = self.suggest_category(instruction)
            if suggested_category != instruction.category:
                if not dry_run:
                    # Directly modify the category (update_instruction doesn't support category parameter)
                    with self.playbook_manager._lock:
                        if instruction.instruction_id in self.playbook_manager._instructions:
                            self.playbook_manager._instructions[
                                instruction.instruction_id
                            ].category = suggested_category
                            self.playbook_manager._instructions[
                                instruction.instruction_id
                            ].updated_at = datetime.now().isoformat()
                            if self.playbook_manager.auto_save:
                                self.playbook_manager.save_playbook()
                    results["recategorized"] += 1
                results["actions"].append(
                    f"Recategorized {instruction.instruction_id}: {instruction.category} -> {suggested_category}"
                )

        return results

    def _compute_similarity(self, inst1: Instruction, inst2: Instruction) -> float:
        """Compute similarity between two instructions.

        Args:
            inst1: First instruction
            inst2: Second instruction

        Returns:
            Similarity score (0-1)
        """
        # If categories are different, they're likely not duplicates
        if inst1.category != inst2.category:
            return 0.0

        # Use semantic similarity if available
        if self._embedding_generator:
            try:
                emb1 = self._embedding_generator.generate_embedding(inst1.content)
                emb2 = self._embedding_generator.generate_embedding(inst2.content)
                similarity = self._embedding_generator.compute_similarity(
                    emb1, emb2.reshape(1, -1)
                )[0]
                return float(similarity)
            except Exception as e:
                logger.debug(f"Failed to compute semantic similarity: {e}")

        # Fallback: Simple token-based similarity
        return self._simple_similarity(inst1.content, inst2.content)

    def _simple_similarity(self, text1: str, text2: str) -> float:
        """Simple token-based similarity (Jaccard similarity).

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0-1)
        """
        # Tokenize
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())

        if not tokens1 or not tokens2:
            return 0.0

        # Jaccard similarity
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)

        return intersection / union if union > 0 else 0.0

    def _determine_merge_strategy(
        self, primary: Instruction, similar_instructions: list[tuple[Instruction, float]]
    ) -> tuple[str, str]:
        """Determine the best merge strategy for a group of similar instructions.

        Args:
            primary: Primary instruction
            similar_instructions: List of (instruction, similarity) tuples

        Returns:
            Tuple of (strategy, reason)
        """
        # If all similarities are very high (>0.95), they're near-duplicates
        avg_similarity = sum(sim for _, sim in similar_instructions) / len(similar_instructions)

        if avg_similarity > 0.95:
            # Keep the one with highest effectiveness
            best_effectiveness = primary.effectiveness_score
            for inst, _ in similar_instructions:
                if inst.effectiveness_score > best_effectiveness:
                    best_effectiveness = inst.effectiveness_score

            if best_effectiveness == primary.effectiveness_score:
                return (
                    "keep_primary",
                    f"Primary has best effectiveness ({primary.effectiveness_score:.1%})",
                )
            else:
                return "keep_primary", "Near-duplicates found, keeping most effective"

        elif avg_similarity > 0.85:
            # Similar but with some differences - merge content
            return (
                "merge_content",
                f"Similar instructions ({avg_similarity:.1%} similar), merging to combine insights",
            )

        else:
            # Somewhat similar but distinct enough to keep separate
            return (
                "keep_both",
                f"Instructions are related but distinct ({avg_similarity:.1%} similar)",
            )

    def _merge_instruction_content(
        self, primary: Instruction, duplicates: list[tuple[Instruction, float]]
    ) -> Instruction:
        """Merge content from duplicate instructions into primary.

        Args:
            primary: Primary instruction to merge into
            duplicates: List of (instruction, similarity) tuples to merge

        Returns:
            Updated instruction (not yet persisted)
        """
        # Combine content (using primary as base)
        merged_content = primary.content

        # Collect unique examples
        merged_examples = list(primary.examples) if primary.examples else []
        for dup_inst, _ in duplicates:
            if dup_inst.examples:
                for example in dup_inst.examples:
                    if example not in merged_examples:
                        merged_examples.append(example)

        # Combine tags
        merged_tags = set(primary.tags) if primary.tags else set()
        for dup_inst, _ in duplicates:
            if dup_inst.tags:
                merged_tags.update(dup_inst.tags)

        # Combine effectiveness scores (weighted average by usage)
        total_usage = primary.usage_count
        weighted_score = primary.effectiveness_score * primary.usage_count

        for dup_inst, _ in duplicates:
            total_usage += dup_inst.usage_count
            weighted_score += dup_inst.effectiveness_score * dup_inst.usage_count

        merged_effectiveness = (
            weighted_score / total_usage if total_usage > 0 else primary.effectiveness_score
        )

        # Create merged instruction (copy of primary with merged data)
        merged = Instruction(
            instruction_id=primary.instruction_id,
            category=primary.category,
            content=merged_content,
            priority=primary.priority,
            usage_count=total_usage,
            success_count=primary.success_count + sum(dup.success_count for dup, _ in duplicates),
            effectiveness_score=merged_effectiveness,
            tags=merged_tags,
            examples=merged_examples,
            created_at=primary.created_at,
            updated_at=primary.updated_at,
            metadata=primary.metadata,
        )

        return merged
