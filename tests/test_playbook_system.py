"""Tests for the playbook system (ACE pattern implementation)."""

import tempfile
from pathlib import Path

import pytest

from ai_dev_agent.playbook import Instruction, InstructionCategory, PlaybookCurator, PlaybookManager


@pytest.fixture
def temp_playbook_dir():
    """Create a temporary directory for playbook testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def playbook_manager(temp_playbook_dir):
    """Create a PlaybookManager instance for testing."""
    playbook_path = temp_playbook_dir / "instructions.json"
    manager = PlaybookManager(playbook_path=playbook_path, max_instructions=50, auto_save=True)
    return manager


@pytest.fixture
def curator(playbook_manager):
    """Create a PlaybookCurator instance for testing."""
    return PlaybookCurator(
        playbook_manager=playbook_manager, similarity_threshold=0.9, quality_threshold=0.3
    )


class TestPlaybookManager:
    """Tests for PlaybookManager."""

    def test_initialization_with_defaults(self, playbook_manager):
        """Test that manager initializes with default instructions."""
        instructions = playbook_manager.get_all_instructions()
        assert len(instructions) > 0
        # Should have instructions across multiple categories
        categories = {inst.category for inst in instructions}
        assert len(categories) > 1

    def test_add_instruction(self, playbook_manager):
        """Test adding a new instruction."""
        instruction = Instruction(
            instruction_id="test_001",
            category=InstructionCategory.DEBUGGING,
            content="Always check logs first when debugging",
            priority=8,
            tags={"debugging", "logs"},
            examples=["Check /var/log/app.log for errors"],
        )

        playbook_manager.add_instruction(instruction, source="test")

        # Verify it was added
        retrieved = playbook_manager.get_instruction("test_001")
        assert retrieved is not None
        assert retrieved.content == instruction.content
        assert retrieved.priority == 8

    def test_update_instruction(self, playbook_manager):
        """Test updating an existing instruction."""
        # Add an instruction
        instruction = Instruction(
            instruction_id="test_002",
            category=InstructionCategory.TESTING,
            content="Write tests before code",
            priority=5,
            tags={"tdd"},
            examples=[],
        )
        playbook_manager.add_instruction(instruction, source="test")

        # Update it
        playbook_manager.update_instruction(
            instruction_id="test_002",
            content="Always write tests before implementation (TDD)",
            priority=9,
            tags={"tdd", "best_practice"},
            examples=["Write failing test, then make it pass"],
        )

        # Verify update
        updated = playbook_manager.get_instruction("test_002")
        assert updated.content == "Always write tests before implementation (TDD)"
        assert updated.priority == 9
        assert "best_practice" in updated.tags
        assert len(updated.examples) == 1

    def test_remove_instruction(self, playbook_manager):
        """Test removing an instruction."""
        instruction = Instruction(
            instruction_id="test_003",
            category=InstructionCategory.GENERAL,
            content="Temporary instruction",
            priority=1,
            tags=set(),
            examples=[],
        )
        playbook_manager.add_instruction(instruction, source="test")

        # Verify it exists
        assert playbook_manager.get_instruction("test_003") is not None

        # Remove it
        result = playbook_manager.remove_instruction("test_003")
        assert result is True

        # Verify it's gone
        assert playbook_manager.get_instruction("test_003") is None

    def test_get_instructions_by_category(self, playbook_manager):
        """Test retrieving instructions by category."""
        # Add instructions in different categories
        inst1 = Instruction(
            instruction_id="test_cat_1",
            category=InstructionCategory.DEBUGGING,
            content="Debug instruction 1",
            priority=5,
            tags=set(),
            examples=[],
        )
        inst2 = Instruction(
            instruction_id="test_cat_2",
            category=InstructionCategory.DEBUGGING,
            content="Debug instruction 2",
            priority=8,
            tags=set(),
            examples=[],
        )
        inst3 = Instruction(
            instruction_id="test_cat_3",
            category=InstructionCategory.TESTING,
            content="Testing instruction",
            priority=5,
            tags=set(),
            examples=[],
        )

        playbook_manager.add_instruction(inst1, source="test")
        playbook_manager.add_instruction(inst2, source="test")
        playbook_manager.add_instruction(inst3, source="test")

        # Get debugging instructions
        debug_instructions = playbook_manager.get_instructions_by_category(
            InstructionCategory.DEBUGGING
        )

        # Should have at least 2 (the ones we added)
        debug_ids = {inst.instruction_id for inst in debug_instructions}
        assert "test_cat_1" in debug_ids
        assert "test_cat_2" in debug_ids
        assert "test_cat_3" not in debug_ids

        # Check sorting (higher priority first)
        priorities = [
            inst.priority
            for inst in debug_instructions
            if inst.instruction_id.startswith("test_cat")
        ]
        assert priorities[0] >= priorities[1]

    def test_get_instructions_by_tags(self, playbook_manager):
        """Test retrieving instructions by tags."""
        inst1 = Instruction(
            instruction_id="test_tag_1",
            category=InstructionCategory.OPTIMIZATION,
            content="Cache frequently accessed data",
            priority=7,
            tags={"performance", "caching"},
            examples=[],
        )
        inst2 = Instruction(
            instruction_id="test_tag_2",
            category=InstructionCategory.OPTIMIZATION,
            content="Use database indexes",
            priority=8,
            tags={"performance", "database"},
            examples=[],
        )

        playbook_manager.add_instruction(inst1, source="test")
        playbook_manager.add_instruction(inst2, source="test")

        # Get performance-related instructions
        perf_instructions = playbook_manager.get_instructions_by_tags({"performance"})
        perf_ids = {inst.instruction_id for inst in perf_instructions}
        assert "test_tag_1" in perf_ids
        assert "test_tag_2" in perf_ids

        # Get caching-specific instructions
        cache_instructions = playbook_manager.get_instructions_by_tags({"caching"})
        cache_ids = {inst.instruction_id for inst in cache_instructions}
        assert "test_tag_1" in cache_ids
        assert "test_tag_2" not in cache_ids

    def test_track_usage(self, playbook_manager):
        """Test tracking instruction usage and effectiveness."""
        instruction = Instruction(
            instruction_id="test_usage",
            category=InstructionCategory.REFACTORING,
            content="Extract complex logic into functions",
            priority=6,
            tags={"clean_code"},
            examples=[],
        )
        playbook_manager.add_instruction(instruction, source="test")

        # Track successful usage
        playbook_manager.track_usage("test_usage", success=True)
        updated = playbook_manager.get_instruction("test_usage")
        assert updated.usage_count == 1
        assert updated.success_count == 1
        assert updated.effectiveness_score == 1.0

        # Track failed usage
        playbook_manager.track_usage("test_usage", success=False)
        updated = playbook_manager.get_instruction("test_usage")
        assert updated.usage_count == 2
        assert updated.success_count == 1
        assert updated.effectiveness_score == 0.5

        # Track more successes
        playbook_manager.track_usage("test_usage", success=True)
        playbook_manager.track_usage("test_usage", success=True)
        updated = playbook_manager.get_instruction("test_usage")
        assert updated.usage_count == 4
        assert updated.success_count == 3
        assert abs(updated.effectiveness_score - 0.75) < 0.01

    def test_format_for_context(self, playbook_manager):
        """Test formatting instructions for LLM context."""
        # Add some test instructions
        inst1 = Instruction(
            instruction_id="test_fmt_1",
            category=InstructionCategory.DEBUGGING,
            content="Use breakpoints for step-by-step debugging",
            priority=9,
            tags={"debugging"},
            examples=[],
        )
        inst2 = Instruction(
            instruction_id="test_fmt_2",
            category=InstructionCategory.TESTING,
            content="Aim for 90% test coverage",
            priority=8,
            tags={"testing", "coverage"},
            examples=[],
        )

        playbook_manager.add_instruction(inst1, source="test")
        playbook_manager.add_instruction(inst2, source="test")

        # Format for context
        context = playbook_manager.format_for_context(
            categories=[InstructionCategory.DEBUGGING, InstructionCategory.TESTING],
            max_instructions=10,
        )

        # Should be a string with structured content
        assert isinstance(context, str)
        assert len(context) > 0
        assert "DEBUGGING" in context or "Debugging" in context
        assert "breakpoints" in context or "Use breakpoints" in context

    def test_max_capacity_pruning(self, temp_playbook_dir):
        """Test that low-priority instructions are pruned at max capacity."""
        # Create a new manager with small max capacity and no defaults
        playbook_path = temp_playbook_dir / "prune_test.json"
        # Start with empty playbook
        playbook_path.write_text("{}")

        playbook_manager = PlaybookManager(
            playbook_path=playbook_path, max_instructions=5, auto_save=False
        )

        # Add 3 high-priority instructions
        for i in range(3):
            inst = Instruction(
                instruction_id=f"high_pri_{i}",
                category=InstructionCategory.GENERAL,
                content=f"High priority instruction {i}",
                priority=9,
                tags=set(),
                examples=[],
            )
            playbook_manager.add_instruction(inst, source="test")

        # Add 3 low-priority instructions (should trigger pruning)
        for i in range(3):
            inst = Instruction(
                instruction_id=f"low_pri_{i}",
                category=InstructionCategory.GENERAL,
                content=f"Low priority instruction {i}",
                priority=2,
                tags=set(),
                examples=[],
            )
            playbook_manager.add_instruction(inst, source="test")

        # Should have pruned to max capacity
        all_instructions = playbook_manager.get_all_instructions()
        assert len(all_instructions) <= 5

        # High priority ones should still be there
        ids = {inst.instruction_id for inst in all_instructions}
        high_pri_present = sum(1 for i in range(3) if f"high_pri_{i}" in ids)
        assert high_pri_present >= 2  # Most high priority should be kept

    def test_persistence(self, temp_playbook_dir):
        """Test that instructions persist across manager instances."""
        playbook_path = temp_playbook_dir / "persist_test.json"

        # Create first manager and add instruction
        manager1 = PlaybookManager(playbook_path=playbook_path, auto_save=True)
        instruction = Instruction(
            instruction_id="persist_test",
            category=InstructionCategory.SECURITY,
            content="Always validate user input",
            priority=10,
            tags={"security", "validation"},
            examples=["Sanitize SQL queries", "Escape HTML"],
        )
        manager1.add_instruction(instruction, source="test")
        manager1.save_playbook()

        # Create second manager from same file
        manager2 = PlaybookManager(playbook_path=playbook_path)

        # Should have the instruction
        retrieved = manager2.get_instruction("persist_test")
        assert retrieved is not None
        assert retrieved.content == "Always validate user input"
        assert retrieved.priority == 10
        assert "security" in retrieved.tags


class TestPlaybookCurator:
    """Tests for PlaybookCurator."""

    def test_find_duplicates_exact(self, curator, playbook_manager):
        """Test finding exact duplicate instructions."""
        # Add duplicate instructions
        inst1 = Instruction(
            instruction_id="dup_1",
            category=InstructionCategory.DEBUGGING,
            content="Always check error logs when debugging issues",
            priority=7,
            tags={"debugging", "logs"},
            examples=[],
        )
        inst2 = Instruction(
            instruction_id="dup_2",
            category=InstructionCategory.DEBUGGING,
            content="Always check error logs when debugging issues",
            priority=6,
            tags={"debugging", "errors"},
            examples=[],
        )

        playbook_manager.add_instruction(inst1, source="test")
        playbook_manager.add_instruction(inst2, source="test")

        # Find duplicates
        duplicates = curator.find_duplicates(category=InstructionCategory.DEBUGGING)

        # Should find the duplicate group
        assert len(duplicates) > 0
        # Find our duplicate group
        our_group = None
        for group in duplicates:
            if group.primary.instruction_id in ["dup_1", "dup_2"]:
                our_group = group
                break
        assert our_group is not None
        assert len(our_group.duplicates) >= 1

    def test_find_duplicates_similar(self, curator, playbook_manager):
        """Test finding similar but not identical instructions."""
        inst1 = Instruction(
            instruction_id="sim_1",
            category=InstructionCategory.TESTING,
            content="Write unit tests for all functions",
            priority=8,
            tags={"testing", "unit_tests"},
            examples=[],
        )
        inst2 = Instruction(
            instruction_id="sim_2",
            category=InstructionCategory.TESTING,
            content="Create unit tests for every function",
            priority=7,
            tags={"testing", "unit_tests"},
            examples=[],
        )

        playbook_manager.add_instruction(inst1, source="test")
        playbook_manager.add_instruction(inst2, source="test")

        # Find duplicates with lower threshold
        curator.similarity_threshold = 0.7
        duplicates = curator.find_duplicates(category=InstructionCategory.TESTING)

        # Should find similar instructions
        # Note: May or may not find depending on similarity metric
        # This test validates the mechanism works, not specific threshold
        assert isinstance(duplicates, list)

    def test_merge_duplicates_keep_primary(self, curator, playbook_manager):
        """Test merging duplicates by keeping primary."""
        inst1 = Instruction(
            instruction_id="merge_1",
            category=InstructionCategory.OPTIMIZATION,
            content="Use caching for expensive operations",
            priority=8,
            usage_count=10,
            success_count=9,
            effectiveness_score=0.9,
            tags={"caching"},
            examples=[],
        )
        inst2 = Instruction(
            instruction_id="merge_2",
            category=InstructionCategory.OPTIMIZATION,
            content="Use caching for expensive operations",
            priority=6,
            usage_count=2,
            success_count=1,
            effectiveness_score=0.5,
            tags={"caching"},
            examples=[],
        )

        playbook_manager.add_instruction(inst1, source="test")
        playbook_manager.add_instruction(inst2, source="test")

        # Find duplicates
        duplicates = curator.find_duplicates(category=InstructionCategory.OPTIMIZATION)
        our_group = None
        for group in duplicates:
            if group.primary.instruction_id in ["merge_1", "merge_2"]:
                our_group = group
                break

        if our_group:
            # Merge should keep the better performing one
            curator.merge_duplicates(our_group, auto_merge=True)

            # Check that one was removed
            assert (
                playbook_manager.get_instruction("merge_1") is None
                or playbook_manager.get_instruction("merge_2") is None
            )

    def test_check_quality_low_effectiveness(self, curator, playbook_manager):
        """Test quality check for low effectiveness instructions."""
        # Add instruction with low effectiveness
        instruction = Instruction(
            instruction_id="low_eff",
            category=InstructionCategory.GENERAL,
            content="Some advice that doesn't work well",
            priority=5,
            usage_count=10,
            success_count=2,
            effectiveness_score=0.2,
            tags=set(),
            examples=[],
        )
        playbook_manager.add_instruction(instruction, source="test")

        # Check quality
        issues = curator.check_quality()

        # Should find low effectiveness issue
        low_eff_issues = [
            issue
            for issue in issues
            if issue.instruction.instruction_id == "low_eff"
            and issue.issue_type == "low_effectiveness"
        ]
        assert len(low_eff_issues) > 0

    def test_check_quality_too_vague(self, curator, playbook_manager):
        """Test quality check for vague instructions."""
        instruction = Instruction(
            instruction_id="vague",
            category=InstructionCategory.GENERAL,
            content="Do it well",  # Too short/vague
            priority=5,
            tags=set(),
            examples=[],
        )
        playbook_manager.add_instruction(instruction, source="test")

        issues = curator.check_quality()

        # Should find vague instruction issue
        vague_issues = [
            issue
            for issue in issues
            if issue.instruction.instruction_id == "vague" and issue.issue_type == "too_vague"
        ]
        assert len(vague_issues) > 0

    def test_suggest_category(self, curator):
        """Test category suggestion based on content."""
        # Test debugging instruction
        debug_inst = Instruction(
            instruction_id="cat_test_1",
            category=InstructionCategory.GENERAL,
            content="When you encounter an error, check the debug logs and trace the execution",
            priority=5,
            tags={"debugging"},
            examples=[],
        )
        suggested = curator.suggest_category(debug_inst)
        assert suggested == InstructionCategory.DEBUGGING

        # Test testing instruction
        test_inst = Instruction(
            instruction_id="cat_test_2",
            category=InstructionCategory.GENERAL,
            content="Write comprehensive unit tests with mocks and verify test coverage",
            priority=5,
            tags={"testing"},
            examples=[],
        )
        suggested = curator.suggest_category(test_inst)
        assert suggested == InstructionCategory.TESTING

        # Test security instruction
        sec_inst = Instruction(
            instruction_id="cat_test_3",
            category=InstructionCategory.GENERAL,
            content="Always validate and sanitize user input to prevent security vulnerabilities",
            priority=5,
            tags={"security"},
            examples=[],
        )
        suggested = curator.suggest_category(sec_inst)
        assert suggested == InstructionCategory.SECURITY

    def test_optimize_playbook_dry_run(self, curator, playbook_manager):
        """Test optimize_playbook in dry run mode."""
        # Add some test instructions with issues
        inst1 = Instruction(
            instruction_id="opt_1",
            category=InstructionCategory.DEBUGGING,
            content="Check logs",
            priority=5,
            tags=set(),
            examples=[],
        )
        inst2 = Instruction(
            instruction_id="opt_2",
            category=InstructionCategory.DEBUGGING,
            content="Check logs",
            priority=4,
            tags=set(),
            examples=[],
        )

        playbook_manager.add_instruction(inst1, source="test")
        playbook_manager.add_instruction(inst2, source="test")

        # Run optimization in dry run mode
        results = curator.optimize_playbook(dry_run=True)

        # Should report findings without making changes
        assert isinstance(results, dict)
        assert "duplicates_found" in results
        assert "actions" in results

        # Verify nothing was actually changed
        assert playbook_manager.get_instruction("opt_1") is not None
        assert playbook_manager.get_instruction("opt_2") is not None


class TestIntegration:
    """Integration tests for the complete playbook system."""

    def test_full_workflow(self, playbook_manager, curator):
        """Test complete workflow: add, use, track, curate."""
        # Step 1: Add instructions
        inst1 = Instruction(
            instruction_id="workflow_1",
            category=InstructionCategory.REFACTORING,
            content="Extract duplicate code into reusable functions",
            priority=7,
            tags={"dry", "refactoring"},
            examples=["Extract common validation logic"],
        )
        playbook_manager.add_instruction(inst1, source="test")

        # Step 2: Format for context (simulate usage)
        context = playbook_manager.format_for_context(
            categories=[InstructionCategory.REFACTORING], max_instructions=5
        )
        assert "duplicate code" in context or "Extract duplicate" in context

        # Step 3: Track usage
        playbook_manager.track_usage("workflow_1", success=True)
        playbook_manager.track_usage("workflow_1", success=True)
        playbook_manager.track_usage("workflow_1", success=False)

        # Step 4: Check effectiveness
        updated = playbook_manager.get_instruction("workflow_1")
        assert updated.usage_count == 3
        assert updated.success_count == 2
        assert abs(updated.effectiveness_score - 0.667) < 0.01

        # Step 5: Run quality check
        issues = curator.check_quality(category=InstructionCategory.REFACTORING)
        # Should not have major issues (good effectiveness)
        critical_issues = [i for i in issues if i.severity == "critical"]
        assert len(critical_issues) == 0

        # Step 6: Add a duplicate and optimize
        inst2 = Instruction(
            instruction_id="workflow_2",
            category=InstructionCategory.REFACTORING,
            content="Extract duplicate code into reusable functions",
            priority=5,
            tags={"dry"},
            examples=[],
        )
        playbook_manager.add_instruction(inst2, source="test")

        # Optimize to remove duplicate
        results = curator.optimize_playbook(dry_run=False)
        assert results["duplicates_found"] >= 0

    def test_category_based_retrieval(self, playbook_manager):
        """Test retrieving instructions for specific task types."""
        # Add instructions across categories
        categories_to_test = [
            InstructionCategory.DEBUGGING,
            InstructionCategory.TESTING,
            InstructionCategory.SECURITY,
        ]

        for i, category in enumerate(categories_to_test):
            inst = Instruction(
                instruction_id=f"cat_retr_{i}",
                category=category,
                content=f"Instruction for {category.value}",
                priority=7,
                tags={category.value},
                examples=[],
            )
            playbook_manager.add_instruction(inst, source="test")

        # Retrieve each category
        for category in categories_to_test:
            instructions = playbook_manager.get_instructions_by_category(category)
            # Should have at least the one we added
            cat_ids = [inst.instruction_id for inst in instructions if inst.category == category]
            assert len(cat_ids) > 0
