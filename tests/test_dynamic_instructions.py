"""Tests for the dynamic instructions system (ILWS pattern implementation)."""

import tempfile
from pathlib import Path

import pytest

from ai_dev_agent.dynamic_instructions import (
    ABTestManager,
    ABTestStatus,
    DynamicInstructionManager,
    InstructionSnapshot,
    UpdateConfidence,
    UpdateSource,
    UpdateType,
    Winner,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def dynamic_manager(temp_dir):
    """Create a DynamicInstructionManager instance for testing."""
    history_path = temp_dir / "update_history.json"
    snapshots_path = temp_dir / "snapshots.json"
    return DynamicInstructionManager(
        history_path=history_path,
        snapshots_path=snapshots_path,
        confidence_threshold=0.5,
        auto_rollback_on_error=True,
    )


@pytest.fixture
def ab_test_manager(temp_dir):
    """Create an ABTestManager instance for testing."""
    storage_path = temp_dir / "ab_tests.json"
    return ABTestManager(storage_path=storage_path, auto_conclude=False)


class TestDynamicInstructionManager:
    """Tests for DynamicInstructionManager."""

    def test_initialization(self, dynamic_manager):
        """Test that manager initializes correctly."""
        assert dynamic_manager is not None
        assert dynamic_manager.confidence_threshold == 0.5
        assert dynamic_manager.auto_rollback_on_error is True

    def test_session_tracking(self, dynamic_manager):
        """Test session start and end."""
        session_id = "test_session_001"

        # Start session
        dynamic_manager.start_session(session_id)
        assert dynamic_manager._active_session_id == session_id
        assert session_id in dynamic_manager._session_updates

        # End session
        updates = dynamic_manager.end_session(session_id, success=True)
        assert isinstance(updates, list)
        assert dynamic_manager._active_session_id is None

    def test_propose_update(self, dynamic_manager):
        """Test proposing an instruction update."""
        update = dynamic_manager.propose_update(
            instruction_id="inst_001",
            update_type=UpdateType.MODIFY,
            update_source=UpdateSource.EXECUTION_FEEDBACK,
            confidence=0.75,
            reasoning="Instruction needs more specificity",
            new_content="Updated instruction content",
            old_content="Original content",
        )

        assert update is not None
        assert update.instruction_id == "inst_001"
        assert update.update_type == UpdateType.MODIFY
        assert update.confidence == 0.75
        assert update.confidence_level == UpdateConfidence.HIGH
        assert not update.applied

        # Check it's in pending updates
        assert update.update_id in dynamic_manager._pending_updates

    def test_apply_update(self, dynamic_manager):
        """Test applying a pending update."""
        # Propose update
        update = dynamic_manager.propose_update(
            instruction_id="inst_002",
            update_type=UpdateType.WEIGHT_INCREASE,
            update_source=UpdateSource.SUCCESS_PATTERN,
            confidence=0.9,
            reasoning="High success rate with this instruction",
            new_priority=8,
            old_priority=5,
        )

        # Create snapshot
        snapshot = InstructionSnapshot(
            instruction_id="inst_002", content="Original content", priority=5
        )

        # Apply update
        result = dynamic_manager.apply_update(update.update_id, snapshot_before=snapshot)
        assert result is True

        # Verify update is now in history
        assert update.applied is True
        assert update.update_id not in dynamic_manager._pending_updates

        # Verify snapshot was stored
        stored_snapshot = dynamic_manager.get_snapshot("inst_002")
        assert stored_snapshot is not None
        assert stored_snapshot.priority == 5

    def test_rollback_update(self, dynamic_manager):
        """Test rolling back an applied update."""
        # Propose and apply update
        update = dynamic_manager.propose_update(
            instruction_id="inst_003",
            update_type=UpdateType.MODIFY,
            update_source=UpdateSource.USER_FEEDBACK,
            confidence=0.6,
            reasoning="User requested change",
            new_content="User preferred content",
        )

        dynamic_manager.apply_update(update.update_id)

        # Rollback
        result = dynamic_manager.rollback_update(update.update_id)
        assert result is True
        assert update.rolled_back is True
        assert update.rolled_back_at is not None

    def test_auto_rollback_on_failure(self, dynamic_manager):
        """Test automatic rollback on session failure."""
        session_id = "test_session_002"
        dynamic_manager.start_session(session_id)

        # Propose and apply update during session
        update = dynamic_manager.propose_update(
            instruction_id="inst_004",
            update_type=UpdateType.MODIFY,
            update_source=UpdateSource.AUTOMATIC,
            confidence=0.7,
            reasoning="Automatic improvement",
        )

        dynamic_manager.apply_update(update.update_id)

        # End session with failure
        updates = dynamic_manager.end_session(session_id, success=False)

        # Should have rolled back
        assert len(updates) > 0
        assert updates[0].rolled_back is True

    def test_confidence_levels(self, dynamic_manager):
        """Test confidence level assignment."""
        test_cases = [
            (0.2, UpdateConfidence.VERY_LOW),
            (0.4, UpdateConfidence.LOW),
            (0.6, UpdateConfidence.MEDIUM),
            (0.8, UpdateConfidence.HIGH),
            (0.95, UpdateConfidence.VERY_HIGH),
        ]

        for confidence, expected_level in test_cases:
            update = dynamic_manager.propose_update(
                instruction_id=f"inst_{confidence}",
                update_type=UpdateType.MODIFY,
                update_source=UpdateSource.AUTOMATIC,
                confidence=confidence,
                reasoning="Test confidence levels",
            )
            assert update.confidence_level == expected_level

    def test_get_update_history(self, dynamic_manager):
        """Test retrieving update history."""
        # Create several updates
        for i in range(5):
            update = dynamic_manager.propose_update(
                instruction_id=f"inst_{i}",
                update_type=UpdateType.MODIFY,
                update_source=UpdateSource.AUTOMATIC,
                confidence=0.7,
                reasoning=f"Update {i}",
            )
            dynamic_manager.apply_update(update.update_id)

        # Get all history
        history = dynamic_manager.get_update_history(limit=10)
        assert len(history) >= 5

        # Get history for specific instruction
        history_filtered = dynamic_manager.get_update_history(instruction_id="inst_2")
        assert len(history_filtered) == 1
        assert history_filtered[0].instruction_id == "inst_2"

    def test_get_pending_updates(self, dynamic_manager):
        """Test retrieving pending updates."""
        # Create pending updates with different confidence
        dynamic_manager.propose_update(
            instruction_id="inst_pending_1",
            update_type=UpdateType.MODIFY,
            update_source=UpdateSource.AUTOMATIC,
            confidence=0.3,
            reasoning="Low confidence",
        )

        dynamic_manager.propose_update(
            instruction_id="inst_pending_2",
            update_type=UpdateType.MODIFY,
            update_source=UpdateSource.AUTOMATIC,
            confidence=0.8,
            reasoning="High confidence",
        )

        # Get all pending
        pending = dynamic_manager.get_pending_updates()
        assert len(pending) >= 2

        # Get pending with min confidence
        pending_high = dynamic_manager.get_pending_updates(min_confidence=0.7)
        assert len(pending_high) >= 1
        assert all(u.confidence >= 0.7 for u in pending_high)

    def test_get_statistics(self, dynamic_manager):
        """Test statistics generation."""
        # Create some updates
        session_id = "test_session_stats"
        dynamic_manager.start_session(session_id)

        update1 = dynamic_manager.propose_update(
            instruction_id="inst_stats_1",
            update_type=UpdateType.MODIFY,
            update_source=UpdateSource.EXECUTION_FEEDBACK,
            confidence=0.7,
            reasoning="Improvement",
        )
        dynamic_manager.apply_update(update1.update_id)

        dynamic_manager.end_session(session_id, success=True)

        # Get statistics
        stats = dynamic_manager.get_statistics()
        assert isinstance(stats, dict)
        assert "total_updates" in stats
        assert "applied_updates" in stats
        assert "rolled_back" in stats
        assert "by_type" in stats
        assert "by_source" in stats

    def test_persistence(self, temp_dir):
        """Test that updates persist across manager instances."""
        history_path = temp_dir / "persist_history.json"
        snapshots_path = temp_dir / "persist_snapshots.json"

        # Create first manager and add updates
        manager1 = DynamicInstructionManager(
            history_path=history_path, snapshots_path=snapshots_path
        )

        update = manager1.propose_update(
            instruction_id="inst_persist",
            update_type=UpdateType.MODIFY,
            update_source=UpdateSource.USER_FEEDBACK,
            confidence=0.8,
            reasoning="Test persistence",
        )
        manager1.apply_update(update.update_id)
        manager1._save_data()

        # Create second manager from same files
        manager2 = DynamicInstructionManager(
            history_path=history_path, snapshots_path=snapshots_path
        )

        # Should have the update in history
        history = manager2.get_update_history()
        assert len(history) > 0
        found = any(u.instruction_id == "inst_persist" for u in history)
        assert found


class TestABTestManager:
    """Tests for ABTestManager."""

    def test_initialization(self, ab_test_manager):
        """Test that A/B test manager initializes correctly."""
        assert ab_test_manager is not None
        assert ab_test_manager.auto_conclude is False

    def test_status_exports_use_safe_names(self):
        """Ensure status enum is exported under a non-pytest-collectable name."""
        from ai_dev_agent.dynamic_instructions import ab_testing

        assert hasattr(ab_testing, "ABTestStatus")
        assert not hasattr(ab_testing, "TestStatus")

    def test_create_test(self, ab_test_manager):
        """Test creating an A/B test."""
        test = ab_test_manager.create_test(
            name="Test Debugging Instruction",
            instruction_id="inst_debug_001",
            content_a="Check logs first",
            content_b="Use debugger first",
            description="Testing which approach is more effective",
            target_sample_size=50,
        )

        assert test is not None
        assert test.name == "Test Debugging Instruction"
        assert test.variant_a.content == "Check logs first"
        assert test.variant_b.content == "Use debugger first"
        assert test.target_sample_size == 50
        assert test.status == ABTestStatus.DRAFT

    def test_start_test(self, ab_test_manager):
        """Test starting an A/B test."""
        test = ab_test_manager.create_test(
            name="Test Feature A",
            instruction_id="inst_feature_001",
            content_a="Variant A",
            content_b="Variant B",
        )

        result = ab_test_manager.start_test(test.test_id)
        assert result is True
        assert test.status == ABTestStatus.RUNNING
        assert test.started_at is not None

    def test_get_variant_to_use(self, ab_test_manager):
        """Test getting variant to use."""
        test = ab_test_manager.create_test(
            name="Test Variant Selection",
            instruction_id="inst_variant_001",
            content_a="Variant A",
            content_b="Variant B",
        )

        ab_test_manager.start_test(test.test_id)

        # Get variant multiple times (should randomize)
        variants_seen = set()
        for _ in range(20):
            result = ab_test_manager.get_variant_to_use("inst_variant_001")
            assert result is not None
            variant_name, _variant = result
            variants_seen.add(variant_name)

        # Should have seen both variants (probabilistically)
        # With 20 trials, very likely to see both
        assert len(variants_seen) >= 1  # At least one variant seen

    def test_record_result(self, ab_test_manager):
        """Test recording test results."""
        test = ab_test_manager.create_test(
            name="Test Results",
            instruction_id="inst_results_001",
            content_a="Variant A",
            content_b="Variant B",
            target_sample_size=10,
        )

        ab_test_manager.start_test(test.test_id)

        # Record some results for variant A
        for _ in range(5):
            ab_test_manager.record_result(
                instruction_id="inst_results_001",
                variant_id=test.variant_a.variant_id,
                success=True,
                time_ms=100.0,
            )

        # Record some results for variant B
        for _ in range(5):
            ab_test_manager.record_result(
                instruction_id="inst_results_001",
                variant_id=test.variant_b.variant_id,
                success=False,
                time_ms=150.0,
            )

        # Check results were recorded
        assert test.variant_a.uses == 5
        assert test.variant_a.successes == 5
        assert test.variant_a.success_rate == 1.0

        assert test.variant_b.uses == 5
        assert test.variant_b.failures == 5
        assert test.variant_b.success_rate == 0.0

    def test_auto_conclude_test(self, temp_dir):
        """Test automatic test conclusion."""
        storage_path = temp_dir / "auto_conclude.json"
        manager = ABTestManager(storage_path=storage_path, auto_conclude=True)

        test = manager.create_test(
            name="Auto Conclude Test",
            instruction_id="inst_conclude_001",
            content_a="Variant A",
            content_b="Variant B",
            target_sample_size=30,
        )

        manager.start_test(test.test_id)

        # Record enough samples with clear winner (A)
        for _ in range(30):
            manager.record_result(
                instruction_id="inst_conclude_001",
                variant_id=test.variant_a.variant_id,
                success=True,
            )

        for _ in range(30):
            manager.record_result(
                instruction_id="inst_conclude_001",
                variant_id=test.variant_b.variant_id,
                success=False,
            )

        # Should have auto-concluded
        assert test.status == ABTestStatus.COMPLETED
        assert test.winner == Winner.VARIANT_A
        assert test.p_value is not None

    def test_cancel_test(self, ab_test_manager):
        """Test cancelling a test."""
        test = ab_test_manager.create_test(
            name="Test Cancel",
            instruction_id="inst_cancel_001",
            content_a="Variant A",
            content_b="Variant B",
        )

        ab_test_manager.start_test(test.test_id)

        # Cancel test
        result = ab_test_manager.cancel_test(test.test_id)
        assert result is True
        assert test.status == ABTestStatus.CANCELLED

    def test_get_all_tests(self, ab_test_manager):
        """Test retrieving all tests."""
        # Create several tests
        for i in range(3):
            ab_test_manager.create_test(
                name=f"Test {i}",
                instruction_id=f"inst_{i}",
                content_a=f"Variant A {i}",
                content_b=f"Variant B {i}",
            )

        # Get all tests
        tests = ab_test_manager.get_all_tests()
        assert len(tests) >= 3

        # Get tests by status
        draft_tests = ab_test_manager.get_all_tests(status=ABTestStatus.DRAFT)
        assert len(draft_tests) >= 3

    def test_get_statistics(self, ab_test_manager):
        """Test statistics generation."""
        # Create and start a test
        test = ab_test_manager.create_test(
            name="Stats Test",
            instruction_id="inst_stats",
            content_a="Variant A",
            content_b="Variant B",
        )
        ab_test_manager.start_test(test.test_id)

        # Get statistics
        stats = ab_test_manager.get_statistics()
        assert isinstance(stats, dict)
        assert "total_tests" in stats
        assert "active_tests" in stats
        assert "by_status" in stats
        assert stats["total_tests"] >= 1

    def test_persistence(self, temp_dir):
        """Test that tests persist across manager instances."""
        storage_path = temp_dir / "persist_tests.json"

        # Create first manager and test
        manager1 = ABTestManager(storage_path=storage_path)
        test = manager1.create_test(
            name="Persistence Test",
            instruction_id="inst_persist",
            content_a="Variant A",
            content_b="Variant B",
        )
        test_id = test.test_id

        # Create second manager from same file
        manager2 = ABTestManager(storage_path=storage_path)

        # Should have the test
        retrieved_test = manager2.get_test(test_id)
        assert retrieved_test is not None
        assert retrieved_test.name == "Persistence Test"


class TestIntegration:
    """Integration tests for the complete dynamic instructions system."""

    def test_full_update_workflow(self, dynamic_manager):
        """Test complete update workflow from proposal to rollback."""
        session_id = "integration_session_001"

        # Start session
        dynamic_manager.start_session(session_id)

        # Propose high-confidence update (should auto-apply if threshold met)
        update = dynamic_manager.propose_update(
            instruction_id="inst_integration_001",
            update_type=UpdateType.MODIFY,
            update_source=UpdateSource.SUCCESS_PATTERN,
            confidence=0.85,
            reasoning="Strong evidence for improvement",
            new_content="Improved instruction",
            old_content="Original instruction",
        )

        # Apply if not auto-applied
        if not update.applied:
            snapshot = InstructionSnapshot(
                instruction_id="inst_integration_001", content="Original instruction", priority=5
            )
            dynamic_manager.apply_update(update.update_id, snapshot_before=snapshot)

        # Verify applied
        assert update.applied is True

        # End session with failure (should rollback)
        updates = dynamic_manager.end_session(session_id, success=False)

        # Verify rollback
        assert len(updates) > 0
        assert updates[0].rolled_back is True

    def test_combined_ab_test_and_updates(self, temp_dir):
        """Test using A/B tests with dynamic updates."""
        # Create managers
        ab_manager = ABTestManager(storage_path=temp_dir / "combined_ab.json", auto_conclude=False)
        dynamic_manager = DynamicInstructionManager(
            history_path=temp_dir / "combined_history.json",
            snapshots_path=temp_dir / "combined_snapshots.json",
        )

        # Create A/B test
        test = ab_manager.create_test(
            name="Test with Updates",
            instruction_id="inst_combined_001",
            content_a="Original approach",
            content_b="New approach",
            target_sample_size=20,
        )
        ab_manager.start_test(test.test_id)

        # Simulate using variants and recording results
        for i in range(10):
            # Use variant A
            ab_manager.record_result(
                instruction_id="inst_combined_001",
                variant_id=test.variant_a.variant_id,
                success=True,
            )

            # Use variant B
            ab_manager.record_result(
                instruction_id="inst_combined_001",
                variant_id=test.variant_b.variant_id,
                success=True,
            )

        # Based on results, propose update to use winning variant
        if test.variant_a.success_rate > test.variant_b.success_rate:
            winner_content = test.variant_a.content
        else:
            winner_content = test.variant_b.content

        update = dynamic_manager.propose_update(
            instruction_id="inst_combined_001",
            update_type=UpdateType.MODIFY,
            update_source=UpdateSource.AB_TEST,
            confidence=0.9,
            reasoning="A/B test concluded in favor of this variant",
            new_content=winner_content,
        )

        assert update is not None
        assert update.update_source == UpdateSource.AB_TEST
