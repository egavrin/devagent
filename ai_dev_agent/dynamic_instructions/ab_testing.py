"""A/B Testing Framework for instruction variants.

Implements statistical A/B testing to determine which instruction variants
perform better in practice.
"""

from __future__ import annotations

import json
import logging
import math
import random
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)


class ABTestStatus(str, Enum):
    """Status of an A/B test."""
    DRAFT = "draft"              # Test created but not started
    RUNNING = "running"          # Test is active
    PAUSED = "paused"            # Test temporarily paused
    COMPLETED = "completed"      # Test finished with conclusion
    CANCELLED = "cancelled"      # Test cancelled without conclusion


class Winner(str, Enum):
    """Winner of an A/B test."""
    VARIANT_A = "variant_a"
    VARIANT_B = "variant_b"
    NO_WINNER = "no_winner"      # No statistically significant difference


@dataclass
class InstructionVariant:
    """A variant in an A/B test."""

    variant_id: str
    instruction_id: str          # ID of the instruction being tested
    content: str                 # Instruction content
    priority: int = 5
    tags: Set[str] = field(default_factory=set)

    # Performance tracking
    uses: int = 0
    successes: int = 0
    failures: int = 0
    total_time_ms: float = 0.0   # Total execution time

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.uses == 0:
            return 0.0
        return self.successes / self.uses

    @property
    def avg_time_ms(self) -> float:
        """Calculate average execution time."""
        if self.uses == 0:
            return 0.0
        return self.total_time_ms / self.uses

    def record_result(self, success: bool, time_ms: float = 0.0) -> None:
        """Record a result for this variant."""
        self.uses += 1
        if success:
            self.successes += 1
        else:
            self.failures += 1
        self.total_time_ms += time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["tags"] = list(self.tags)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> InstructionVariant:
        """Create from dictionary."""
        if "tags" in data and isinstance(data["tags"], list):
            data["tags"] = set(data["tags"])
        return cls(**data)


@dataclass
class ABTest:
    """An A/B test comparing two instruction variants."""

    test_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""

    # Variants
    variant_a: InstructionVariant = field(default_factory=lambda: InstructionVariant("", "", ""))
    variant_b: InstructionVariant = field(default_factory=lambda: InstructionVariant("", "", ""))

    # Test configuration
    target_sample_size: int = 100      # Minimum samples per variant
    confidence_level: float = 0.95     # Statistical confidence (95%)
    min_effect_size: float = 0.05      # Minimum detectable effect (5%)

    # Status
    status: ABTestStatus = ABTestStatus.DRAFT
    winner: Winner = Winner.NO_WINNER

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    # Statistics
    p_value: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None

    @property
    def total_uses(self) -> int:
        """Total uses across both variants."""
        return self.variant_a.uses + self.variant_b.uses

    @property
    def is_complete(self) -> bool:
        """Check if test has enough samples."""
        return (self.variant_a.uses >= self.target_sample_size and
                self.variant_b.uses >= self.target_sample_size)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["status"] = self.status.value
        data["winner"] = self.winner.value
        data["variant_a"] = self.variant_a.to_dict()
        data["variant_b"] = self.variant_b.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ABTest:
        """Create from dictionary."""
        if "status" in data and isinstance(data["status"], str):
            data["status"] = ABTestStatus(data["status"])
        if "winner" in data and isinstance(data["winner"], str):
            data["winner"] = Winner(data["winner"])
        if "variant_a" in data:
            data["variant_a"] = InstructionVariant.from_dict(data["variant_a"])
        if "variant_b" in data:
            data["variant_b"] = InstructionVariant.from_dict(data["variant_b"])
        return cls(**data)


class ABTestManager:
    """Manages A/B tests for instruction variants."""

    DEFAULT_STORAGE_PATH = Path.home() / ".devagent" / "dynamic_instructions" / "ab_tests.json"

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        auto_conclude: bool = True
    ):
        """Initialize the A/B test manager.

        Args:
            storage_path: Path to test storage
            auto_conclude: Automatically conclude tests when statistically significant
        """
        self.storage_path = storage_path or self.DEFAULT_STORAGE_PATH
        self.auto_conclude = auto_conclude

        # Thread safety
        self._lock = threading.RLock()

        # In-memory storage
        self._tests: Dict[str, ABTest] = {}  # test_id -> test
        self._active_tests: Dict[str, str] = {}  # instruction_id -> test_id

        # Load existing tests
        self._load_tests()

    def _load_tests(self) -> None:
        """Load tests from storage."""
        if not self.storage_path.exists():
            return

        try:
            with open(self.storage_path, "r") as f:
                data = json.load(f)
                for test_data in data.get("tests", []):
                    test = ABTest.from_dict(test_data)
                    self._tests[test.test_id] = test

                    # Rebuild active tests index
                    if test.status == ABTestStatus.RUNNING:
                        inst_id = test.variant_a.instruction_id
                        self._active_tests[inst_id] = test.test_id

            logger.info(f"Loaded {len(self._tests)} A/B tests")
        except Exception as e:
            logger.error(f"Failed to load A/B tests: {e}")

    def _save_tests(self) -> None:
        """Save tests to storage."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, "w") as f:
                json.dump({
                    "tests": [test.to_dict() for test in self._tests.values()],
                    "saved_at": datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save A/B tests: {e}")

    def create_test(
        self,
        name: str,
        instruction_id: str,
        content_a: str,
        content_b: str,
        description: str = "",
        priority_a: int = 5,
        priority_b: int = 5,
        target_sample_size: int = 100,
        confidence_level: float = 0.95
    ) -> ABTest:
        """Create a new A/B test.

        Args:
            name: Test name
            instruction_id: ID of instruction being tested
            content_a: Variant A content
            content_b: Variant B content
            description: Test description
            priority_a: Priority for variant A
            priority_b: Priority for variant B
            target_sample_size: Samples needed per variant
            confidence_level: Statistical confidence level

        Returns:
            Created ABTest object
        """
        with self._lock:
            variant_a = InstructionVariant(
                variant_id=f"{instruction_id}_a",
                instruction_id=instruction_id,
                content=content_a,
                priority=priority_a
            )

            variant_b = InstructionVariant(
                variant_id=f"{instruction_id}_b",
                instruction_id=instruction_id,
                content=content_b,
                priority=priority_b
            )

            test = ABTest(
                name=name,
                description=description,
                variant_a=variant_a,
                variant_b=variant_b,
                target_sample_size=target_sample_size,
                confidence_level=confidence_level,
                status=ABTestStatus.DRAFT
            )

            self._tests[test.test_id] = test
            self._save_tests()

            logger.info(f"Created A/B test: {name}")
            return test

    def start_test(self, test_id: str) -> bool:
        """Start an A/B test.

        Args:
            test_id: Test ID

        Returns:
            True if started successfully
        """
        with self._lock:
            if test_id not in self._tests:
                logger.warning(f"Test {test_id} not found")
                return False

            test = self._tests[test_id]

            if test.status != ABTestStatus.DRAFT and test.status != ABTestStatus.PAUSED:
                logger.warning(f"Test {test_id} cannot be started (status: {test.status})")
                return False

            test.status = ABTestStatus.RUNNING
            test.started_at = datetime.now().isoformat()

            # Register as active test
            inst_id = test.variant_a.instruction_id
            self._active_tests[inst_id] = test_id

            self._save_tests()
            logger.info(f"Started A/B test: {test.name}")
            return True

    def get_variant_to_use(self, instruction_id: str) -> Optional[Tuple[str, InstructionVariant]]:
        """Get which variant to use for an instruction.

        Args:
            instruction_id: Instruction ID

        Returns:
            Tuple of (variant_name, variant) or None if no active test
        """
        with self._lock:
            # Check if there's an active test
            if instruction_id not in self._active_tests:
                return None

            test_id = self._active_tests[instruction_id]
            test = self._tests.get(test_id)

            if not test or test.status != ABTestStatus.RUNNING:
                return None

            # Simple randomization (50/50 split)
            if random.random() < 0.5:
                return ("variant_a", test.variant_a)
            else:
                return ("variant_b", test.variant_b)

    def record_result(
        self,
        instruction_id: str,
        variant_id: str,
        success: bool,
        time_ms: float = 0.0
    ) -> None:
        """Record a result for a variant.

        Args:
            instruction_id: Instruction ID
            variant_id: Variant ID
            success: Whether task was successful
            time_ms: Execution time in milliseconds
        """
        with self._lock:
            if instruction_id not in self._active_tests:
                return

            test_id = self._active_tests[instruction_id]
            test = self._tests.get(test_id)

            if not test or test.status != ABTestStatus.RUNNING:
                return

            # Record result for appropriate variant
            if variant_id == test.variant_a.variant_id:
                test.variant_a.record_result(success, time_ms)
            elif variant_id == test.variant_b.variant_id:
                test.variant_b.record_result(success, time_ms)
            else:
                logger.warning(f"Unknown variant ID: {variant_id}")
                return

            # Check if test is complete and should be concluded
            if self.auto_conclude and test.is_complete:
                self._analyze_and_conclude(test_id)

            self._save_tests()

    def _analyze_and_conclude(self, test_id: str) -> None:
        """Analyze test results and conclude if statistically significant.

        Args:
            test_id: Test ID
        """
        with self._lock:
            test = self._tests.get(test_id)
            if not test:
                return

            # Perform two-proportion z-test
            p_value = self._two_proportion_z_test(
                test.variant_a.successes, test.variant_a.uses,
                test.variant_b.successes, test.variant_b.uses
            )

            test.p_value = p_value

            # Determine winner based on p-value and confidence level
            alpha = 1 - test.confidence_level

            if p_value < alpha:
                # Statistically significant difference
                if test.variant_a.success_rate > test.variant_b.success_rate:
                    test.winner = Winner.VARIANT_A
                else:
                    test.winner = Winner.VARIANT_B
            else:
                # No significant difference
                test.winner = Winner.NO_WINNER

            # Mark as completed
            test.status = ABTestStatus.COMPLETED
            test.completed_at = datetime.now().isoformat()

            # Remove from active tests
            inst_id = test.variant_a.instruction_id
            if inst_id in self._active_tests:
                del self._active_tests[inst_id]

            logger.info(f"Concluded A/B test '{test.name}': winner={test.winner.value}, p={p_value:.4f}")
            self._save_tests()

    def _two_proportion_z_test(
        self,
        successes_a: int,
        total_a: int,
        successes_b: int,
        total_b: int
    ) -> float:
        """Perform two-proportion z-test.

        Args:
            successes_a: Successes in variant A
            total_a: Total samples in variant A
            successes_b: Successes in variant B
            total_b: Total samples in variant B

        Returns:
            P-value
        """
        if total_a == 0 or total_b == 0:
            return 1.0  # Cannot determine

        p_a = successes_a / total_a
        p_b = successes_b / total_b

        # Pooled proportion
        p_pool = (successes_a + successes_b) / (total_a + total_b)

        # Standard error
        se = math.sqrt(p_pool * (1 - p_pool) * (1/total_a + 1/total_b))

        if se == 0:
            return 1.0

        # Z-score
        z = (p_a - p_b) / se

        # Two-tailed p-value (approximation using normal distribution)
        # For simplicity, using approximation: p ≈ 2 * (1 - Φ(|z|))
        # where Φ is the standard normal CDF
        p_value = 2 * (1 - self._normal_cdf(abs(z)))

        return max(0.0, min(1.0, p_value))

    def _normal_cdf(self, x: float) -> float:
        """Approximate standard normal CDF.

        Args:
            x: Value

        Returns:
            CDF at x
        """
        # Using error function approximation
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    def get_test(self, test_id: str) -> Optional[ABTest]:
        """Get a test by ID.

        Args:
            test_id: Test ID

        Returns:
            ABTest or None
        """
        with self._lock:
            return self._tests.get(test_id)

    def get_all_tests(
        self,
        status: Optional[ABTestStatus] = None
    ) -> List[ABTest]:
        """Get all tests with optional status filter.

        Args:
            status: Filter by status

        Returns:
            List of ABTest objects
        """
        with self._lock:
            tests = list(self._tests.values())

            if status:
                tests = [t for t in tests if t.status == status]

            return sorted(tests, key=lambda t: t.created_at, reverse=True)

    def cancel_test(self, test_id: str) -> bool:
        """Cancel a running test.

        Args:
            test_id: Test ID

        Returns:
            True if cancelled successfully
        """
        with self._lock:
            test = self._tests.get(test_id)
            if not test:
                return False

            if test.status not in [ABTestStatus.RUNNING, ABTestStatus.PAUSED]:
                logger.warning(f"Test {test_id} cannot be cancelled (status: {test.status})")
                return False

            test.status = ABTestStatus.CANCELLED

            # Remove from active tests
            inst_id = test.variant_a.instruction_id
            if inst_id in self._active_tests:
                del self._active_tests[inst_id]

            self._save_tests()
            logger.info(f"Cancelled A/B test: {test.name}")
            return True

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about A/B tests.

        Returns:
            Dictionary with statistics
        """
        with self._lock:
            total_tests = len(self._tests)
            by_status = {}
            by_winner = {}

            for test in self._tests.values():
                status = test.status.value
                by_status[status] = by_status.get(status, 0) + 1

                if test.status == ABTestStatus.COMPLETED:
                    winner = test.winner.value
                    by_winner[winner] = by_winner.get(winner, 0) + 1

            completed_tests = [t for t in self._tests.values() if t.status == ABTestStatus.COMPLETED]
            avg_p_value = (sum(t.p_value for t in completed_tests if t.p_value) / len(completed_tests)
                          if completed_tests else 0.0)

            return {
                "total_tests": total_tests,
                "active_tests": len(self._active_tests),
                "by_status": by_status,
                "by_winner": by_winner,
                "average_p_value": avg_p_value,
                "completed_tests": len(completed_tests)
            }
