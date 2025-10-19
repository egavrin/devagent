"""Dynamic Instructions - ILWS (Instruction-Level Weight Shaping) Pattern.

This module provides real-time instruction updates and A/B testing capabilities
for continuous learning and optimization during task execution.
"""

from .manager import (
    DynamicInstructionManager,
    InstructionUpdate,
    InstructionSnapshot,
    UpdateConfidence,
    UpdateType,
    UpdateSource
)

from .ab_testing import (
    ABTestManager,
    ABTest,
    InstructionVariant,
    ABTestStatus,
    Winner
)

__all__ = [
    # Manager
    "DynamicInstructionManager",
    "InstructionUpdate",
    "InstructionSnapshot",
    "UpdateConfidence",
    "UpdateType",
    "UpdateSource",

    # A/B Testing
    "ABTestManager",
    "ABTest",
    "InstructionVariant",
    "ABTestStatus",
    "Winner",
]
