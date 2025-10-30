"""Dynamic Instructions - ILWS (Instruction-Level Weight Shaping) Pattern.

This module provides real-time instruction updates and A/B testing capabilities
for continuous learning and optimization during task execution.
"""

from .ab_testing import ABTest, ABTestManager, ABTestStatus, InstructionVariant, Winner
from .manager import (
    DynamicInstructionManager,
    InstructionSnapshot,
    InstructionUpdate,
    UpdateConfidence,
    UpdateSource,
    UpdateType,
)

__all__ = [
    "ABTest",
    # A/B Testing
    "ABTestManager",
    "ABTestStatus",
    # Manager
    "DynamicInstructionManager",
    "InstructionSnapshot",
    "InstructionUpdate",
    "InstructionVariant",
    "UpdateConfidence",
    "UpdateSource",
    "UpdateType",
    "Winner",
]
