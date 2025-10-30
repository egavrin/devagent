"""Playbook system for DevAgent - ACE pattern implementation.

This module implements an evolving playbook of instructions that:
1. Maintains structured instructions organized by category
2. Updates incrementally (deltas) without full rewrites
3. Avoids context collapse through careful curation
4. Tracks instruction effectiveness over time
"""

from .curator import PlaybookCurator
from .manager import Instruction, InstructionCategory, PlaybookManager

__all__ = [
    "Instruction",
    "InstructionCategory",
    "PlaybookCurator",
    "PlaybookManager",
]
