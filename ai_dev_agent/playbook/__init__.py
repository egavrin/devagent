"""Playbook system for DevAgent - ACE pattern implementation.

This module implements an evolving playbook of instructions that:
1. Maintains structured instructions organized by category
2. Updates incrementally (deltas) without full rewrites
3. Avoids context collapse through careful curation
4. Tracks instruction effectiveness over time
"""

from .manager import PlaybookManager, Instruction, InstructionCategory
from .curator import PlaybookCurator

__all__ = [
    "PlaybookManager",
    "Instruction",
    "InstructionCategory",
    "PlaybookCurator",
]
