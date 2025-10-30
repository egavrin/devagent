"""Code editing module exports."""

from .context import FileContext, gather_file_contexts
from .diff_utils import DiffError, apply_patch, extract_diff
from .editor import CodeEditor, DiffProposal

__all__ = [
    "CodeEditor",
    "DiffError",
    "DiffProposal",
    "FileContext",
    "apply_patch",
    "extract_diff",
    "gather_file_contexts",
]
