"""Filesystem tool implementations."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

from ai_dev_agent.tools.code.code_edit.diff_utils import DiffError, DiffProcessor

from ..names import EDIT, READ
from ..registry import ToolContext, ToolSpec, registry
from .search_replace import _fs_edit

if TYPE_CHECKING:
    from collections.abc import Mapping

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schemas" / "tools"


def _resolve_path(repo_root: Path, relative: str) -> Path:
    candidate = (repo_root / relative).resolve()
    if repo_root not in candidate.parents and candidate != repo_root:
        raise ValueError(f"Path '{relative}' escapes repository root")
    return candidate


def _fs_read(payload: Mapping[str, Any], context: ToolContext) -> Mapping[str, Any]:
    repo_root = context.repo_root
    files: list[dict[str, str]] = []
    byte_range = payload.get("byte_range")
    context_lines = payload.get("context_lines")

    paths_param = payload.get("paths")
    if paths_param is None:
        single_path = payload.get("path")
        paths_param = [single_path] if isinstance(single_path, str) and single_path.strip() else []
    elif isinstance(paths_param, (str, bytes)):
        # Allow degenerate inputs that pass validation but provide a single string
        if isinstance(paths_param, bytes):
            paths_param = [paths_param.decode("utf-8", errors="ignore")]
        else:
            paths_param = [paths_param]

    if not paths_param:
        raise ValueError("No readable paths provided")

    for rel in paths_param:
        target = _resolve_path(repo_root, rel)
        if not target.exists() or not target.is_file():
            raise ValueError(f"File '{rel}' not found in workspace")
        text = target.read_text(encoding="utf-8", errors="replace")
        original_text = text

        if byte_range is not None:
            start, end = byte_range
            text = original_text[start:end]
        elif context_lines is not None:
            lines = original_text.splitlines()
            if context_lines >= len(lines):
                text = original_text
            else:
                text = "\n".join(lines[:context_lines])

        digest = hashlib.sha256(original_text.encode("utf-8", errors="ignore")).hexdigest()
        files.append(
            {
                "path": rel,
                "content": text,
                "sha256": digest,
            }
        )

    return {"files": files}


def _parse_diff_stats(diff: str) -> tuple[int, int, list[str], list[str]]:
    """Parse diff statistics supporting both git format and simple unified diff.

    Returns: (total_lines_changed, file_count, changed_files, new_files)
    """
    total_lines = 0
    files: set[str] = set()
    new_files: set[str] = set()
    current_file: str | None = None
    last_old_file: str | None = None

    for line in diff.splitlines():
        if line.startswith("diff --git "):
            # Git format: diff --git a/file.txt b/file.txt
            parts = line.split()
            if len(parts) >= 4:
                a_part = parts[2][2:]  # Remove "a/" prefix
                b_part = parts[3][2:]  # Remove "b/" prefix
                current_file = b_part
                if a_part == "/dev/null":
                    new_files.add(b_part)
                files.add(b_part)
        elif line.startswith("--- "):
            # Unified diff old file marker
            old_file = line[4:].strip()
            if old_file == "/dev/null":
                last_old_file = "/dev/null"
            elif old_file:
                # Extract just the filename, removing a/ prefix if present
                if old_file.startswith("a/"):
                    old_file = old_file[2:]
                last_old_file = old_file
        elif line.startswith("+++ "):
            # Unified diff new file marker
            new_file = line[4:].strip()
            if new_file and new_file != "/dev/null":
                # Extract just the filename, removing b/ prefix if present
                if new_file.startswith("b/"):
                    new_file = new_file[2:]
                current_file = new_file
                files.add(new_file)
                # Check if this is a new file creation (old file was /dev/null)
                if last_old_file == "/dev/null":
                    new_files.add(new_file)
        elif line.startswith("+") or line.startswith("-"):
            # Count changed lines (excluding hunk headers like @@ -1 +1 @@)
            if not line.startswith("+++") and not line.startswith("---"):
                total_lines += 1
                if current_file:
                    files.add(current_file)

    return total_lines, len(files), sorted(files), sorted(new_files)


def _fs_write_patch(payload: Mapping[str, Any], context: ToolContext) -> Mapping[str, Any]:
    """Apply a diff patch using DiffProcessor with validation and fallback.

    This provides:
    - Pre-validation of diff format
    - Three-tier fallback: git apply → patch command → Python fallback
    - Detailed error messages for troubleshooting
    - Conflict marker detection
    """
    repo_root = context.repo_root
    diff = payload["diff"]

    # Create DiffProcessor for enhanced diff handling
    processor = DiffProcessor(repo_root)

    try:
        # Validate the diff before attempting to apply
        # This catches format errors early with helpful messages
        validation = processor._validate_diff(diff)

        if validation.errors:
            # Return validation errors as rejected hunks
            return {
                "applied": False,
                "rejected_hunks": [f"Diff validation failed: {'; '.join(validation.errors)}"],
                "new_files": [],
                "changed_files": [],
                "diff_stats": {"lines": 0, "files": 0},
            }

        # Apply the diff with three-tier fallback strategy
        success = processor.apply_diff_safely(diff)

        if not success:
            # This shouldn't normally happen after validation, but handle it
            return {
                "applied": False,
                "rejected_hunks": ["Patch application failed after validation"],
                "new_files": [],
                "changed_files": [],
                "diff_stats": {"lines": 0, "files": 0},
            }

        # Parse stats from successfully applied diff
        lines, file_count, files, new_files = _parse_diff_stats(diff)

        return {
            "applied": True,
            "rejected_hunks": [],
            "new_files": new_files,
            "changed_files": files,
            "diff_stats": {"lines": lines, "files": file_count},
        }

    except DiffError as exc:
        # DiffProcessor raised an error - capture the detailed message
        error_msg = str(exc)
        return {
            "applied": False,
            "rejected_hunks": [error_msg],
            "new_files": [],
            "changed_files": [],
            "diff_stats": {"lines": 0, "files": 0},
        }

    except Exception as exc:
        # Unexpected error - provide context for debugging
        error_msg = f"Unexpected error applying patch: {type(exc).__name__}: {exc}"
        return {
            "applied": False,
            "rejected_hunks": [error_msg],
            "new_files": [],
            "changed_files": [],
            "diff_stats": {"lines": 0, "files": 0},
        }


registry.register(
    ToolSpec(
        name=READ,
        handler=_fs_read,
        request_schema_path=SCHEMA_DIR / "read.request.json",
        response_schema_path=SCHEMA_DIR / "read.response.json",
        description=(
            "Read file contents from the repository. Provide 'paths' (list of file paths) to read. "
            "Optional parameters: 'context_lines' (int) to limit output, or 'byte_range' ([start, end]) "
            "for large files. Returns contents with line numbers. Use this after find/grep to examine "
            "specific files you've located."
        ),
        category="file_read",
    )
)

registry.register(
    ToolSpec(
        name=EDIT,
        handler=_fs_edit,
        request_schema_path=SCHEMA_DIR / "edit.request.json",
        response_schema_path=SCHEMA_DIR / "edit.response.json",
        description=(
            "Edit or create files using SEARCH/REPLACE blocks. "
            "WORKFLOW (MANDATORY): 1) READ file first, 2) Copy EXACT text from read output (character-for-character, "
            "including ALL spaces/tabs/newlines) into SEARCH block, 3) Put new code in REPLACE block. "
            "PRE-VALIDATION: ALL blocks checked before ANY changes applied - if even ONE character doesn't match exactly, "
            "NO changes applied, file unchanged. NEVER paraphrase, retype, add comments, or 'clean up' code in SEARCH block - "
            "it MUST be byte-for-byte identical to file. EXAMPLE: Read shows 'def f(x):\\n    return x' → SEARCH must be "
            "EXACTLY 'def f(x):\\n    return x' (not 'def f(x):  # function' or 'def f(x): return x'). For new files, use empty SEARCH blocks."
        ),
        category="command",
    )
)


__all__ = ["_fs_read", "_fs_write_patch", "_fs_edit"]
