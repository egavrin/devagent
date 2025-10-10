"""Filesystem tool implementations."""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from ..registry import ToolSpec, ToolContext, registry
from ..names import READ, WRITE

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schemas" / "tools"


def _resolve_path(repo_root: Path, relative: str) -> Path:
    candidate = (repo_root / relative).resolve()
    if repo_root not in candidate.parents and candidate != repo_root:
        raise ValueError(f"Path '{relative}' escapes repository root")
    return candidate


def _fs_read(payload: Mapping[str, Any], context: ToolContext) -> Mapping[str, Any]:
    repo_root = context.repo_root
    files: list[Dict[str, str]] = []
    byte_range = payload.get("byte_range")
    context_lines = payload.get("context_lines")

    paths_param = payload.get("paths")
    if paths_param is None:
        single_path = payload.get("path")
        if isinstance(single_path, str) and single_path.strip():
            paths_param = [single_path]
        else:
            paths_param = []
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
                text = "\n".join(lines[: context_lines])

        digest = hashlib.sha256(original_text.encode("utf-8", errors="ignore")).hexdigest()
        files.append({
            "path": rel,
            "content": text,
            "sha256": digest,
        })

    return {"files": files}


def _parse_diff_stats(diff: str) -> tuple[int, int, list[str], list[str]]:
    total_lines = 0
    files: set[str] = set()
    new_files: set[str] = set()
    current_file: str | None = None
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                a_part = parts[2][2:]
                b_part = parts[3][2:]
                current_file = b_part
                if a_part == "/dev/null":
                    new_files.add(b_part)
                files.add(b_part)
        elif line.startswith("+++") or line.startswith("---"):
            continue
        elif line.startswith("+") or line.startswith("-"):
            total_lines += 1
            if current_file:
                files.add(current_file)
    return total_lines, len(files), sorted(files), sorted(new_files)


def _run_git_apply(repo_root: Path, diff: str, check_only: bool) -> subprocess.CompletedProcess[str]:
    args = ["git", "apply"]
    if check_only:
        args.append("--check")
    process = subprocess.run(
        args,
        cwd=str(repo_root),
        input=diff,
        text=True,
        capture_output=True,
        check=False,
    )
    return process


def _fs_write_patch(payload: Mapping[str, Any], context: ToolContext) -> Mapping[str, Any]:
    repo_root = context.repo_root
    diff = payload["diff"]

    process = _run_git_apply(repo_root, diff, check_only=True)
    if process.returncode != 0:
        return {
            "applied": False,
            "rejected_hunks": [process.stderr.strip() or process.stdout.strip()],
            "new_files": [],
            "changed_files": [],
            "diff_stats": {"lines": 0, "files": 0},
        }

    apply_proc = _run_git_apply(repo_root, diff, check_only=False)
    if apply_proc.returncode != 0:
        return {
            "applied": False,
            "rejected_hunks": [apply_proc.stderr.strip() or apply_proc.stdout.strip()],
            "new_files": [],
            "changed_files": [],
            "diff_stats": {"lines": 0, "files": 0},
        }

    lines, file_count, files, new_files = _parse_diff_stats(diff)
    return {
        "applied": True,
        "rejected_hunks": [],
        "new_files": new_files,
        "changed_files": files,
        "diff_stats": {"lines": lines, "files": file_count},
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
        name=WRITE,
        handler=_fs_write_patch,
        request_schema_path=SCHEMA_DIR / "write.request.json",
        response_schema_path=SCHEMA_DIR / "write.response.json",
        description=(
            "Apply a unified diff patch to modify existing files. Requires 'diff' (string in unified diff format). "
            "Preferred over rewriting entire files as it shows precise changes and is safer. Automatically validates "
            "patch format and applies changes atomically. Use this for surgical code modifications."
        ),
        category="command",
    )
)


__all__ = ["_fs_read", "_fs_write_patch"]
