"""Diff-based risk metrics."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from ai_dev_agent.core.utils.logger import get_logger

LOGGER = get_logger(__name__)


@dataclass
class DiffMetrics:
    """Summary of diff size and concentration."""

    total_lines: int
    file_count: int
    concentration: float
    files: list[str]


def compute_diff_metrics(
    repo_root: Path, *, compare_ref: str | None = None, include_untracked: bool = False
) -> DiffMetrics:
    """Compute simple diff metrics for the working tree.

    Parameters
    ----------
    repo_root:
        Git repository root.
    compare_ref:
        Optional git ref to diff against. Defaults to HEAD.
    include_untracked:
        When true, incorporate untracked files by counting their line totals.
    """

    repo_root = Path(repo_root)
    ref = compare_ref or "HEAD"

    numstat_cmd = ["git", "diff", "--numstat", ref]
    name_only_cmd = ["git", "diff", "--name-only", ref]

    numstat = _run_git(numstat_cmd, repo_root)
    names = _run_git(name_only_cmd, repo_root)

    total_lines = 0
    files: list[str] = []

    for line in numstat.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        try:
            added = int(parts[0]) if parts[0] != "-" else 0
            removed = int(parts[1]) if parts[1] != "-" else 0
        except ValueError:
            continue
        total_lines += added + removed
        files.append(parts[2])

    if not files:
        files = [entry for entry in names.splitlines() if entry.strip()]

    file_count = len(set(files))
    concentration = float(total_lines) / file_count if file_count else 0.0

    if include_untracked:
        untracked = _collect_untracked(repo_root)
        for path in untracked:
            try:
                text = (repo_root / path).read_text(encoding="utf-8")
                total_lines += text.count("\n") + 1
                files.append(path)
            except OSError:
                continue
        file_count = len(set(files))
        concentration = float(total_lines) / file_count if file_count else 0.0

    return DiffMetrics(
        total_lines=total_lines,
        file_count=file_count,
        concentration=concentration,
        files=sorted(set(files)),
    )


def _collect_untracked(repo_root: Path) -> list[str]:
    output = _run_git(["git", "ls-files", "--other", "--exclude-standard"], repo_root)
    return [line.strip() for line in output.splitlines() if line.strip()]


def _run_git(command: list[str], repo_root: Path) -> str:
    try:
        process = subprocess.run(
            command,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:  # pragma: no cover - defensive
        LOGGER.error("Failed to execute git command %s: %s", " ".join(command), exc)
        return ""
    if process.returncode != 0:
        LOGGER.debug("git command failed (%s): %s", process.returncode, process.stderr.strip())
        return ""
    return process.stdout


__all__ = ["DiffMetrics", "compute_diff_metrics"]
