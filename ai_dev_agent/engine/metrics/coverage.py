"""Patch-level coverage computation."""

from __future__ import annotations

import re
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from ai_dev_agent.core.utils.logger import get_logger

LOGGER = get_logger(__name__)

_HUNK_PATTERN = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


@dataclass
class PatchCoverageResult:
    """Coverage statistics for the modified portion of the codebase."""

    covered_lines: int
    total_lines: int
    ratio: float
    per_file: dict[str, dict[str, list[int]]] = field(default_factory=dict)

    @property
    def uncovered_lines(self) -> int:
        return max(self.total_lines - self.covered_lines, 0)


def compute_patch_coverage(
    repo_root: Path,
    *,
    coverage_xml: Path | None = None,
    compare_ref: str | None = None,
) -> PatchCoverageResult | None:
    """Compute coverage for modified lines using coverage.py XML reports."""

    repo_root = Path(repo_root)
    coverage_path = coverage_xml or repo_root / "coverage.xml"
    if not coverage_path.exists():
        LOGGER.debug("Coverage XML not found at %s", coverage_path)
        return None

    changed_lines = _collect_changed_lines(repo_root, compare_ref)
    if not changed_lines:
        return PatchCoverageResult(covered_lines=0, total_lines=0, ratio=1.0, per_file={})

    coverage_map = _load_coverage_map(coverage_path, repo_root)
    total = 0
    covered = 0
    per_file: dict[str, dict[str, list[int]]] = {}

    for path, lines in changed_lines.items():
        total += len(lines)
        covered_lines = sorted(lines & coverage_map.get(path, set()))
        covered += len(covered_lines)
        uncovered_lines = sorted(lines - coverage_map.get(path, set()))
        per_file[path] = {
            "covered": covered_lines,
            "uncovered": uncovered_lines,
        }

    ratio = covered / total if total else 1.0
    return PatchCoverageResult(
        covered_lines=covered,
        total_lines=total,
        ratio=ratio,
        per_file=per_file,
    )


def _collect_changed_lines(repo_root: Path, compare_ref: str | None) -> dict[str, set[int]]:
    ref = compare_ref or "HEAD"
    command = ["git", "diff", ref, "--unified=0", "--no-color"]
    try:
        process = subprocess.run(
            command,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:  # pragma: no cover - defensive
        LOGGER.error("Failed to run git diff for coverage: %s", exc)
        return {}
    if process.returncode != 0:
        LOGGER.debug("git diff returned %s: %s", process.returncode, process.stderr.strip())
        return {}

    changed: dict[str, set[int]] = {}
    current_file: str | None = None

    for line in process.stdout.splitlines():
        if line.startswith("+++ "):
            path = line[4:].strip()
            if path == "/dev/null":
                current_file = None
                continue
            if path.startswith("b/"):
                path = path[2:]
            current_file = path
            changed.setdefault(current_file, set())
            continue
        if line.startswith("@@") and current_file:
            match = _HUNK_PATTERN.match(line)
            if not match:
                continue
            start = int(match.group(3))
            count = int(match.group(4) or "1")
            for number in range(start, start + count):
                changed[current_file].add(number)
    return {path: lines for path, lines in changed.items() if lines}


def _load_coverage_map(coverage_xml: Path, repo_root: Path) -> dict[str, set[int]]:
    try:
        tree = ET.parse(str(coverage_xml))
    except (ET.ParseError, OSError) as exc:
        LOGGER.error("Failed to parse coverage XML %s: %s", coverage_xml, exc)
        return {}

    root = tree.getroot()
    coverage_map: dict[str, set[int]] = {}

    for cls in root.iter("class"):
        filename = cls.get("filename")
        if not filename:
            continue
        normalized = _normalize_path(filename, repo_root)
        hits: set[int] = coverage_map.setdefault(normalized, set())
        for line in cls.findall(".//line"):
            number = line.get("number")
            if not number:
                continue
            try:
                line_number = int(number)
            except ValueError:
                continue
            hits_value = line.get("hits")
            try:
                hit_count = int(hits_value) if hits_value is not None else 0
            except ValueError:
                hit_count = 0
            if hit_count > 0:
                hits.add(line_number)
    return coverage_map


def _normalize_path(path: str, repo_root: Path) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        try:
            candidate = candidate.relative_to(repo_root)
        except ValueError:
            return candidate.as_posix()
    return candidate.as_posix()


__all__ = ["PatchCoverageResult", "compute_patch_coverage"]
