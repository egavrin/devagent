"""Secret detection utilities for changed files."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ai_dev_agent.core.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

LOGGER = get_logger(__name__)

_PATTERNS = {
    "aws_access_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "aws_secret_key": re.compile(r"(?<![A-Z0-9])[A-Za-z0-9/+=]{40}(?![A-Za-z0-9/+=])"),
    "google_api_key": re.compile(r"AIza[0-9A-Za-z\-_]{35}"),
    "slack_token": re.compile(r"xox[baprs]-[0-9A-Za-z-]{10,48}"),
    "generic_secret": re.compile(r"(?i)secret(_?key)?\s*[:=]\s*['\"]?[A-Za-z0-9/+=]{16,}"),
}

_ENTROPY_TOKEN = re.compile(r"['\"]([A-Za-z0-9+/=]{24,})['\"]")


@dataclass
class SecretFinding:
    path: str
    line: int
    detector: str
    snippet: str


@dataclass
class SecretScanResult:
    findings: list[SecretFinding]

    @property
    def count(self) -> int:
        return len(self.findings)


def scan_for_secrets(repo_root: Path, files: Sequence[str]) -> SecretScanResult:
    """Scan target files for known secret patterns and entropy heuristics."""

    repo_root = Path(repo_root)
    findings: list[SecretFinding] = []

    for rel_path in files:
        path = repo_root / rel_path
        if not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            LOGGER.warning("Unable to read %s for secret scan: %s", rel_path, exc)
            continue
        for idx, line in enumerate(content.splitlines(), start=1):
            findings.extend(_scan_line(rel_path, idx, line))
    return SecretScanResult(findings=findings)


def _scan_line(rel_path: str, line_no: int, line: str) -> list[SecretFinding]:
    matches: list[SecretFinding] = []
    for name, pattern in _PATTERNS.items():
        if pattern.search(line):
            snippet = line.strip()
            matches.append(
                SecretFinding(path=rel_path, line=line_no, detector=name, snippet=snippet[:120])
            )
    for entropy_match in _ENTROPY_TOKEN.finditer(line):
        token = entropy_match.group(1)
        if _shannon_entropy(token) >= 4.5:
            snippet = line.strip()
            matches.append(
                SecretFinding(
                    path=rel_path, line=line_no, detector="high_entropy", snippet=snippet[:120]
                )
            )
    return matches


def _shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    probabilities = [text.count(char) / len(text) for char in set(text)]
    return -sum(prob * math.log2(prob) for prob in probabilities if prob > 0)


__all__ = ["SecretFinding", "SecretScanResult", "scan_for_secrets"]
