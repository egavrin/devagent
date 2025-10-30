"""Enhanced utilities for diff validation, preview, and application."""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ai_dev_agent.core.utils.logger import get_logger

LOGGER = get_logger(__name__)

DIFF_BLOCK = re.compile(r"```diff\s*(?P<diff>.*?)```", re.DOTALL)


class DiffError(RuntimeError):
    """Raised when diff extraction or application fails."""


@dataclass
class DiffValidationResult:
    """Result of diff validation."""

    is_valid: bool
    errors: list[str]
    warnings: list[str]
    affected_files: list[str]
    lines_added: int
    lines_removed: int

    @property
    def has_issues(self) -> bool:
        return bool(self.errors or self.warnings)


@dataclass
class DiffPreview:
    """Rich diff preview with syntax highlighting and context."""

    original_diff: str
    formatted_diff: str
    summary: str
    file_changes: dict[str, dict[str, int]]  # file -> {added, removed, modified}
    validation_result: DiffValidationResult


class DiffProcessor:
    """Enhanced diff processing with validation and preview capabilities."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root

    @staticmethod
    def _normalize_diff_path(path: str) -> str:
        """Normalize diff paths by stripping leading prefixes."""
        if path.startswith("a/") or path.startswith("b/"):
            return path[2:]
        return path

    def extract_and_validate_diff(self, text: str) -> tuple[str, DiffValidationResult]:
        """Extract diff from text and validate it."""
        diff_text = self._extract_diff(text)
        validation = self._validate_diff(diff_text)
        return diff_text, validation

    def create_preview(self, diff_text: str) -> DiffPreview:
        """Create a rich preview of the diff."""
        validation = self._validate_diff(diff_text)
        formatted_diff = self._format_diff_for_display(diff_text)
        summary = self._create_diff_summary(diff_text, validation)
        file_changes = self._analyze_file_changes(diff_text)

        return DiffPreview(
            original_diff=diff_text,
            formatted_diff=formatted_diff,
            summary=summary,
            file_changes=file_changes,
            validation_result=validation,
        )

    def apply_diff_safely(self, diff_text: str) -> bool:
        """Apply diff with safety checks before applying it."""
        validation = self._validate_diff(diff_text)

        if validation.errors:
            raise DiffError(f"Diff validation failed: {'; '.join(validation.errors)}")
        return self._apply_diff(diff_text)

    def _extract_diff(self, text: str) -> str:
        """Extract diff from text with enhanced parsing."""
        # Try to find diff in code block first
        match = DIFF_BLOCK.search(text)
        if match:
            diff_text = match.group("diff").strip("\n")
        else:
            # Look for diff markers in the text
            lines = text.splitlines()
            diff_start = None
            diff_end = None

            for i, line in enumerate(lines):
                if line.startswith("--- ") or line.startswith("diff --git"):
                    diff_start = i
                    break

            if diff_start is not None:
                # Find the end of the diff
                for i in range(diff_start + 1, len(lines)):
                    if lines[i].startswith("--- ") and i > diff_start + 3:  # New diff starts
                        diff_end = i
                        break

                if diff_end is None:
                    diff_end = len(lines)

                diff_text = "\n".join(lines[diff_start:diff_end]).strip("\n")
            else:
                diff_text = text.strip("\n")

        # Validate basic diff structure
        if "---" not in diff_text or "+++" not in diff_text:
            raise DiffError("No unified diff found in response. Expected '---' and '+++' markers.")

        # Normalize line endings and trailing whitespace
        normalized = [line.rstrip() for line in diff_text.splitlines()]
        diff_text = "\n".join(normalized) + "\n"

        return diff_text

    def _validate_diff(self, diff_text: str) -> DiffValidationResult:
        """Validate diff structure and content."""
        errors = []
        warnings = []
        affected_files = []
        lines_added = 0
        lines_removed = 0

        lines = diff_text.splitlines()
        current_file = None
        in_hunk = False
        hunk_context = 0
        last_old_file: str | None = None

        for line_num, line in enumerate(lines, 1):
            # File headers
            if line.startswith("--- "):
                old_file_raw = line[4:].strip()
                if old_file_raw != "/dev/null":
                    last_old_file = self._normalize_diff_path(old_file_raw)
                    file_path = Path(last_old_file)
                    if not file_path.is_absolute():
                        full_path = self.repo_root / file_path
                        if not full_path.exists() and last_old_file not in affected_files:
                            warnings.append(f"File {last_old_file} does not exist (new file?)")
                else:
                    last_old_file = None
                in_hunk = False

            elif line.startswith("+++ "):
                new_file_raw = line[4:].strip()
                if new_file_raw != "/dev/null":
                    current_file = self._normalize_diff_path(new_file_raw)
                    if current_file not in affected_files:
                        affected_files.append(current_file)
                else:
                    current_file = last_old_file
                    if current_file and current_file not in affected_files:
                        affected_files.append(current_file)

            elif line.startswith("@@"):
                # Hunk header validation
                hunk_match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
                if not hunk_match:
                    errors.append(f"Invalid hunk header at line {line_num}: {line}")
                else:
                    in_hunk = True
                    hunk_context = 0

            elif in_hunk:
                if line.startswith("+") and not line.startswith("+++"):
                    lines_added += 1
                elif line.startswith("-") and not line.startswith("---"):
                    lines_removed += 1
                elif line.startswith(" "):
                    hunk_context += 1
                elif line.strip() == "":
                    # Empty line in hunk context
                    pass
                else:
                    warnings.append(
                        f"Unexpected line format in hunk at line {line_num}: {line[:50]}..."
                    )

        # Additional validation checks
        if not affected_files:
            errors.append("No files found in diff")

        if lines_added == 0 and lines_removed == 0:
            warnings.append("Diff appears to have no actual changes")

        # Check for extremely large diffs
        if lines_added + lines_removed > 1000:
            warnings.append(f"Large diff detected: {lines_added + lines_removed} lines changed")

        return DiffValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            affected_files=affected_files,
            lines_added=lines_added,
            lines_removed=lines_removed,
        )

    def _format_diff_for_display(self, diff_text: str) -> str:
        """Format diff for better display with color coding hints."""
        lines = diff_text.splitlines()
        formatted_lines = []

        for line in lines:
            if line.startswith("+++"):
                formatted_lines.append(f"🟢 {line}")
            elif line.startswith("---"):
                formatted_lines.append(f"🔴 {line}")
            elif line.startswith("@@"):
                formatted_lines.append(f"🔵 {line}")
            elif line.startswith("+"):
                formatted_lines.append(f"+ {line[1:]}")
            elif line.startswith("-"):
                formatted_lines.append(f"- {line[1:]}")
            elif line.startswith(" "):
                formatted_lines.append(f"  {line[1:]}")
            else:
                formatted_lines.append(line)

        return "\n".join(formatted_lines)

    def _create_diff_summary(self, diff_text: str, validation: DiffValidationResult) -> str:
        """Create a human-readable summary of the diff."""
        summary_parts = []

        # File summary
        file_count = len(validation.affected_files)
        if file_count == 1:
            summary_parts.append("1 file modified")
        else:
            summary_parts.append(f"{file_count} files modified")

        # Change summary
        changes = []
        if validation.lines_added > 0:
            changes.append(f"+{validation.lines_added} lines added")
        if validation.lines_removed > 0:
            changes.append(f"-{validation.lines_removed} lines removed")

        if changes:
            summary_parts.append(f"({', '.join(changes)})")

        # Validation status
        if validation.errors:
            summary_parts.append(f"⚠️  {len(validation.errors)} errors")
        if validation.warnings:
            summary_parts.append(f"🔶 {len(validation.warnings)} warnings")

        return " • ".join(summary_parts)

    def _analyze_file_changes(self, diff_text: str) -> dict[str, dict[str, int]]:
        """Analyze changes per file."""
        file_changes: dict[str, dict[str, int]] = {}
        current_file: str | None = None
        last_old_file: str | None = None

        for line in diff_text.splitlines():
            if line.startswith("+++ "):
                new_file_raw = line[4:].strip()
                if new_file_raw != "/dev/null":
                    current_file = self._normalize_diff_path(new_file_raw)
                else:
                    current_file = last_old_file
                if current_file and current_file not in file_changes:
                    file_changes[current_file] = {"added": 0, "removed": 0, "modified": 0}
            elif line.startswith("--- "):
                old_file_raw = line[4:].strip()
                if old_file_raw != "/dev/null":
                    last_old_file = self._normalize_diff_path(old_file_raw)
                else:
                    last_old_file = None

            elif current_file and line.startswith("+") and not line.startswith("+++"):
                file_changes[current_file]["added"] += 1

            elif current_file and line.startswith("-") and not line.startswith("---"):
                file_changes[current_file]["removed"] += 1

        # Calculate modified lines (lines that are both added and removed)
        for file_data in file_changes.values():
            file_data["modified"] = min(file_data["added"], file_data["removed"])

        return file_changes

    def _apply_diff(self, diff_text: str) -> bool:
        """Apply the diff using git apply with fallback to patch."""
        try:
            # Try git apply first
            process = subprocess.run(
                ["git", "apply", "--allow-empty", "--unidiff-zero", "--whitespace=fix", "-"],
                input=diff_text.encode("utf-8"),
                cwd=str(self.repo_root),
                capture_output=True,
                timeout=30,
            )

            if process.returncode == 0:
                LOGGER.info("Diff applied successfully using git apply")
                return True

            LOGGER.debug("git apply failed: %s", process.stderr.decode("utf-8"))

            # Fallback to patch command
            patch_process = subprocess.run(
                ["patch", "-p1", "--forward", "--batch"],
                input=diff_text.encode("utf-8"),
                cwd=str(self.repo_root),
                capture_output=True,
                timeout=30,
            )

            if patch_process.returncode == 0:
                LOGGER.info("Diff applied successfully using patch")
                return True

            if self._apply_simple_fallback(diff_text):
                LOGGER.info("Diff applied using simplified Python fallback")
                return True

            # Both methods failed
            error_msg = f"git apply: {process.stderr.decode('utf-8')}\npatch: {patch_process.stderr.decode('utf-8')}"
            raise DiffError(f"Failed to apply diff:\n{error_msg}")

        except subprocess.TimeoutExpired:
            raise DiffError("Diff application timed out")
        except Exception as exc:
            raise DiffError(f"Unexpected error applying diff: {exc}")

    def _apply_simple_fallback(self, diff_text: str) -> bool:
        """Attempt to handle simple diff operations (currently deletions)."""
        lines = diff_text.splitlines()

        # Detect if diff contains additions; if so, we cannot safely process here
        has_additions = any(line.startswith("+") and not line.startswith("+++") for line in lines)
        if has_additions:
            return False

        deletions = self._collect_deletion_targets(lines)
        if not deletions:
            return False

        success = False
        for rel_path in deletions:
            path = (self.repo_root / rel_path).resolve()
            try:
                if path.is_file():
                    path.unlink()
                    LOGGER.info("Removed file via fallback: %s", rel_path)
                    success = True
                elif path.is_dir():
                    shutil.rmtree(path)
                    LOGGER.info("Removed directory via fallback: %s", rel_path)
                    success = True
                else:
                    LOGGER.warning("Fallback deletion target not found: %s", rel_path)
            except OSError as exc:
                LOGGER.error("Failed to remove %s via fallback: %s", rel_path, exc)
                raise DiffError(f"Fallback deletion failed for {rel_path}: {exc}") from exc

        return success

    def _collect_deletion_targets(self, lines: list[str]) -> list[str]:
        """Collect files marked for deletion in the diff."""
        deletions: list[str] = []
        current_old: str | None = None

        for line in lines:
            if line.startswith("--- "):
                old_candidate = line[4:].strip()
                if old_candidate != "/dev/null":
                    current_old = self._normalize_diff_path(old_candidate)
                else:
                    current_old = None
            elif line.startswith("+++ "):
                new_candidate = line[4:].strip()
                if new_candidate == "/dev/null" and current_old:
                    deletions.append(current_old)
                    current_old = None
                else:
                    current_old = None

        return deletions


# Legacy functions for backward compatibility
def extract_diff(text: str) -> str:
    """Legacy function for extracting diff."""
    processor = DiffProcessor(Path.cwd())
    diff_text, _ = processor.extract_and_validate_diff(text)
    return diff_text


def apply_patch(diff_text: str, repo_root: Path) -> None:
    """Legacy function for applying patches."""
    processor = DiffProcessor(repo_root)
    if not processor.apply_diff_safely(diff_text):
        raise DiffError("Failed to apply patch")


__all__ = [
    "DiffError",
    "DiffPreview",
    "DiffProcessor",
    "DiffValidationResult",
    "apply_patch",
    "extract_diff",
]
