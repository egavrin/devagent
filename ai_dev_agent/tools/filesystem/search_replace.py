"""SEARCH/REPLACE block EDIT tool implementation (Aider-style format).

This module implements the SEARCH/REPLACE edit format used by Aider, which is
more familiar to LLMs due to its similarity to git merge conflict markers.

Format:
    path/to/file.py
    ```python
    <<<<<<< SEARCH
    exact content to find
    =======
    replacement content
    >>>>>>> REPLACE
    ```
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from ai_dev_agent.core.utils.logger import get_logger

if TYPE_CHECKING:
    from ai_dev_agent.tools.registry import ToolContext

LOGGER = get_logger(__name__)


# ---------------------------------------------------------------------------
# SEARCH/REPLACE format markers
# ---------------------------------------------------------------------------

SEARCH_MARKER = "<<<<<<< SEARCH"
DIVIDER_MARKER = "======="
REPLACE_MARKER = ">>>>>>> REPLACE"


# ---------------------------------------------------------------------------
# Fuzzy matching helpers (inspired by Aider's editblock_coder.py)
# ---------------------------------------------------------------------------


def find_similar_lines(search_text: str, content: str, threshold: float = 0.6) -> str | None:
    """Find most similar chunk in content for error diagnostics.

    When context matching fails, this helps the LLM understand what went wrong
    by showing what the file actually contains that's closest to what they sent.

    Args:
        search_text: The text the LLM was searching for
        content: The actual file content
        threshold: Minimum similarity ratio (0.0-1.0) to return a match

    Returns:
        A snippet of similar content with context, or None if no match above threshold
    """
    search_lines = search_text.splitlines()
    content_lines = content.splitlines()

    if not search_lines or not content_lines:
        return None

    best_ratio = 0.0
    best_start = -1

    # Slide a window of search_lines size over content_lines
    for i in range(max(1, len(content_lines) - len(search_lines) + 1)):
        chunk = content_lines[i : i + len(search_lines)]
        ratio = SequenceMatcher(None, search_lines, chunk).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_start = i

    if best_ratio < threshold:
        return None

    # Return with context (±3 lines)
    context_lines = 3
    start = max(0, best_start - context_lines)
    end = min(len(content_lines), best_start + len(search_lines) + context_lines)
    return "\n".join(content_lines[start:end])


class PatchFormatError(ValueError):
    """Raised when the edit block is malformed."""


class PatchApplyError(RuntimeError):
    """Raised when an action cannot be applied to the filesystem."""


@dataclass
class EditBlock:
    """Represents a single SEARCH/REPLACE block."""

    path: str
    search: str  # Text to find (empty = new file)
    replace: str  # Text to replace with (empty = delete)


@dataclass
class ParseResult:
    """Result of parsing edit blocks."""

    blocks: list[EditBlock]
    warnings: list[str]


class SearchReplaceParser:
    """Parse SEARCH/REPLACE formatted text into structured edit blocks.

    Parses the Aider-style format:
        path/to/file.py
        ```python
        <<<<<<< SEARCH
        content to find
        =======
        replacement content
        >>>>>>> REPLACE
        ```

    For new files, SEARCH section is empty:
        new_file.py
        ```python
        <<<<<<< SEARCH
        =======
        new content
        >>>>>>> REPLACE
        ```
    """

    # Regex to extract file path before a fenced code block
    # Matches: "path/to/file.ext" or "path/to/file.ext" followed by newline and ```
    PATH_PATTERN = re.compile(
        r"^([^\s`][^\n`]*?\.[\w]+)\s*\n```",
        re.MULTILINE,
    )

    def __init__(self, text: str):
        self.text = text or ""
        self.warnings: list[str] = []

    def parse(self) -> list[EditBlock]:
        """Parse the text to edit blocks."""
        return self.parse_with_warnings().blocks

    def parse_with_warnings(self) -> ParseResult:
        """Parse edit blocks, returning any warnings."""
        blocks: list[EditBlock] = []
        self.warnings = []

        if not self.text.strip():
            raise PatchFormatError("Edit input is empty.")

        # Find all SEARCH/REPLACE blocks
        # Pattern: path, then fenced block containing SEARCH/REPLACE markers
        block_pattern = re.compile(
            r"^([^\s`\n][^\n`]*?)\s*\n"  # File path on its own line
            r"```[\w]*\s*\n"  # Opening fence with optional language
            r"<<<<<<< SEARCH\s*\n"  # SEARCH marker
            r"(.*?)"  # SEARCH content (can be empty)
            r"=======\s*\n"  # Divider
            r"(.*?)"  # REPLACE content (can be empty)
            r">>>>>>> REPLACE\s*\n?"  # REPLACE marker
            r"```",  # Closing fence
            re.MULTILINE | re.DOTALL,
        )

        matches = list(block_pattern.finditer(self.text))

        if not matches:
            # Try without fences (some LLMs omit them)
            unfenced_pattern = re.compile(
                r"^([^\s<\n][^\n<]*?)\s*\n"  # File path
                r"<<<<<<< SEARCH\s*\n"  # SEARCH marker
                r"(.*?)"  # SEARCH content
                r"=======\s*\n"  # Divider
                r"(.*?)"  # REPLACE content
                r">>>>>>> REPLACE",  # REPLACE marker
                re.MULTILINE | re.DOTALL,
            )
            matches = list(unfenced_pattern.finditer(self.text))
            if matches:
                self.warnings.append("Parsed SEARCH/REPLACE blocks without fence markers")

        if not matches:
            raise PatchFormatError(
                "No SEARCH/REPLACE blocks found.\n\n"
                "Expected format:\n"
                "path/to/file.py\n"
                "```python\n"
                "<<<<<<< SEARCH\n"
                "content to find\n"
                "=======\n"
                "replacement content\n"
                ">>>>>>> REPLACE\n"
                "```"
            )

        for match in matches:
            path = match.group(1).strip()
            search = match.group(2)
            replace = match.group(3)

            # Normalize: remove trailing newline from search/replace if present
            # but preserve internal content
            if search.endswith("\n"):
                search = search[:-1]
            if replace.endswith("\n"):
                replace = replace[:-1]

            blocks.append(EditBlock(path=path, search=search, replace=replace))

        return ParseResult(blocks=blocks, warnings=self.warnings)


class EditBlockApplier:
    """Apply parsed SEARCH/REPLACE blocks to the filesystem."""

    def __init__(self, repo_root: Path):
        self.repo_root = Path(repo_root)
        self._warnings: list[str] = []

    def apply(self, blocks: Sequence[EditBlock], dry_run: bool = False) -> dict[str, Any]:
        """Apply edit blocks to files.

        Args:
            blocks: List of EditBlock instances
            dry_run: If True, validate without writing

        Returns:
            Dict with success status, errors, warnings, and file lists
        """
        errors: list[str] = []
        changed_files: set[str] = set()
        new_files: set[str] = set()
        applied_count = 0
        self._warnings = []

        for idx, block in enumerate(blocks, start=1):
            try:
                is_new = self._apply_block(block, changed_files, new_files, dry_run, idx)
                if is_new:
                    new_files.add(block.path)
                else:
                    changed_files.add(block.path)
                applied_count += 1
            except PatchApplyError as exc:
                errors.append(str(exc))
                LOGGER.warning("Failed to apply block %d: %s", idx, exc)

        success = not errors
        return {
            "success": success,
            "changes_applied": applied_count if success else 0,
            "errors": errors,
            "warnings": self._warnings,
            "changed_files": sorted(changed_files - new_files),
            "new_files": sorted(new_files),
        }

    def _apply_block(
        self,
        block: EditBlock,
        changed_files: set[str],
        new_files: set[str],
        dry_run: bool,
        block_idx: int,
    ) -> bool:
        """Apply a single edit block.

        Returns:
            True if this created a new file, False if it modified existing
        """
        target = self._resolve_path(block.path)

        # Empty SEARCH = new file creation
        if not block.search:
            if target.exists():
                # File exists - treat as append to end of file
                content = target.read_text(encoding="utf-8")
                # Ensure proper newline separation
                if content and not content.endswith("\n"):
                    content += "\n"
                new_content = content + block.replace
                if not new_content.endswith("\n"):
                    new_content += "\n"
                if not dry_run:
                    target.write_text(new_content, encoding="utf-8")
                    LOGGER.info("Appended to file %s", block.path)
                return False
            else:
                # Create new file
                if not dry_run:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    content = block.replace
                    if content and not content.endswith("\n"):
                        content += "\n"
                    target.write_text(content, encoding="utf-8")
                    LOGGER.info("Created file %s", block.path)
                return True

        # Empty REPLACE = delete content (or whole file if SEARCH matches all)
        if not block.replace and block.search:
            if not target.exists():
                raise PatchApplyError(f"Block {block_idx}: Cannot edit missing file: {block.path}")
            content = target.read_text(encoding="utf-8")

            # Find and remove the search content
            match_result = self._find_content_fuzzy(content, block.search, block.path, block_idx)
            if match_result is None:
                self._raise_search_not_found(block.search, content, block.path, block_idx)

            pos, end, _, warning = match_result
            if warning:
                self._warnings.append(f"Block {block_idx} in '{block.path}': {warning}")

            new_content = content[:pos] + content[end:]
            if not dry_run:
                target.write_text(new_content, encoding="utf-8")
                LOGGER.info("Removed content from file %s", block.path)
            return False

        # Normal replacement
        if not target.exists():
            raise PatchApplyError(f"Block {block_idx}: Cannot edit missing file: {block.path}")

        content = target.read_text(encoding="utf-8")
        match_result = self._find_content_fuzzy(content, block.search, block.path, block_idx)
        if match_result is None:
            self._raise_search_not_found(block.search, content, block.path, block_idx)

        pos, end, _, warning = match_result
        if warning:
            self._warnings.append(f"Block {block_idx} in '{block.path}': {warning}")

        new_content = content[:pos] + block.replace + content[end:]
        if not dry_run:
            target.write_text(new_content, encoding="utf-8")
            LOGGER.info("Updated file %s", block.path)
        return False

    def _find_content_fuzzy(
        self,
        content: str,
        search: str,
        path: str,
        block_idx: int,
    ) -> tuple[int, int, str, str | None] | None:
        """Find search text in content using layered fuzzy matching.

        Tries matching strategies in order of strictness:
        1. Exact match
        2. Trailing whitespace tolerance
        3. Leading whitespace tolerance

        Returns:
            (start_pos, end_pos, matched_text, warning) or None if no match
        """
        # Layer 1: Exact match
        pos = content.find(search)
        if pos != -1:
            return pos, pos + len(search), search, None

        search_lines = search.splitlines()

        # Layer 2: Trailing whitespace tolerance
        match = self._find_with_trailing_ws_tolerance(content, search_lines)
        if match:
            pos, end, matched = match
            return pos, end, matched, "Matched after stripping trailing whitespace"

        # Layer 3: Leading whitespace tolerance
        match = self._find_with_leading_ws_tolerance(content, search_lines)
        if match:
            pos, end, matched = match
            return pos, end, matched, "Matched after adjusting indentation"

        return None

    def _find_with_trailing_ws_tolerance(
        self, content: str, search_lines: list[str]
    ) -> tuple[int, int, str] | None:
        """Find match with trailing whitespace stripped from each line."""
        content_lines = content.splitlines(keepends=True)
        stripped_search = [line.rstrip() for line in search_lines]

        for i in range(max(1, len(content_lines) - len(search_lines) + 1)):
            chunk = content_lines[i : i + len(search_lines)]
            chunk_stripped = [line.rstrip() for line in chunk]

            if stripped_search == chunk_stripped:
                char_pos = sum(len(content_lines[j]) for j in range(i))
                char_end = char_pos + sum(len(line) for line in chunk)
                matched_text = "".join(chunk).rstrip("\n")
                return char_pos, char_end, matched_text

        return None

    def _find_with_leading_ws_tolerance(
        self, content: str, search_lines: list[str]
    ) -> tuple[int, int, str] | None:
        """Find match ignoring uniform leading whitespace differences."""
        content_lines = content.splitlines(keepends=True)

        leading = [len(line) - len(line.lstrip()) for line in search_lines if line.strip()]
        if not leading:
            return None
        min_leading = min(leading)

        stripped_search = [
            line[min_leading:].rstrip() if line.strip() else line.rstrip() for line in search_lines
        ]

        for i in range(max(1, len(content_lines) - len(search_lines) + 1)):
            chunk = content_lines[i : i + len(search_lines)]
            chunk_stripped = [line.lstrip().rstrip() for line in chunk]
            search_stripped_content = [line.lstrip().rstrip() for line in stripped_search]

            if chunk_stripped == search_stripped_content:
                char_pos = sum(len(content_lines[j]) for j in range(i))
                char_end = char_pos + sum(len(line) for line in chunk)
                matched_text = "".join(chunk).rstrip("\n")
                return char_pos, char_end, matched_text

        return None

    def _raise_search_not_found(self, search: str, content: str, path: str, block_idx: int) -> None:
        """Raise a helpful error when SEARCH content not found."""
        similar = find_similar_lines(search, content, threshold=0.5)

        search_preview = search[:200] + "..." if len(search) > 200 else search
        error_parts = [f"Block {block_idx}: SEARCH content not found in '{path}'."]
        error_parts.append(f"\n\nYour SEARCH block contains:\n```\n{search_preview}\n```")

        if similar:
            error_parts.append(f"\n\nDid you mean to match this?\n```\n{similar}\n```")
        else:
            content_preview = "\n".join(content.splitlines()[:20])
            if len(content.splitlines()) > 20:
                content_preview += "\n..."
            error_parts.append(
                f"\n\nActual file content (first 20 lines):\n```\n{content_preview}\n```"
            )

        # Detect insertion attempt with invented content
        if self._looks_like_invented_content(search, content):
            error_parts.append(
                "\n\n**TIP: For INSERTIONS (adding new content), use an empty SEARCH block:**\n"
                f"```\n{path}\n```\n"
                "<<<<<<< SEARCH\n"
                "=======\n"
                "your new content here\n"
                ">>>>>>> REPLACE\n"
                "```\n"
            )

        error_parts.append(
            "\n\n**IMPORTANT:** The SEARCH content must EXACTLY match the file. "
            "Use READ tool first to see the actual content."
        )
        raise PatchApplyError("".join(error_parts))

    def _looks_like_invented_content(self, search: str, content: str) -> bool:
        """Detect if SEARCH content looks invented (not from the file)."""
        search_lines = search.strip().splitlines()
        if not search_lines:
            return False

        # Check if any search line actually exists in the file
        for line in search_lines:
            stripped = line.strip()
            if stripped and stripped in content:
                return False

        # No lines match - likely invented
        return True

    def _resolve_path(self, relative: str) -> Path:
        candidate = (self.repo_root / relative).resolve()
        repo_root = self.repo_root.resolve()
        try:
            candidate.relative_to(repo_root)
        except ValueError as exc:
            raise PatchApplyError(f"Path '{relative}' escapes repository root.") from exc
        return candidate


def _fs_edit(payload: Mapping[str, Any], context: "ToolContext") -> Mapping[str, Any]:
    """Filesystem edit handler that consumes SEARCH/REPLACE format.

    Expected payload:
        {"patch": "<SEARCH/REPLACE blocks>"}

    Format:
        path/to/file.py
        ```python
        <<<<<<< SEARCH
        content to find
        =======
        replacement content
        >>>>>>> REPLACE
        ```
    """
    patch_text = payload.get("patch")
    if not patch_text or not isinstance(patch_text, str):
        return {
            "success": False,
            "changes_applied": 0,
            "errors": ["Missing required parameter: patch"],
            "warnings": [],
            "changed_files": [],
            "new_files": [],
        }

    all_warnings: list[str] = []

    try:
        parser = SearchReplaceParser(patch_text)
        parse_result = parser.parse_with_warnings()
        blocks = parse_result.blocks
        all_warnings.extend(parse_result.warnings)
    except PatchFormatError as exc:
        return {
            "success": False,
            "changes_applied": 0,
            "errors": [str(exc)],
            "warnings": [],
            "changed_files": [],
            "new_files": [],
        }

    applier = EditBlockApplier(context.repo_root)
    validation = applier.apply(blocks, dry_run=True)
    if not validation["success"]:
        validation["warnings"] = all_warnings + validation.get("warnings", [])
        return validation

    result = applier.apply(blocks, dry_run=False)
    result["warnings"] = all_warnings + result.get("warnings", [])
    return result


__all__ = [
    "EditBlock",
    "EditBlockApplier",
    "ParseResult",
    "PatchApplyError",
    "PatchFormatError",
    "SearchReplaceParser",
    "_fs_edit",
    "find_similar_lines",
]
