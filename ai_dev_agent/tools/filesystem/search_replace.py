"""SEARCH/REPLACE format file editing tool implementation.

This module provides an alternative to unified diff format that's easier for LLMs to use.
Based on successful implementations in Cline and Aider.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Mapping, Optional, Tuple

from ai_dev_agent.core.utils.logger import get_logger

if TYPE_CHECKING:
    from ai_dev_agent.tools.registry import ToolContext

LOGGER = get_logger(__name__)


def _resolve_path(repo_root: Path, relative: str) -> Path:
    """Resolve a path relative to repo root with security checks.

    Args:
        repo_root: The repository root directory
        relative: A relative or absolute path

    Returns:
        Resolved absolute path

    Raises:
        ValueError: If the path would escape the repository root
    """
    # Convert to Path if string
    rel_path = Path(relative)

    # If absolute path, check it's under repo_root
    if rel_path.is_absolute():
        candidate = rel_path.resolve()
    else:
        # For relative paths, join with repo_root
        candidate = repo_root / rel_path
        # For files that don't exist yet, resolve parent then add filename
        if not candidate.exists():
            parent = candidate.parent.resolve()
            candidate = parent / candidate.name
        else:
            candidate = candidate.resolve()

    # Check if path is within repo_root
    try:
        candidate.relative_to(repo_root.resolve())
    except ValueError:
        raise ValueError(f"Path '{relative}' escapes repository root")

    return candidate


@dataclass
class SearchReplaceBlock:
    """Represents a single SEARCH/REPLACE operation."""

    search: str
    replace: str
    start_pos: int = -1
    end_pos: int = -1


class SearchReplaceMatcher:
    """Implements three-tier matching strategy for SEARCH/REPLACE blocks."""

    def __init__(self, file_content: str):
        """Initialize matcher with file content.

        Args:
            file_content: The current content of the file being edited
        """
        self.file_content = file_content
        self.content_lines = file_content.splitlines(keepends=True)

    def find_match(self, search_text: str, start_position: int = 0) -> Optional[Tuple[int, int]]:
        """Find the position of search_text in file_content.

        Uses a three-tier matching strategy:
        1. Exact match (fastest, most reliable)
        2. Line-trimmed match (handles whitespace issues)
        3. Anchor match (for blocks with 3+ lines)

        Args:
            search_text: The text to search for
            start_position: Position to start searching from

        Returns:
            Tuple of (start_index, end_index) if found, None otherwise
        """
        # Tier 1: Exact match
        exact_pos = self.file_content.find(search_text, start_position)
        if exact_pos != -1:
            LOGGER.debug("Found exact match at position %d", exact_pos)
            return (exact_pos, exact_pos + len(search_text))

        # Tier 2: Line-trimmed match (handles whitespace differences)
        line_match = self._line_trimmed_match(search_text, start_position)
        if line_match:
            LOGGER.debug("Found line-trimmed match at lines %s", line_match)
            return line_match

        # Tier 3: Anchor match (first and last lines)
        anchor_match = self._anchor_match(search_text, start_position)
        if anchor_match:
            LOGGER.debug("Found anchor match at position %s", anchor_match)
            return anchor_match

        return None

    def _line_trimmed_match(
        self, search_text: str, start_pos: int = 0
    ) -> Optional[Tuple[int, int]]:
        """Match ignoring leading/trailing whitespace per line.

        Args:
            search_text: Text to search for
            start_pos: Position to start searching from

        Returns:
            Tuple of (start_index, end_index) if found, None otherwise
        """
        search_lines = search_text.splitlines()
        if not search_lines:
            return None

        # Create trimmed versions for comparison
        search_trimmed = [line.strip() for line in search_lines]

        # Convert start_pos to line number
        chars_seen = 0
        start_line = 0
        for i, line in enumerate(self.content_lines):
            if chars_seen >= start_pos:
                start_line = i
                break
            chars_seen += len(line)

        # Search for matching lines
        for i in range(start_line, len(self.content_lines) - len(search_lines) + 1):
            window = self.content_lines[i : i + len(search_lines)]
            window_trimmed = [line.strip() for line in window]

            if search_trimmed == window_trimmed:
                # Calculate character positions
                start_char = sum(len(line) for line in self.content_lines[:i])
                end_char = sum(len(line) for line in self.content_lines[: i + len(search_lines)])
                return (start_char, end_char)

        return None

    def _anchor_match(self, search_text: str, start_pos: int = 0) -> Optional[Tuple[int, int]]:
        """Match using first and last lines as anchors.

        Only works for blocks with 3 or more lines.

        Args:
            search_text: Text to search for
            start_pos: Position to start searching from

        Returns:
            Tuple of (start_index, end_index) if found, None otherwise
        """
        lines = search_text.splitlines()
        if len(lines) < 3:
            return None

        first_line = lines[0]
        last_line = lines[-1]

        # Find all positions where first line appears
        pos = start_pos
        while True:
            first_pos = self.file_content.find(first_line, pos)
            if first_pos == -1:
                break

            # Check if last line appears at the right distance
            expected_end = first_pos + len(search_text)
            last_line_start = expected_end - len(last_line)

            if (
                last_line_start >= 0
                and self.file_content[last_line_start:expected_end] == last_line
            ):
                # Verify the content between anchors has the same line count
                between_content = self.file_content[first_pos + len(first_line) : last_line_start]
                if between_content.count("\n") == len(lines) - 2:
                    return (first_pos, expected_end)

            pos = first_pos + 1

        return None


def parse_search_replace_blocks(changes_text: str) -> List[SearchReplaceBlock]:
    """Parse SEARCH/REPLACE blocks from input text.

    Supports multiple formats:
    - <<<<<<< SEARCH / ======= / >>>>>>> REPLACE (Aider style)
    - ------- SEARCH / ======= / +++++++ REPLACE (Cline style)

    Args:
        changes_text: Text containing SEARCH/REPLACE blocks

    Returns:
        List of SearchReplaceBlock objects

    Raises:
        ValueError: If the format is invalid
    """
    blocks = []
    lines = changes_text.splitlines(keepends=True)

    # Regex patterns for different marker styles
    search_start_pattern = re.compile(r"^(<{3,}|---+)\s*SEARCH\s*$")
    separator_pattern = re.compile(r"^={3,}\s*$")
    replace_end_pattern = re.compile(r"^(>{3,}|\+{3,})\s*REPLACE\s*$")

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        # Look for SEARCH marker
        if search_start_pattern.match(line):
            search_lines = []
            i += 1

            # Collect lines until separator
            while i < len(lines):
                line = lines[i].rstrip()
                if separator_pattern.match(line):
                    break
                search_lines.append(lines[i])
                i += 1
            else:
                raise ValueError(f"Missing separator '=======' after SEARCH block at line {i}")

            # Now collect REPLACE lines
            replace_lines = []
            i += 1

            while i < len(lines):
                line = lines[i].rstrip()
                if replace_end_pattern.match(line):
                    break
                replace_lines.append(lines[i])
                i += 1
            else:
                raise ValueError(f"Missing REPLACE end marker after separator at line {i}")

            # Create block
            search_text = "".join(search_lines).rstrip("\n")
            replace_text = "".join(replace_lines).rstrip("\n")
            blocks.append(SearchReplaceBlock(search=search_text, replace=replace_text))

        i += 1

    return blocks


def apply_replacements(
    file_path: Path, blocks: List[SearchReplaceBlock]
) -> Tuple[str, List[str], List[str]]:
    """Apply multiple SEARCH/REPLACE blocks to a file.

    Args:
        file_path: Path to the file to edit
        blocks: List of SearchReplaceBlock objects

    Returns:
        Tuple of (modified_content, applied_blocks, errors)
    """
    # Handle new file creation
    if not file_path.exists():
        # For new files, we only support empty search block (entire file replacement)
        # or a single block with any content
        if len(blocks) == 1:
            if not blocks[0].search:
                # Empty search = full file content
                return blocks[0].replace, ["Created new file"], []
            else:
                # Non-empty search for new file - just use replace content
                return blocks[0].replace, ["Created new file with initial content"], []
        else:
            return "", [], ["Cannot apply multiple SEARCH/REPLACE blocks to a non-existent file"]

    content = file_path.read_text(encoding="utf-8")
    matcher = SearchReplaceMatcher(content)

    applied = []
    errors = []
    last_position = 0

    # Track replacements to handle overlaps and out-of-order blocks
    replacements = []

    for i, block in enumerate(blocks):
        # Handle empty search block (replace entire file)
        if not block.search:
            if i == 0 and len(blocks) == 1:
                # Special case: single empty search block means replace entire file
                return block.replace, ["Replaced entire file contents"], []
            else:
                errors.append(
                    f"Block {i+1}: Empty SEARCH blocks only allowed for full file replacement"
                )
                continue

        # Find the match
        match_pos = matcher.find_match(block.search, last_position)
        if not match_pos:
            # Try searching from the beginning (for out-of-order replacements)
            match_pos = matcher.find_match(block.search, 0)
            if not match_pos:
                errors.append(
                    f"Block {i+1}: SEARCH text not found in file:\n{block.search[:100]}..."
                )
                continue

        block.start_pos, block.end_pos = match_pos
        replacements.append(block)

        # Update last_position for sequential searching
        if block.start_pos >= last_position:
            last_position = block.end_pos

    # Sort replacements by position (to handle out-of-order blocks)
    replacements.sort(key=lambda b: b.start_pos, reverse=True)

    # Apply replacements from end to start (to preserve positions)
    for block in replacements:
        content = content[: block.start_pos] + block.replace + content[block.end_pos :]
        applied.append(f"Replaced at position {block.start_pos}-{block.end_pos}")

    return content, applied, errors


def _fs_edit(payload: Mapping[str, Any], context: ToolContext) -> Mapping[str, Any]:
    """Handler for SEARCH/REPLACE format file editing.

    Args:
        payload: Request payload containing 'path' and 'changes'
        context: Tool execution context

    Returns:
        Response dictionary with success status and details
    """
    try:
        # Extract parameters
        file_path = payload.get("path")
        changes = payload.get("changes")

        if not file_path:
            return {
                "success": False,
                "changes_applied": 0,
                "errors": ["Missing required parameter: path"],
                "warnings": [],
            }

        if not changes:
            return {
                "success": False,
                "changes_applied": 0,
                "errors": ["Missing required parameter: changes"],
                "warnings": [],
            }

        # Resolve file path with security checks
        try:
            target_path = _resolve_path(context.repo_root, file_path)
        except ValueError as e:
            return {
                "success": False,
                "changes_applied": 0,
                "errors": [str(e)],
                "warnings": [],
            }

        # Check if file exists - if not, we'll create it
        creating_new_file = not target_path.exists()
        if creating_new_file:
            LOGGER.info(f"File {file_path} doesn't exist, will create it")

        # Parse SEARCH/REPLACE blocks
        try:
            blocks = parse_search_replace_blocks(changes)
        except ValueError as e:
            return {
                "success": False,
                "changes_applied": 0,
                "errors": [f"Invalid SEARCH/REPLACE format: {e}"],
                "warnings": [],
            }

        if not blocks:
            return {
                "success": False,
                "changes_applied": 0,
                "errors": ["No SEARCH/REPLACE blocks found in changes"],
                "warnings": [],
            }

        # Apply replacements
        new_content, applied, errors = apply_replacements(target_path, blocks)

        # Write the modified content if any changes were applied
        if applied:
            # Ensure parent directory exists for new files
            if creating_new_file:
                target_path.parent.mkdir(parents=True, exist_ok=True)

            target_path.write_text(new_content, encoding="utf-8")

            if creating_new_file:
                LOGGER.info("Created new file: %s", file_path)
            else:
                LOGGER.info(
                    "Applied %d/%d SEARCH/REPLACE blocks to %s",
                    len(applied),
                    len(blocks),
                    file_path,
                )

        # Determine success based on whether all blocks were applied
        success = len(errors) == 0 and len(applied) == len(blocks)

        # Add warnings for partial success
        warnings = []
        if applied and errors:
            warnings.append(f"Partially applied: {len(applied)}/{len(blocks)} blocks succeeded")

        return {
            "success": success,
            "changes_applied": len(applied),
            "errors": errors,
            "warnings": warnings,
        }

    except Exception as e:
        LOGGER.exception("Unexpected error in _fs_edit")
        return {
            "success": False,
            "changes_applied": 0,
            "errors": [f"Unexpected error: {type(e).__name__}: {e}"],
            "warnings": [],
        }
