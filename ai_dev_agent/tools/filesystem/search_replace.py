"""Patch-based EDIT tool implementation using apply_patch style directives."""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from ai_dev_agent.core.utils.logger import get_logger

if TYPE_CHECKING:
    from ai_dev_agent.tools.registry import ToolContext

LOGGER = get_logger(__name__)


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


PATCH_BEGIN = "*** Begin Patch"
PATCH_END = "*** End Patch"
UPDATE_MARKER = "*** Update File:"
ADD_MARKER = "*** Add File:"
DELETE_MARKER = "*** Delete File:"
MOVE_MARKER = "*** Move to:"
EOF_MARKER = "*** End of File"


class PatchFormatError(ValueError):
    """Raised when the patch body is malformed."""


class PatchApplyError(RuntimeError):
    """Raised when an action cannot be applied to the filesystem."""


@dataclass
class PatchChunk:
    """Represents a single @@ chunk in an update action."""

    context: str | None
    old_lines: list[str]
    new_lines: list[str]
    is_end_of_file: bool = False
    operations: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class PatchAction:
    """Represents a single action (add/update/delete)."""

    type: str
    path: str
    move_path: str | None = None
    content: str | None = None
    chunks: list[PatchChunk] = field(default_factory=list)


@dataclass
class ParseResult:
    """Result of parsing a patch, including any auto-correction warnings."""

    actions: list[PatchAction]
    warnings: list[str]


class PatchParser:
    """Parse apply_patch formatted text into structured actions."""

    def __init__(self, text: str):
        self.original_text = text or ""
        self.lines = list(text.splitlines())  # Make mutable copy
        self.index = 0
        self.warnings: list[str] = []

    def parse(self) -> list[PatchAction]:
        """Parse the patch text to actions.

        Note: Use parse_with_warnings() to also get auto-correction warnings.
        """
        result = self.parse_with_warnings()
        return result.actions

    def parse_with_warnings(self) -> ParseResult:
        """Parse the patch text to actions, returning any auto-correction warnings."""
        actions: list[PatchAction] = []
        self.warnings = []

        if not self.lines:
            raise PatchFormatError("Patch is empty.")

        # Skip empty lines until begin sentinel
        while self.index < len(self.lines) and not self.lines[self.index].strip():
            self.index += 1

        if self.index >= len(self.lines) or not self.lines[self.index].startswith(PATCH_BEGIN):
            raise PatchFormatError("Patch must start with '*** Begin Patch'.")
        self.index += 1

        while self.index < len(self.lines):
            line = self.lines[self.index]
            if line.startswith(PATCH_END):
                break
            if not line.strip():
                self.index += 1
                continue

            # Try to auto-correct missing colons before rejecting
            if self._missing_colon(line):
                corrected, warning = self._auto_correct_marker(line)
                if corrected:
                    self.lines[self.index] = corrected
                    line = corrected
                    self.warnings.append(warning)
                    LOGGER.info("Auto-corrected marker: %s", warning)
                else:
                    # Can't auto-correct, raise helpful error
                    expected = self._expected_directive(line)
                    raise PatchFormatError(
                        f"Format error: Missing colon in directive.\n"
                        f"Your line:   {line}\n"
                        f"Required:    {expected} <path>\n"
                        f"             {'~' * (len(expected) - 1)}^ add colon here"
                    )

            if line.startswith(UPDATE_MARKER):
                actions.append(self._parse_update(line))
            elif line.startswith(ADD_MARKER):
                actions.append(self._parse_add(line))
            elif line.startswith(DELETE_MARKER):
                actions.append(self._parse_delete(line))
            else:
                raise PatchFormatError(f"Unknown directive: {line}")

        if not actions:
            raise PatchFormatError("Patch does not contain any actions.")

        return ParseResult(actions=actions, warnings=self.warnings)

    def _auto_correct_marker(self, line: str) -> tuple[str | None, str]:
        """Attempt to auto-correct a marker missing its colon.

        Returns:
            (corrected_line, warning_message) or (None, "") if can't correct
        """
        # Define marker variants without colon and their corrections
        corrections = [
            ("*** Update File ", UPDATE_MARKER),
            ("*** Add File ", ADD_MARKER),
            ("*** Delete File ", DELETE_MARKER),
            ("*** Move to ", MOVE_MARKER),
            # Also handle no space after "File"
            ("*** Update File", UPDATE_MARKER),
            ("*** Add File", ADD_MARKER),
            ("*** Delete File", DELETE_MARKER),
            ("*** Move to", MOVE_MARKER),
        ]

        for wrong_prefix, correct_marker in corrections:
            if line.startswith(wrong_prefix) and not line.startswith(correct_marker):
                # Extract the path part
                path_part = line[len(wrong_prefix) :].strip()
                corrected = f"{correct_marker} {path_part}"
                warning = f"Auto-corrected '{wrong_prefix.strip()}' to '{correct_marker}' for path '{path_part}'"
                return corrected, warning

        return None, ""

    def _parse_path(self, line: str, marker: str) -> str:
        path = line[len(marker) :].strip()
        if not path:
            raise PatchFormatError(f"{marker} directive missing path.")
        return path

    def _parse_update(self, line: str) -> PatchAction:
        path = self._parse_path(line, UPDATE_MARKER)
        self.index += 1
        move_path: str | None = None
        if self.index < len(self.lines):
            move_line = self.lines[self.index]
            if move_line.startswith(MOVE_MARKER):
                move_path = self._parse_path(move_line, MOVE_MARKER)
                self.index += 1
            elif self._looks_like_missing_colon(move_line, MOVE_MARKER):
                # Try to auto-correct the Move to directive
                corrected, warning = self._auto_correct_marker(move_line)
                if corrected:
                    self.lines[self.index] = corrected
                    move_path = self._parse_path(corrected, MOVE_MARKER)
                    self.warnings.append(warning)
                    LOGGER.info("Auto-corrected marker: %s", warning)
                    self.index += 1
                else:
                    raise PatchFormatError(
                        f"Expected '{MOVE_MARKER}' prefix with a colon. Received: {move_line}"
                    )

        chunks: list[PatchChunk] = []
        while self.index < len(self.lines):
            current = self.lines[self.index]
            if current.startswith("***"):
                break
            if not current.strip():
                self.index += 1
                continue
            if not current.startswith("@@"):
                raise PatchFormatError(f"Expected '@@' chunk header, got: {current}")
            chunks.append(self._parse_chunk(current))

        if not chunks:
            raise PatchFormatError(f"Update action for '{path}' is missing @@ chunks.")

        return PatchAction(type="update", path=path, move_path=move_path, chunks=chunks)

    def _parse_chunk(self, header: str) -> PatchChunk:
        context = header[2:].strip() or None
        self.index += 1
        old_lines: list[str] = []
        new_lines: list[str] = []
        operations: list[tuple[str, str]] = []
        is_end_of_file = False

        while self.index < len(self.lines):
            line = self.lines[self.index]
            # Check for EOF marker first (before generic *** check)
            if line == EOF_MARKER:
                is_end_of_file = True
                self.index += 1
                break
            if line.startswith("@@") or line.startswith("***"):
                break

            if not line:
                self.index += 1
                continue

            prefix = line[0]
            content = line[1:] if len(line) > 1 else ""
            if prefix not in (" ", "+", "-"):
                raise PatchFormatError(f"Invalid chunk line: {line}")

            operations.append((prefix, content))
            if prefix == " ":
                old_lines.append(content)
                new_lines.append(content)
            elif prefix == "-":
                old_lines.append(content)
            elif prefix == "+":
                new_lines.append(content)

            self.index += 1

        return PatchChunk(
            context=context,
            old_lines=old_lines,
            new_lines=new_lines,
            is_end_of_file=is_end_of_file,
            operations=operations,
        )

    def _parse_add(self, line: str) -> PatchAction:
        path = self._parse_path(line, ADD_MARKER)
        self.index += 1
        lines: list[str] = []
        while self.index < len(self.lines):
            current = self.lines[self.index]
            if current.startswith("***"):
                break
            if current.startswith("+"):
                lines.append(current[1:])
                self.index += 1
            elif not current.strip():
                lines.append("")
                self.index += 1
            else:
                raise PatchFormatError(f"Invalid line in add block for '{path}': {current}")

        content = ("\n".join(lines) + "\n") if lines else ""
        return PatchAction(type="add", path=path, content=content)

    def _parse_delete(self, line: str) -> PatchAction:
        path = self._parse_path(line, DELETE_MARKER)
        self.index += 1
        return PatchAction(type="delete", path=path)

    def _missing_colon(self, line: str) -> bool:
        """Detect directives that omit the required colon."""
        return any(
            self._looks_like_missing_colon(line, marker)
            for marker in (UPDATE_MARKER, ADD_MARKER, DELETE_MARKER)
        )

    @staticmethod
    def _looks_like_missing_colon(line: str, marker: str) -> bool:
        if not marker.endswith(":"):
            return False
        prefix = marker[:-1]
        return line.startswith(prefix) and not line.startswith(marker)

    @staticmethod
    def _expected_directive(line: str) -> str:
        if line.startswith(UPDATE_MARKER[:-1]):
            return UPDATE_MARKER
        if line.startswith(ADD_MARKER[:-1]):
            return ADD_MARKER
        if line.startswith(DELETE_MARKER[:-1]):
            return DELETE_MARKER
        return "***"


class PatchApplier:
    """Apply parsed actions to the filesystem."""

    def __init__(self, repo_root: Path):
        self.repo_root = Path(repo_root)
        self._warnings: list[str] = []

    def apply(self, actions: Sequence[PatchAction], dry_run: bool = False) -> dict[str, Any]:
        errors: list[str] = []
        changed_files: set[str] = set()
        new_files: set[str] = set()
        applied_count = 0
        self._warnings = []  # Reset warnings for each apply call

        for idx, action in enumerate(actions, start=1):
            try:
                if action.type == "add":
                    self._apply_add(action, new_files, dry_run)
                elif action.type == "delete":
                    self._apply_delete(action, changed_files, dry_run)
                elif action.type == "update":
                    self._apply_update(action, changed_files, dry_run)
                else:
                    raise PatchApplyError(f"Unsupported action type: {action.type}")
                applied_count += 1
            except PatchApplyError as exc:
                errors.append(str(exc))
                LOGGER.warning("Failed to apply action %d: %s", idx, exc)

        success = not errors
        return {
            "success": success,
            "changes_applied": applied_count if not success else (0 if dry_run else len(actions)),
            "errors": errors,
            "warnings": self._warnings,
            "changed_files": sorted(changed_files),
            "new_files": sorted(new_files),
        }

    def _apply_add(self, action: PatchAction, new_files: set[str], dry_run: bool) -> None:
        target = self._resolve_path(action.path)
        if target.exists():
            raise PatchApplyError(f"Cannot create new file '{action.path}': File already exists.")
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(action.content or "", encoding="utf-8")
        new_files.add(action.path)
        if not dry_run:
            LOGGER.info("Created file %s", action.path)

    def _apply_delete(self, action: PatchAction, changed_files: set[str], dry_run: bool) -> None:
        target = self._resolve_path(action.path)
        if not target.exists():
            raise PatchApplyError(f"Delete File Error - missing file: {action.path}")
        if target.is_dir():
            raise PatchApplyError(f"Delete File Error - '{action.path}' is a directory.")
        if not dry_run:
            target.unlink()
        changed_files.add(action.path)
        if not dry_run:
            LOGGER.info("Deleted file %s", action.path)

    def _apply_update(self, action: PatchAction, changed_files: set[str], dry_run: bool) -> None:
        source = self._resolve_path(action.path)
        if not source.exists():
            raise PatchApplyError(f"Update File Error - missing file: {action.path}")
        if source.is_dir():
            raise PatchApplyError(f"Update File Error - '{action.path}' is a directory.")

        content = source.read_text(encoding="utf-8")
        updated_content = content
        last_position = 0

        for chunk_index, chunk in enumerate(action.chunks, start=1):
            updated_content, last_position = self._apply_chunk(
                updated_content, chunk, action.path, chunk_index, last_position
            )

        destination_path = action.move_path or action.path
        dest = self._resolve_path(destination_path)
        if action.move_path and dest.exists():
            raise PatchApplyError(f"Move target '{action.move_path}' already exists.")

        if not dry_run:
            if action.move_path:
                dest.parent.mkdir(parents=True, exist_ok=True)
                source.unlink()
                dest.write_text(updated_content, encoding="utf-8")
            else:
                source.write_text(updated_content, encoding="utf-8")

        changed_files.add(destination_path)
        if action.move_path:
            changed_files.add(action.path)
        if not dry_run:
            if action.move_path:
                LOGGER.info("Renamed %s -> %s", action.path, action.move_path)
            else:
                LOGGER.info("Updated file %s", action.path)

    def _apply_chunk(
        self,
        content: str,
        chunk: PatchChunk,
        path: str,
        chunk_index: int,
        last_position: int,
    ) -> tuple[str, int]:
        old_text = "\n".join(chunk.old_lines)
        new_text = "\n".join(chunk.new_lines)

        if old_text:
            # Use layered fuzzy matching (inspired by Aider)
            match_result = self._find_context_fuzzy(
                content, old_text, chunk.old_lines, last_position, path, chunk_index
            )
            if match_result is None:
                # Build a helpful error message with similar lines suggestion
                self._raise_context_not_found(old_text, content, path, chunk_index)

            pos, end, actual_old_text, warning = match_result
            if warning:
                self._warnings.append(f"Chunk {chunk_index} in '{path}': {warning}")
                LOGGER.info("Fuzzy match in chunk %d of '%s': %s", chunk_index, path, warning)

            replaced = content[:pos] + new_text + content[end:]
            return replaced, pos + len(new_text)

        insert_at = self._locate_insertion_point(content, chunk, last_position)
        replaced = content[:insert_at] + new_text + content[insert_at:]
        return replaced, insert_at + len(new_text)

    def _find_context_fuzzy(
        self,
        content: str,
        old_text: str,
        old_lines: list[str],
        start_pos: int,
        path: str,
        chunk_index: int,
    ) -> tuple[int, int, str, str | None] | None:
        """Find old_text in content using layered fuzzy matching.

        Tries matching strategies in order of strictness:
        1. Exact match (current behavior)
        2. Trailing whitespace tolerance (rstrip each line)
        3. Leading whitespace tolerance (uniform indent differences)

        Returns:
            (start_pos, end_pos, matched_text, warning) or None if no match found
        """
        # Layer 1: Exact match (try from last_position first, then from start)
        pos = content.find(old_text, start_pos)
        if pos != -1:
            return pos, pos + len(old_text), old_text, None
        pos = content.find(old_text)
        if pos != -1:
            return pos, pos + len(old_text), old_text, None

        # Layer 2: Trailing whitespace tolerance
        match = self._find_with_trailing_ws_tolerance(content, old_lines, start_pos)
        if match:
            pos, end, matched = match
            return pos, end, matched, "Matched after stripping trailing whitespace"

        # Layer 3: Leading whitespace tolerance (Aider's key insight)
        match = self._find_with_leading_ws_tolerance(content, old_lines, start_pos)
        if match:
            pos, end, matched = match
            return pos, end, matched, "Matched after adjusting indentation"

        return None

    def _find_with_trailing_ws_tolerance(
        self, content: str, old_lines: list[str], start_pos: int
    ) -> tuple[int, int, str] | None:
        """Find match with trailing whitespace stripped from each line."""
        content_lines = content.splitlines(keepends=True)
        stripped_old = [line.rstrip() for line in old_lines]

        for i in range(len(content_lines) - len(old_lines) + 1):
            chunk = content_lines[i : i + len(old_lines)]
            chunk_stripped = [line.rstrip() for line in chunk]

            if stripped_old == chunk_stripped:
                # Calculate character positions
                char_pos = sum(len(content_lines[j]) for j in range(i))
                char_end = char_pos + sum(len(line) for line in chunk)
                matched_text = "".join(chunk).rstrip("\n")
                return char_pos, char_end, matched_text

        return None

    def _find_with_leading_ws_tolerance(
        self, content: str, old_lines: list[str], start_pos: int
    ) -> tuple[int, int, str] | None:
        """Find match ignoring uniform leading whitespace differences.

        This handles the common case where LLMs get indentation wrong uniformly
        across all lines (e.g., missing 4 spaces from every line).
        """
        content_lines = content.splitlines(keepends=True)

        # Calculate minimum leading whitespace in old_lines (non-empty lines only)
        leading = [len(line) - len(line.lstrip()) for line in old_lines if line.strip()]
        if not leading:
            return None
        min_leading = min(leading)

        # Strip that minimum from old_lines for comparison
        stripped_old = [
            line[min_leading:].rstrip() if line.strip() else line.rstrip() for line in old_lines
        ]

        # Search for match ignoring leading whitespace
        for i in range(len(content_lines) - len(old_lines) + 1):
            chunk = content_lines[i : i + len(old_lines)]

            # Check if non-whitespace content matches (after stripping leading ws)
            chunk_stripped = [line.lstrip().rstrip() for line in chunk]
            old_stripped_content = [line.lstrip().rstrip() for line in stripped_old]

            if chunk_stripped == old_stripped_content:
                # Found a match - calculate positions
                char_pos = sum(len(content_lines[j]) for j in range(i))
                char_end = char_pos + sum(len(line) for line in chunk)
                matched_text = "".join(chunk).rstrip("\n")
                return char_pos, char_end, matched_text

        return None

    def _raise_context_not_found(
        self, old_text: str, content: str, path: str, chunk_index: int
    ) -> None:
        """Raise a helpful error with similar-line suggestions."""
        # Try to find similar content for diagnostic purposes
        similar = find_similar_lines(old_text, content, threshold=0.5)

        # Detect if this looks like an insertion attempt with invented anchors
        old_lines = old_text.strip().splitlines()
        looks_like_insertion = self._looks_like_invented_anchor(old_lines, content)

        # Build the error message
        old_preview = old_text[:200] + "..." if len(old_text) > 200 else old_text
        error_parts = [f"Chunk {chunk_index}: context not found in '{path}'."]
        error_parts.append(f"\nYour patch expects:\n{old_preview}")

        if looks_like_insertion:
            # Provide specific guidance for insertion tasks
            error_parts.append(
                "\n\n⚠️ INSERTION DETECTED: Your - lines don't exist in the file."
                "\n\nTo INSERT new content, use one of these approaches:"
                "\n\n1. APPEND AT END (recommended for new sections):"
                "\n   *** Update File: " + path + "\n   @@"
                "\n   +"
                "\n   +## Your New Section"
                "\n   +Your content here."
                "\n   *** End of File"
                "\n\n2. INSERT AFTER EXISTING LINE (copy an ACTUAL line from the file):"
                "\n   @@"
                "\n   -## Existing Section  <-- must be EXACT text from file"
                "\n   +## Existing Section"
                "\n   +"
                "\n   +## Your New Section"
            )
            # Show actual section headers in the file
            headers = [line for line in content.splitlines() if line.startswith("## ")]
            if headers:
                error_parts.append(f"\n\nActual section headers in '{path}':")
                for h in headers[:10]:
                    error_parts.append(f"\n  {h}")
        elif similar:
            error_parts.append(f"\n\nDid you mean to match these lines?\n{similar}")
        else:
            # Show first N lines of file content
            content_preview = "\n".join(content.splitlines()[:15])
            if len(content.splitlines()) > 15:
                content_preview += "\n..."
            error_parts.append(f"\n\nActual file content (first 15 lines):\n{content_preview}")

        error_parts.append("\n\nTIP: Use READ tool to get exact file content before editing.")
        raise PatchApplyError("".join(error_parts))

    def _looks_like_invented_anchor(self, old_lines: list[str], content: str) -> bool:
        """Detect if old_lines look like invented section headers for an insertion.

        Returns True if the - lines appear to be section headers or anchor text
        that the LLM invented rather than copied from the actual file.
        """
        if not old_lines:
            return False

        # Check if any line looks like a markdown header
        has_header_like = any(
            line.strip().startswith("#")
            or line.strip().startswith("##")
            or line.strip().lower().startswith("contributing")
            or line.strip().lower().startswith("license")
            for line in old_lines
        )

        if not has_header_like:
            return False

        # Check if these headers actually exist in the file
        content_lower = content.lower()
        for line in old_lines:
            stripped = line.strip()
            if stripped and stripped.lower() in content_lower:
                return False  # Found at least one matching line

        # Headers present but none match - likely invented
        return True

    def _locate_insertion_point(self, content: str, chunk: PatchChunk, start_pos: int) -> int:
        if chunk.context:
            context_pos = content.find(chunk.context)
            if context_pos != -1:
                return context_pos + len(chunk.context)
        if chunk.is_end_of_file:
            return len(content)
        return min(start_pos, len(content))

    def _resolve_path(self, relative: str) -> Path:
        candidate = (self.repo_root / relative).resolve()
        repo_root = self.repo_root.resolve()
        try:
            candidate.relative_to(repo_root)
        except ValueError as exc:
            raise PatchApplyError(f"Path '{relative}' escapes repository root.") from exc
        return candidate


def _fs_edit(payload: Mapping[str, Any], context: "ToolContext") -> Mapping[str, Any]:
    """Filesystem edit handler that consumes apply_patch format."""
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
        parser = PatchParser(patch_text)
        parse_result = parser.parse_with_warnings()
        actions = parse_result.actions
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

    applier = PatchApplier(context.repo_root)
    validation = applier.apply(actions, dry_run=True)
    if not validation["success"]:
        # Include parser warnings even on validation failure
        validation["warnings"] = all_warnings + validation.get("warnings", [])
        return validation

    result = applier.apply(actions, dry_run=False)
    # Combine all warnings
    result["warnings"] = all_warnings + result.get("warnings", [])
    return result


__all__ = [
    "ParseResult",
    "PatchAction",
    "PatchApplyError",
    "PatchChunk",
    "PatchFormatError",
    "PatchParser",
    "_fs_edit",
    "find_similar_lines",
]
