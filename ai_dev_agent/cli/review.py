"""Review command helpers and execution logic."""

from __future__ import annotations

import fnmatch
import re
from collections import deque
from collections.abc import Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import click

from ai_dev_agent.agents.schemas import VIOLATION_SCHEMA
from ai_dev_agent.core.utils.logger import get_logger
from ai_dev_agent.tools.patch_analysis import PatchParser

from .react.executor import _execute_react_assistant
from .review_context import ContextOrchestrator, SourceContextProvider
from .utils import _record_invocation, get_llm_client, resolve_prompt_input

if TYPE_CHECKING:
    from ai_dev_agent.core.utils.config import Settings

LOGGER = get_logger(__name__)

# Cache for parsed patches: key=(path, mtime, size) -> parsed_data
_PATCH_CACHE: dict[tuple[str, float, int | None], dict[str, Any]] = {}


def _normalize_applies_to_pattern(raw: str) -> str | None:
    """Convert a glob-style rule pattern into a regex string."""
    if not raw:
        return None

    # Split on commas or whitespace to support simple multi-pattern lists
    tokens = [token.strip() for token in re.split(r"[,\s]+", raw) if token.strip()]
    if not tokens:
        return None

    regex_parts: list[str] = []
    for token in tokens:
        if token.lower().startswith("regex:"):
            custom = token[6:].strip()
            if custom:
                regex_parts.append(custom)
            continue

        if any(ch in token for ch in ("(", ")", "[", "]", "{", "}", "|", "\\")):
            regex_parts.append(token)
            continue

        translated = fnmatch.translate(token)
        if translated.endswith("\\Z"):
            translated = translated[:-2] + "$"
        regex_parts.append(translated)

    if not regex_parts:
        return None

    if len(regex_parts) == 1:
        return regex_parts[0]

    return "|".join(f"(?:{part})" for part in regex_parts)


def extract_applies_to_pattern(rule_content: str) -> str | None:
    """Extract the 'Applies To' pattern from rule content.

    Looks for patterns like:
    - ## Applies To
      pattern_here
    - Applies To: pattern_here
    - scope: pattern_here

    Returns:
        Regex pattern string or None if not found
    """
    # Try multiple patterns to be robust
    patterns = [
        r"##\s*Applies\s+To\s*\n\s*([^\n]+)",  # ## Applies To\n  pattern
        r"Applies\s+To:\s*([^\n]+)",  # Applies To: pattern
        r"scope:\s*([^\n]+)",  # scope: pattern
        r"##\s*Scope\s*\n\s*([^\n]+)",  # ## Scope\n  pattern
    ]

    for pattern in patterns:
        match = re.search(pattern, rule_content, re.IGNORECASE)
        if match:
            extracted = match.group(1).strip()
            # Remove quotes if present
            extracted = extracted.strip("\"'`")
            if extracted:
                normalized = _normalize_applies_to_pattern(extracted)
                if normalized:
                    LOGGER.debug(
                        "Extracted 'Applies To' pattern (normalized): %s -> %s",
                        extracted,
                        normalized,
                    )
                    return normalized
                LOGGER.debug("Extracted 'Applies To' pattern: %s", extracted)
                return extracted

    LOGGER.debug("No 'Applies To' pattern found in rule")
    return None


def parse_patch_file(patch_path: Path) -> dict[str, Any]:
    """Parse the patch file into structured data for review.

    Uses caching based on file path and modification time to avoid re-parsing
    unchanged patches.
    """
    # Check cache
    try:
        stat = patch_path.stat()
        cache_key = (str(patch_path), stat.st_mtime, getattr(stat, "st_size", None))

        if cache_key in _PATCH_CACHE:
            LOGGER.debug("Using cached parse result for %s", patch_path.name)
            return _PATCH_CACHE[cache_key]
    except OSError:
        pass  # If stat fails, just parse without caching

    # Parse the patch
    try:
        content = patch_path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:  # pragma: no cover - defensive guard
        raise click.ClickException(f"Failed to read patch file '{patch_path}': {exc}") from exc

    parser = PatchParser(content, include_context=True)
    parsed = parser.parse()

    # Store in cache
    try:
        cache_key = (str(patch_path), stat.st_mtime, getattr(stat, "st_size", None))
        _PATCH_CACHE[cache_key] = parsed

        # Limit cache size to prevent unbounded growth
        if len(_PATCH_CACHE) > 10:
            # Remove oldest entry (first key)
            oldest_key = next(iter(_PATCH_CACHE))
            del _PATCH_CACHE[oldest_key]
            LOGGER.debug("Cache evicted oldest entry to maintain size limit")
    except (NameError, OSError):
        pass  # If caching fails, continue without it

    return parsed


def collect_patch_review_data(
    parsed_patch: Mapping[str, Any],
) -> tuple[dict[str, dict[int, str]], dict[str, dict[int, str]], set[str]]:
    """Collect lookup tables for added/removed lines and parsed file set."""

    added_lines: dict[str, dict[int, str]] = {}
    removed_lines: dict[str, dict[int, str]] = {}
    parsed_files: set[str] = set()
    for file_entry in parsed_patch.get("files", []):
        path = file_entry.get("path")
        if not isinstance(path, str):
            continue
        parsed_files.add(path)
        added_lookup: dict[int, str] = {}
        removed_lookup: dict[int, str] = {}
        for hunk in file_entry.get("hunks", []):
            if not isinstance(hunk, Mapping):
                continue
            for added in hunk.get("added_lines", []):
                if not isinstance(added, Mapping):
                    continue
                line_no = added.get("line_number")
                content_value = added.get("content")
                if isinstance(line_no, int) and isinstance(content_value, str):
                    added_lookup[line_no] = content_value
            for removed in hunk.get("removed_lines", []):
                if not isinstance(removed, Mapping):
                    continue
                line_no = removed.get("line_number")
                content_value = removed.get("content")
                if isinstance(line_no, int) and isinstance(content_value, str):
                    removed_lookup[line_no] = content_value
        if added_lookup:
            added_lines[path] = added_lookup
        if removed_lookup:
            removed_lines[path] = removed_lookup

    return added_lines, removed_lines, parsed_files


def format_patch_dataset(parsed_patch: Mapping[str, Any], filter_pattern: str | None = None) -> str:
    """Format parsed patch data into a text block for the reviewer.

    Args:
        parsed_patch: Parsed patch structure from PatchParser
        filter_pattern: Optional regex pattern to filter files (e.g., from rule's "Applies To")

    Returns:
        Formatted text representation of patch data
    """
    lines: list[str] = []
    files: Sequence[Mapping[str, Any]] = parsed_patch.get("files", []) or []

    if not files:
        lines.append("No files with additions were detected in this patch.")
        return "\n".join(lines)

    # Apply filter if provided
    filtered_files = files
    if filter_pattern:
        try:
            pattern_re = re.compile(filter_pattern)
            filtered_files = [f for f in files if f.get("path") and pattern_re.search(f["path"])]
        except re.error:
            LOGGER.warning("Invalid filter pattern '%s', showing all files", filter_pattern)
            filtered_files = files

    if not filtered_files:
        lines.append(f"No files matching pattern '{filter_pattern}' were found in this patch.")
        return "\n".join(lines)

    for file_entry in filtered_files:
        path = file_entry.get("path")
        change_type = file_entry.get("change_type", "modified")
        language = file_entry.get("language", "unknown")
        chunk_index = file_entry.get("_chunk_index")
        chunk_total = file_entry.get("_chunk_total")
        chunk_suffix = ""
        if isinstance(chunk_index, int) and isinstance(chunk_total, int) and chunk_total > 1:
            chunk_suffix = f" (segment {chunk_index + 1}/{chunk_total})"
        lines.append(f"FILE: {path}{chunk_suffix}")
        lines.append(f"  Change type: {change_type}")
        lines.append(f"  Language: {language}")

        hunks = file_entry.get("hunks", []) or []
        total_added = sum(len(hunk.get("added_lines", []) or []) for hunk in hunks)
        total_removed = sum(len(hunk.get("removed_lines", []) or []) for hunk in hunks)

        lines.append(f"  Total added lines: {total_added}")
        lines.append(f"  Total removed lines: {total_removed}")

        if not hunks:
            lines.append("  HUNK: (none)")
            lines.append("    CONTEXT:")
            lines.append("      (none)")
            lines.append("    ADDED LINES:")
            lines.append("      (none)")
            lines.append("    REMOVED LINES:")
            lines.append("      (none)")
            lines.append("")
            continue

        for hunk in hunks:
            header = hunk.get("header")
            header_display = header.strip() if isinstance(header, str) else ""
            lines.append(f"  HUNK: {header_display or '(no header)'}")

            context_block = hunk.get("context_lines", []) or []
            lines.append("    CONTEXT:")
            if context_block:
                for ctx_entry in context_block:
                    line_number = ctx_entry.get("line_number")
                    content = ctx_entry.get("content", "")
                    if isinstance(line_number, int):
                        lines.append(f"      {line_number:4d} | {content}")
                    else:
                        lines.append(f"      | {content}")
            else:
                lines.append("      (none)")

            added_lines = hunk.get("added_lines", []) or []
            lines.append("    ADDED LINES:")
            if added_lines:
                for entry in added_lines:
                    line_number = entry.get("line_number")
                    content = entry.get("content", "")
                    if isinstance(line_number, int):
                        lines.append(f"      {line_number:4d} + {content}")
                    else:
                        lines.append(f"      + {content}")
            else:
                lines.append("      (none)")

            removed_lines = hunk.get("removed_lines", []) or []
            lines.append("    REMOVED LINES:")
            if removed_lines:
                for entry in removed_lines:
                    line_number = entry.get("line_number")
                    content = entry.get("content", "")
                    if isinstance(line_number, int):
                        lines.append(f"      {line_number:4d} - {content}")
                    else:
                        lines.append(f"      - {content}")
            else:
                lines.append("      (none)")

            lines.append("")  # Blank line between hunks

        lines.append("")  # Blank line between files

    return "\n".join(lines).rstrip()


def _estimate_hunk_impact(hunk: Mapping[str, Any]) -> int:
    added = len(hunk.get("added_lines", []) or [])
    removed = len(hunk.get("removed_lines", []) or [])
    context = len(hunk.get("context_lines", []) or [])
    return added + removed + context


def _split_large_file_entries(
    files: Sequence[Mapping[str, Any]],
    *,
    max_hunks_per_group: int,
    max_lines_per_group: int,
) -> list[Mapping[str, Any]]:
    if max_hunks_per_group <= 0 and max_lines_per_group <= 0:
        return list(files)

    results: list[Mapping[str, Any]] = []
    for file_entry in files:
        hunks = list(file_entry.get("hunks", []) or [])
        if not hunks:
            results.append(file_entry)
            continue

        groups: list[list[Mapping[str, Any]]] = []
        current: list[Mapping[str, Any]] = []
        hunk_counter = 0
        line_counter = 0

        for hunk in hunks:
            impact = _estimate_hunk_impact(hunk)
            should_split = current and (
                (max_hunks_per_group > 0 and hunk_counter >= max_hunks_per_group)
                or (max_lines_per_group > 0 and line_counter + impact > max_lines_per_group)
            )
            if should_split:
                groups.append(current)
                current = []
                hunk_counter = 0
                line_counter = 0
            current.append(hunk)
            hunk_counter += 1
            line_counter += impact

        if current:
            groups.append(current)

        if len(groups) <= 1:
            results.append(file_entry)
            continue

        total = len(groups)
        for index, group in enumerate(groups):
            new_entry = dict(file_entry)
            new_entry["hunks"] = group
            new_entry["_chunk_index"] = index
            new_entry["_chunk_total"] = total
            results.append(new_entry)

    return results


def _estimate_entry_lines(file_entry: Mapping[str, Any]) -> int:
    return sum(_estimate_hunk_impact(h) for h in file_entry.get("hunks", []) or [])


def _chunk_patch_files(
    files: Sequence[Mapping[str, Any]],
    max_files_per_chunk: int = 10,
    max_lines_per_chunk: int = 0,
    overlap_lines: int = 0,
) -> list[list[Mapping[str, Any]]]:
    """Split patch files into manageable chunks with optional overlap.

    Args:
        files: Sequence of file entries to chunk
        max_files_per_chunk: Maximum files per chunk
        max_lines_per_chunk: Maximum lines per chunk
        overlap_lines: Number of lines to overlap between chunks (for quality)

    Returns:
        List of file chunks, with overlap if specified
    """
    if not files:
        return []

    effective_max_files = max_files_per_chunk if max_files_per_chunk > 0 else len(files)
    effective_max_lines = max_lines_per_chunk if max_lines_per_chunk > 0 else 0

    chunked: list[list[Mapping[str, Any]]] = []
    current_chunk: list[Mapping[str, Any]] = []
    current_lines = 0

    for entry in files:
        entry_lines = _estimate_entry_lines(entry)
        should_split = current_chunk and (
            (effective_max_files and len(current_chunk) >= effective_max_files)
            or (effective_max_lines and current_lines + entry_lines > effective_max_lines)
        )
        if should_split:
            chunked.append(current_chunk)
            current_chunk = []
            current_lines = 0
        current_chunk.append(entry)
        current_lines += entry_lines

    if current_chunk:
        chunked.append(current_chunk)

    # Apply overlap between chunks if requested and there are multiple chunks
    if overlap_lines > 0 and len(chunked) > 1:
        chunked = _apply_chunk_overlap(chunked, overlap_lines)

    return chunked


def _apply_chunk_overlap(
    chunks: list[list[Mapping[str, Any]]],
    overlap_lines: int,
) -> list[list[Mapping[str, Any]]]:
    """Apply overlap between consecutive chunks for quality improvement.

    Takes files from the end of chunk N and adds them to the beginning of chunk N+1,
    ensuring violations near boundaries aren't missed.

    Args:
        chunks: List of file entry chunks
        overlap_lines: Target lines to overlap between chunks

    Returns:
        Chunks with overlap applied
    """
    if len(chunks) <= 1 or overlap_lines <= 0:
        return chunks

    overlapped: list[list[Mapping[str, Any]]] = [chunks[0]]  # First chunk unchanged

    for i in range(1, len(chunks)):
        prev_chunk = chunks[i - 1]
        current_chunk = list(chunks[i])  # Copy to avoid mutation

        # Collect files from end of previous chunk until we have ~overlap_lines
        overlap_files: list[Mapping[str, Any]] = []
        accumulated_lines = 0

        for file_entry in reversed(prev_chunk):
            file_lines = _estimate_entry_lines(file_entry)
            if accumulated_lines + file_lines > overlap_lines and overlap_files:
                break  # Stop if we've accumulated enough
            overlap_files.insert(0, file_entry)
            accumulated_lines += file_lines

        # Prepend overlap files to current chunk (avoiding duplicates)
        current_paths = {f.get("path") for f in current_chunk if f.get("path")}
        for overlap_file in overlap_files:
            if overlap_file.get("path") not in current_paths:
                current_chunk.insert(0, overlap_file)

        overlapped.append(current_chunk)

    return overlapped


def _compute_dynamic_line_limit(
    configured_limit: int,
    rule_text: str,
    files: Sequence[Mapping[str, Any]],
) -> int:
    """Derive a chunk line limit based on rule size and patch impact."""
    DEFAULT_LINES_LIMIT = 1500  # Increased from 320 to use model capacity better
    base_limit = configured_limit if configured_limit > 0 else DEFAULT_LINES_LIMIT

    total_line_impact = sum(_estimate_entry_lines(entry) for entry in files)
    rule_token_estimate = max(len(rule_text) // 4, 0)

    # Combined token estimate (rule + patch, assuming ~2 tokens per line)
    combined_tokens = rule_token_estimate + (total_line_impact * 2)

    # Only shrink if approaching model's 200K context window
    # Use 75% of context (150K tokens) as safe threshold
    if combined_tokens > 150_000:
        base_limit = min(base_limit, 400)
    elif combined_tokens > 100_000:
        base_limit = min(base_limit, 800)
    elif combined_tokens > 75_000:
        base_limit = min(base_limit, 1200)

    if configured_limit <= 0 and base_limit == DEFAULT_LINES_LIMIT:
        return 0
    return base_limit


def _compute_dynamic_file_limit(
    configured_limit: int,
    rule_text: str,
    files: Sequence[Mapping[str, Any]],
) -> int:
    """Derive a file-per-chunk limit considering rule size and patch footprint."""
    DEFAULT_FILES_LIMIT = 50  # Increased from 10 to reduce number of chunks
    base_limit = configured_limit if configured_limit > 0 else DEFAULT_FILES_LIMIT

    total_line_impact = sum(_estimate_entry_lines(entry) for entry in files)
    rule_token_estimate = max(len(rule_text) // 4, 0)

    # Combined token estimate (rule + patch, assuming ~2 tokens per line)
    combined_tokens = rule_token_estimate + (total_line_impact * 2)

    # Only shrink for truly massive contexts approaching 200K token limit
    if combined_tokens > 150_000:
        base_limit = min(base_limit, 10)
    elif combined_tokens > 100_000:
        base_limit = min(base_limit, 20)
    elif combined_tokens > 75_000:
        base_limit = min(base_limit, 35)

    if configured_limit <= 0 and base_limit == DEFAULT_FILES_LIMIT:
        return len(files) or 1
    return max(1, base_limit)


def _estimate_prompt_tokens(*segments: str) -> int:
    """Approximate token usage by dividing character count by 4."""
    total_chars = sum(len(segment) for segment in segments if segment)
    return max(total_chars // 4, 0)


def _refine_chunks_for_token_budget(
    chunks: Sequence[list[Mapping[str, Any]]],
    *,
    rule_text: str,
    filter_pattern: str | None,
    token_budget: int,
) -> list[list[Mapping[str, Any]]]:
    """Split chunks further if combined rule+patch text exceeds the budget."""
    refined: list[list[Mapping[str, Any]]] = []
    for chunk in chunks:
        queue: deque[list[Mapping[str, Any]]] = deque([chunk])
        while queue:
            current = queue.popleft()
            subset_patch = {"files": current}
            dataset_text = format_patch_dataset(subset_patch, filter_pattern=filter_pattern)
            approx_tokens = _estimate_prompt_tokens(rule_text, dataset_text)
            if approx_tokens > token_budget and len(current) > 1:
                midpoint = max(1, len(current) // 2)
                queue.appendleft(current[midpoint:])
                queue.appendleft(current[:midpoint])
                continue
            refined.append(current)
    return refined


def validate_review_response(
    response: dict[str, Any],
    *,
    added_lines: dict[str, dict[int, str]],
    removed_lines: dict[str, dict[int, str]] | None = None,
    parsed_files: set[str],
) -> dict[str, Any]:
    """Ensure reviewer output references actual lines from the parsed patch."""

    if not isinstance(response, dict):
        raise click.ClickException("Reviewer output is not valid JSON.")

    violations = response.get("violations")
    if not isinstance(violations, list):
        raise click.ClickException("Reviewer output missing 'violations' array.")

    summary = response.get("summary", {})
    if summary is None:
        summary = {}
    if not isinstance(summary, Mapping):
        raise click.ClickException("Reviewer output has invalid 'summary' payload.")

    valid_violations: list[dict[str, Any]] = []
    discarded_entries: list[str] = []

    severity_aliases = {
        "critical": "error",
        "high": "error",
        "major": "error",
        "medium": "warning",
        "moderate": "warning",
        "minor": "info",
        "low": "info",
    }
    allowed_severities = {"error", "warning", "info"}
    change_type_aliases = {
        "addition": "added",
        "add": "added",
        "added": "added",
        "modification": "added",
        "update": "added",
        "deletion": "removed",
        "delete": "removed",
        "deleted": "removed",
        "removed": "removed",
        "removal": "removed",
    }

    for entry in violations:
        if not isinstance(entry, Mapping):
            discarded_entries.append("non-object violation entry")
            continue

        sanitized: dict[str, Any] = dict(entry)
        file_path = sanitized.get("file")
        line_number = sanitized.get("line")
        snippet = sanitized.get("code_snippet")
        change_type_value = str(sanitized.get("change_type", "added")).strip().lower()
        change_type = change_type_aliases.get(change_type_value, "added")
        sanitized["change_type"] = change_type

        severity_value = sanitized.get("severity")
        normalized_severity = "warning"
        if isinstance(severity_value, str):
            candidate = severity_value.strip().lower()
            candidate = severity_aliases.get(candidate, candidate)
            if candidate in allowed_severities:
                normalized_severity = candidate
        sanitized["severity"] = normalized_severity

        if not isinstance(file_path, str) or not isinstance(line_number, int):
            discarded_entries.append(str(entry))
            continue

        if change_type == "removed":
            removed_for_file = (removed_lines or {}).get(file_path)
            if removed_for_file is None:
                discarded_entries.append(f"{file_path}:{line_number} (removed line not in patch)")
                continue
            actual_line = removed_for_file.get(line_number)
            if actual_line is None:
                discarded_entries.append(f"{file_path}:{line_number} (line not removed)")
                continue
            if isinstance(snippet, str) and snippet.strip() != actual_line.strip():
                discarded_entries.append(f"{file_path}:{line_number} (content mismatch)")
            else:
                sanitized.setdefault("severity", "warning")
                valid_violations.append(sanitized)
            continue

        added_for_file = added_lines.get(file_path)
        if added_for_file is None:
            discarded_entries.append(f"{file_path}:{line_number} (file not in patch)")
            continue

        actual_line = added_for_file.get(line_number)
        if actual_line is None:
            discarded_entries.append(f"{file_path}:{line_number} (line not added)")
            continue

        if isinstance(snippet, str) and snippet.strip() != actual_line.strip():
            discarded_entries.append(f"{file_path}:{line_number} (content mismatch)")
            continue

        valid_violations.append(sanitized)

    if discarded_entries:
        preview = ", ".join(discarded_entries[:5])
        if len(discarded_entries) > 5:
            preview += ", ..."
        LOGGER.warning(
            "Discarded %s invalid violation(s) from reviewer output: %s",
            len(discarded_entries),
            preview,
        )

    total_violations = summary.get("total_violations")
    if isinstance(total_violations, int) and total_violations != len(violations):
        LOGGER.debug(
            "Adjusting total_violations from %s to %s based on validated entries",
            total_violations,
            len(valid_violations),
        )

    files_reviewed = summary.get("files_reviewed")
    if isinstance(files_reviewed, int) and files_reviewed != len(parsed_files):
        LOGGER.debug(
            "Adjusting files_reviewed from %s to %s based on parsed files",
            files_reviewed,
            len(parsed_files),
        )

    normalized_summary = dict(summary)
    normalized_summary["total_violations"] = len(valid_violations)
    normalized_summary["files_reviewed"] = len(parsed_files)
    if discarded_entries:
        normalized_summary["discarded_violations"] = len(discarded_entries)

    rule_name = normalized_summary.get("rule_name")
    if not isinstance(rule_name, str) or not rule_name.strip():
        normalized_summary["rule_name"] = normalized_summary.get("rule_name") or ""

    return {
        "violations": valid_violations,
        "summary": normalized_summary,
    }


def _build_review_prompt(
    patch_rel: Path,
    patch_dataset: str,
    *,
    context_section: str | None = None,
) -> str:
    context_block = ""
    if context_section:
        context_block = f"\n## Additional Source Context\n\n{context_section}\n"
    return f"""# Code Review Task: {patch_rel}

You are reviewing a patch file against a specific coding rule. Your task is to identify violations with precision and accuracy.

## Critical Instructions

1. **USE ONLY THE PRE-PARSED PATCH DATASET BELOW** as your source of truth
   - Do NOT attempt to read files using tools
   - Do NOT guess at line numbers or content
   - All line numbers and code snippets MUST come from the "ADDED LINES" sections

2. **EXACT MATCHING REQUIRED**
   - The `file` field must exactly match the file path shown in the dataset
   - The `line` field must exactly match a line number from "ADDED LINES"
   - The `code_snippet` field must exactly match the content from that line (strip whitespace for comparison)

3. **OUTPUT FORMAT**
   - Return valid JSON conforming to the schema provided
   - Each violation MUST reference an actual added line from the dataset
   - If no violations found, return empty violations array

## Review Workflow

### Step 1: Understand the Rule
- Read the rule's title, description, and scope
- Study the "Detect" section to understand what constitutes a violation
- Review examples of BAD and GOOD code patterns

### Step 2: Analyze the Patch Dataset
- The dataset shows files with their added/removed lines
- Focus ONLY on "ADDED LINES" sections (marked with +)
- Note the exact line numbers next to each added line

### Step 3: Identify Violations
For each added line:
- Check if it matches the violation pattern described in the rule
- Verify the file matches the rule's "Applies To" pattern
- If it's a violation:
  * Record the EXACT file path from the dataset
  * Record the EXACT line number from the "ADDED LINES" section
  * Copy the EXACT code snippet (you may strip leading/trailing whitespace)
  * Write a clear, actionable message using the rule's template

### Step 4: Validate Your Output
Before submitting:
- Verify each violation references an actual line from "ADDED LINES"
- Confirm line numbers match exactly
- Ensure file paths are correct
- Check that code snippets match (ignoring whitespace)

## Common Mistakes to Avoid

❌ **DON'T** report violations for lines not in the patch
❌ **DON'T** use line numbers from context or removed lines
❌ **DON'T** guess or estimate line numbers
❌ **DON'T** read files with tools - use only the dataset below
❌ **DON'T** report the same violation multiple times

✅ **DO** use exact line numbers from "ADDED LINES" sections
✅ **DO** copy exact file paths from the dataset
✅ **DO** match code snippets precisely
✅ **DO** return empty violations if no issues found
{context_block}
## Patch Dataset (Your Single Source of Truth)

{patch_dataset}

---

Now review the patch and output your findings in the required JSON format.
"""


def run_review(
    ctx: click.Context,
    *,
    patch_file: str,
    rule_file: str,
    json_output: bool,
    settings: Settings,
) -> dict[str, Any]:
    """Execute the review command and return the validated JSON result."""

    if not settings.api_key:
        raise click.ClickException(
            "No API key configured (DEVAGENT_API_KEY). Code review requires an LLM."
        )

    patch_path = Path(patch_file).resolve()
    rule_path = Path(rule_file).resolve()

    try:
        rule_content = rule_path.read_text(encoding="utf-8")
    except Exception as exc:
        raise click.ClickException(f"Failed to read rule file '{rule_file}': {exc}") from exc

    original_workspace_root = getattr(settings, "workspace_root", None)
    original_max_tool_output_chars = getattr(settings, "max_tool_output_chars", None)
    original_max_context_tokens = getattr(settings, "max_context_tokens", None)
    original_response_headroom = getattr(settings, "response_headroom_tokens", None)
    # Dynamic instructions feature has been removed

    try:
        review_workspace: Path | None = None
        common_parts: list[Path] = []
        for p1, p2 in zip(patch_path.parents, rule_path.parents):
            if p1 == p2:
                common_parts.append(p1)
                break

        if common_parts:
            candidate = common_parts[0]
            candidate_anchor = Path(candidate.anchor)
            candidate_is_root = candidate == candidate_anchor
            candidate_within_original = False
            if original_workspace_root is not None:
                try:
                    candidate.relative_to(original_workspace_root)
                    candidate_within_original = True
                except ValueError:
                    candidate_within_original = False
            if (
                original_workspace_root is None or candidate_within_original
            ) and not candidate_is_root:
                review_workspace = candidate

        if review_workspace is not None:
            settings.workspace_root = review_workspace
            patch_rel = patch_path.relative_to(review_workspace)
        elif original_workspace_root is not None:
            try:
                patch_rel = patch_path.relative_to(original_workspace_root)
            except ValueError:
                patch_rel = patch_path
        else:
            patch_rel = patch_path

        # Extract "Applies To" pattern from rule to filter patch dataset
        filter_pattern = extract_applies_to_pattern(rule_content)

        parsed_patch = parse_patch_file(patch_path)
        original_files: Sequence[Mapping[str, Any]] = parsed_patch.get("files", []) or []
        original_files = sorted(
            original_files,
            key=lambda entry: (str(entry.get("path", "")), entry.get("_chunk_index", 0)),
        )

        max_lines_setting = getattr(settings, "review_max_lines_per_chunk", 1500)
        dynamic_line_limit = _compute_dynamic_line_limit(
            max_lines_setting,
            rule_content,
            original_files,
        )

        expanded_files = _split_large_file_entries(
            original_files,
            max_hunks_per_group=getattr(settings, "review_max_hunks_per_chunk", 50),
            max_lines_per_group=dynamic_line_limit if dynamic_line_limit > 0 else max_lines_setting,
        )
        expanded_files = sorted(
            expanded_files,
            key=lambda entry: (str(entry.get("path", "")), entry.get("_chunk_index", 0)),
        )

        max_files_per_chunk_setting = getattr(settings, "review_max_files_per_chunk", 50)
        max_files_per_chunk = _compute_dynamic_file_limit(
            max_files_per_chunk_setting,
            rule_content,
            expanded_files,
        )
        max_lines_per_chunk = dynamic_line_limit

        overlap_lines = getattr(settings, "review_chunk_overlap_lines", 100)
        chunks = _chunk_patch_files(
            expanded_files,
            max_files_per_chunk=max_files_per_chunk,
            max_lines_per_chunk=max_lines_per_chunk,
            overlap_lines=overlap_lines,
        )
        max(1, len(chunks))
        use_chunking = len(chunks) > 1

        # Build context orchestrator (will be used per-chunk to avoid budget exhaustion)
        context_orchestrator = ContextOrchestrator(
            [
                SourceContextProvider(
                    pad_lines=getattr(settings, "review_context_pad_lines", 40),
                    max_lines_per_item=getattr(settings, "review_context_max_lines", 600),
                )
            ],
            max_total_lines=getattr(settings, "review_context_max_total_lines", 1500),
        )

        def build_fallback_summary(
            file_count: int,
            existing_summary: Mapping[str, Any] | None = None,
        ) -> dict[str, Any]:
            summary: dict[str, Any] = dict(existing_summary or {})
            summary["total_violations"] = 0
            summary["files_reviewed"] = file_count
            rule_name_value = summary.get("rule_name")
            if not isinstance(rule_name_value, str) or not rule_name_value.strip():
                summary["rule_name"] = rule_path.stem
            return summary

        system_extension_with_rule = f"""# Coding Rule: {rule_path.name}

{rule_content}

---

The above rule has been pre-loaded for your review. Do NOT use the read tool to access it.
Follow the workflow in the user prompt to analyze the patch against this rule."""

        _record_invocation(
            ctx, overrides={"patch": patch_file, "rule": rule_file, "mode": "review"}
        )

        if json_output:
            ctx.obj["silent_mode"] = True

        # Configure aggressive context management for review sessions
        # Reviews can accumulate large context through multiple iterations,
        # especially with large patches and verbose rules
        settings.max_tool_output_chars = 1_000_000
        settings.max_context_tokens = 100_000  # Conservative limit to prevent overflow
        settings.response_headroom_tokens = 10_000  # Leave room for response generation

        try:
            cli_pkg = import_module("ai_dev_agent.cli")
            llm_factory = getattr(cli_pkg, "get_llm_client", get_llm_client)
        except ModuleNotFoundError:
            llm_factory = get_llm_client

        try:
            client = llm_factory(ctx)
        except click.ClickException as exc:
            raise click.ClickException(f"Failed to create LLM client: {exc}") from exc

        class _DeterministicClient:
            """Wrapper that enforces settings.temperature for reproducible reviews."""

            def __init__(self, inner, temperature: float):
                self._inner = inner
                self._temperature = temperature

            def complete(self, messages, *args, **kwargs):
                kwargs.pop("temperature", None)
                kwargs["temperature"] = self._temperature
                kwargs.setdefault("extra_headers", None)
                return self._inner.complete(messages, **kwargs)

            def invoke_tools(self, messages, tools, *, temperature=0.0, **kwargs):
                kwargs["temperature"] = self._temperature
                if "top_p" in kwargs:
                    kwargs["top_p"] = 0.0
                return self._inner.invoke_tools(messages, tools, **kwargs)

            def __getattr__(self, name):
                return getattr(self._inner, name)

        client = _DeterministicClient(client, settings.temperature)

        original_session_id = ctx.obj.get("_session_id")
        review_session_id = f"review-{uuid4()}"
        ctx.obj["_session_id"] = review_session_id

        def reset_session(previous: str | None) -> None:
            if previous:
                ctx.obj["_session_id"] = previous
            else:
                ctx.obj.pop("_session_id", None)

        def execute_dataset(
            dataset_text: str,
            chunk_files: list[Mapping[str, Any]],
            *,
            chunk_index: int | None = None,
            total_expected_chunks: int | None = None,
        ) -> tuple[dict[str, Any], set[str], bool]:
            if not chunk_files:
                summary = build_fallback_summary(0)
                return {"violations": [], "summary": summary}, set(), False

            subset_patch = {"files": chunk_files}
            added_lines, removed_lines, parsed_files = collect_patch_review_data(subset_patch)

            chunk_header = ""
            if chunk_index is not None and total_expected_chunks and total_expected_chunks > 1:
                chunk_header = f"(Chunk {chunk_index + 1} of {total_expected_chunks})\n\n"

            # Build context per-chunk to ensure each chunk gets its own context budget
            context_section = context_orchestrator.build_section(
                settings.workspace_root,
                chunk_files,
            )

            # Append global context message if provided
            resolved_context = resolve_prompt_input(settings.global_context_message)
            if resolved_context:
                global_context_block = f"\n\n## Global Context\n\n{resolved_context}"
                if context_section:
                    context_section += global_context_block
                else:
                    context_section = global_context_block.lstrip()

            prompt = _build_review_prompt(
                patch_rel,
                f"{chunk_header}{dataset_text}",
                context_section=context_section or None,
            )

            ctx.obj["_session_id"] = review_session_id

            try:
                execution_payload = _execute_react_assistant(
                    ctx,
                    client,
                    settings,
                    prompt,
                    use_planning=False,
                    system_extension=system_extension_with_rule,
                    format_schema=VIOLATION_SCHEMA,
                    agent_type="reviewer",
                    suppress_final_output=True,
                )
            except click.ClickException as exc:
                message_text = str(exc)
                if (
                    "Assistant response did not contain valid JSON matching the required schema"
                    in message_text
                ):
                    LOGGER.debug(
                        "LLM output missing required JSON; returning fallback summary: %s", exc
                    )
                    summary = build_fallback_summary(len(parsed_files))
                    return {"violations": [], "summary": summary}, parsed_files, False
                raise

            if not isinstance(execution_payload, dict):
                raise click.ClickException(
                    "Review execution did not return diagnostics for validation."
                )

            run_result = execution_payload.get("result")
            needs_retry = False
            if run_result is None:
                raise click.ClickException(
                    "Review execution missing run metadata; cannot validate results."
                )
            else:
                try:
                    for step in getattr(run_result, "steps", []) or []:
                        observation = getattr(step, "observation", None)
                        if (
                            observation
                            and getattr(observation, "tool", None) == "read"
                            and not getattr(observation, "success", True)
                        ):
                            needs_retry = True
                            break
                except AttributeError:
                    needs_retry = False
            if needs_retry:
                return (
                    {"violations": [], "summary": build_fallback_summary(len(parsed_files))},
                    parsed_files,
                    True,
                )

            final_json = execution_payload.get("final_json")
            if final_json is None:
                raise click.ClickException("Reviewer did not produce JSON output.")

            try:
                validated = validate_review_response(
                    final_json,
                    added_lines=added_lines,
                    removed_lines=removed_lines,
                    parsed_files=parsed_files,
                )
            except click.ClickException as exc:
                existing_summary: dict[str, Any] = {}
                if isinstance(final_json, Mapping):
                    summary_value = final_json.get("summary")
                    if isinstance(summary_value, Mapping):
                        existing_summary = {str(k): v for k, v in summary_value.items()}

                validated = {
                    "violations": [],
                    "summary": build_fallback_summary(len(parsed_files), existing_summary),
                }
                LOGGER.debug("Reviewer output discarded due to validation failure: %s", exc)

            return validated, parsed_files, False

        datasets: list[tuple[int, str, list[Mapping[str, Any]]]] = []
        compiled_pattern: re.Pattern[str] | None = None
        if filter_pattern:
            try:
                compiled_pattern = re.compile(filter_pattern)
            except re.error:
                LOGGER.warning(
                    "Invalid filter pattern '%s', processing without filter", filter_pattern
                )
                filter_pattern = None
                compiled_pattern = None

        token_budget = getattr(settings, "review_token_budget", 80_000)
        chunks = _refine_chunks_for_token_budget(
            chunks,
            rule_text=rule_content,
            filter_pattern=filter_pattern,
            token_budget=token_budget,
        )
        max(1, len(chunks))
        use_chunking = len(chunks) > 1

        for idx, chunk in enumerate(chunks):
            sub_patch = {"files": chunk}
            dataset_text = format_patch_dataset(sub_patch, filter_pattern=filter_pattern)

            if compiled_pattern:
                filtered_chunk = [
                    entry
                    for entry in chunk
                    if isinstance(entry.get("path"), str) and compiled_pattern.search(entry["path"])
                ]
            else:
                filtered_chunk = list(chunk)

            if not filtered_chunk:
                continue

            datasets.append((idx, dataset_text, filtered_chunk))

        if not datasets:
            reset_session(original_session_id)
            summary = build_fallback_summary(0)
            return {"violations": [], "summary": summary}

        aggregated_violations: list[dict[str, Any]] = []
        seen_violations: set[tuple[str, int]] = set()  # Track (file, line) to deduplicate overlaps
        files_reviewed: set[str] = set()
        last_summary: dict[str, Any] | None = None
        unverified_chunks: list[int] = []

        total_expected_chunks = len(datasets) if use_chunking else None
        pending = deque(
            (idx, dataset_text, chunk_subset, False) for idx, dataset_text, chunk_subset in datasets
        )

        while pending:
            chunk_index, dataset_text, chunk_subset, retried = pending.popleft()

            if retried:
                target_workspace = original_workspace_root or settings.workspace_root
            else:
                target_workspace = review_workspace or settings.workspace_root
            if target_workspace is not None:
                settings.workspace_root = target_workspace

            validated, parsed_files, needs_retry = execute_dataset(
                dataset_text,
                chunk_subset,
                chunk_index=chunk_index if use_chunking else None,
                total_expected_chunks=total_expected_chunks,
            )

            if needs_retry:
                if not retried:
                    pending.append((chunk_index, dataset_text, chunk_subset, True))
                    continue
                unverified_chunks.append(chunk_index)
                validated = {
                    "violations": [],
                    "summary": build_fallback_summary(len(chunk_subset), {"unverified": True}),
                }
                parsed_files = {
                    entry.get("path", "")
                    for entry in chunk_subset
                    if isinstance(entry.get("path"), str)
                }

            # Deduplicate violations from overlapped regions (file + line uniqueness)
            for violation in validated["violations"]:
                file_path = violation.get("file", "")
                line_number = violation.get("line", 0)
                message_text = violation.get("message", "")
                severity_value = violation.get("severity", "")
                change_type = violation.get("change_type", "")
                violation_key = (file_path, line_number, severity_value, message_text, change_type)

                if violation_key not in seen_violations:
                    seen_violations.add(violation_key)
                    aggregated_violations.append(violation)

            files_reviewed.update(parsed_files)
            last_summary = validated["summary"]

            if review_workspace is not None:
                settings.workspace_root = review_workspace

        final_summary: dict[str, Any] = dict(last_summary or {})
        final_summary["total_violations"] = len(aggregated_violations)
        final_summary["files_reviewed"] = len(files_reviewed)
        rule_name_value = final_summary.get("rule_name")
        if not isinstance(rule_name_value, str) or not rule_name_value.strip():
            final_summary["rule_name"] = rule_path.stem
        if unverified_chunks:
            final_summary["unverified_chunks"] = len(unverified_chunks)

        reset_session(original_session_id)

        return {
            "violations": aggregated_violations,
            "summary": final_summary,
        }
    finally:
        if original_workspace_root is not None:
            settings.workspace_root = original_workspace_root
        if original_max_tool_output_chars is not None:
            settings.max_tool_output_chars = original_max_tool_output_chars
        if original_max_context_tokens is not None:
            settings.max_context_tokens = original_max_context_tokens
        if original_response_headroom is not None:
            settings.response_headroom_tokens = original_response_headroom
        # Dynamic instructions feature has been removed - no restoration needed


__all__ = [
    "collect_patch_review_data",
    "format_patch_dataset",
    "parse_patch_file",
    "run_review",
    "validate_review_response",
]
