"""Review command helpers and execution logic."""
from __future__ import annotations

from importlib import import_module
import fnmatch
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple
from uuid import uuid4

import click

import re

from ai_dev_agent.agents.schemas import VIOLATION_SCHEMA
from ai_dev_agent.core.utils.config import Settings
from ai_dev_agent.core.utils.logger import get_logger
from ai_dev_agent.tools.patch_analysis import PatchParser

from .react.executor import _execute_react_assistant
from .review_context import ContextOrchestrator, SourceContextProvider
from .utils import _record_invocation, get_llm_client

LOGGER = get_logger(__name__)

# Cache for parsed patches: key=(path, mtime, size) -> parsed_data
_PATCH_CACHE: Dict[Tuple[str, float, Optional[int]], Dict[str, Any]] = {}


def _normalize_applies_to_pattern(raw: str) -> Optional[str]:
    """Convert a glob-style rule pattern into a regex string."""
    if not raw:
        return None

    # Split on commas or whitespace to support simple multi-pattern lists
    tokens = [token.strip() for token in re.split(r"[,\s]+", raw) if token.strip()]
    if not tokens:
        return None

    regex_parts: List[str] = []
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


def extract_applies_to_pattern(rule_content: str) -> Optional[str]:
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
        r'##\s*Applies\s+To\s*\n\s*([^\n]+)',  # ## Applies To\n  pattern
        r'Applies\s+To:\s*([^\n]+)',            # Applies To: pattern
        r'scope:\s*([^\n]+)',                   # scope: pattern
        r'##\s*Scope\s*\n\s*([^\n]+)',         # ## Scope\n  pattern
    ]

    for pattern in patterns:
        match = re.search(pattern, rule_content, re.IGNORECASE)
        if match:
            extracted = match.group(1).strip()
            # Remove quotes if present
            extracted = extracted.strip('"\'`')
            if extracted:
                normalized = _normalize_applies_to_pattern(extracted)
                if normalized:
                    LOGGER.debug("Extracted 'Applies To' pattern (normalized): %s -> %s", extracted, normalized)
                    return normalized
                LOGGER.debug("Extracted 'Applies To' pattern: %s", extracted)
                return extracted

    LOGGER.debug("No 'Applies To' pattern found in rule")
    return None


def parse_patch_file(patch_path: Path) -> Dict[str, Any]:
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
    parsed_patch: Mapping[str, Any]
) -> Tuple[Dict[str, Dict[int, str]], Dict[str, Dict[int, str]], Set[str]]:
    """Collect lookup tables for added/removed lines and parsed file set."""

    added_lines: Dict[str, Dict[int, str]] = {}
    removed_lines: Dict[str, Dict[int, str]] = {}
    parsed_files: Set[str] = set()
    for file_entry in parsed_patch.get("files", []):
        path = file_entry.get("path")
        if not isinstance(path, str):
            continue
        parsed_files.add(path)
        added_lookup: Dict[int, str] = {}
        removed_lookup: Dict[int, str] = {}
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


def format_patch_dataset(parsed_patch: Mapping[str, Any], filter_pattern: Optional[str] = None) -> str:
    """Format parsed patch data into a text block for the reviewer.

    Args:
        parsed_patch: Parsed patch structure from PatchParser
        filter_pattern: Optional regex pattern to filter files (e.g., from rule's "Applies To")

    Returns:
        Formatted text representation of patch data
    """
    lines: List[str] = []
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
) -> List[Mapping[str, Any]]:
    if max_hunks_per_group <= 0 and max_lines_per_group <= 0:
        return list(files)

    results: List[Mapping[str, Any]] = []
    for file_entry in files:
        hunks = list(file_entry.get("hunks", []) or [])
        if not hunks:
            results.append(file_entry)
            continue

        groups: List[List[Mapping[str, Any]]] = []
        current: List[Mapping[str, Any]] = []
        hunk_counter = 0
        line_counter = 0

        for hunk in hunks:
            impact = _estimate_hunk_impact(hunk)
            should_split = (
                current
                and (
                    (max_hunks_per_group > 0 and hunk_counter >= max_hunks_per_group)
                    or (max_lines_per_group > 0 and line_counter + impact > max_lines_per_group)
                )
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
) -> List[List[Mapping[str, Any]]]:
    """Split patch files into manageable chunks."""
    if not files:
        return []

    effective_max_files = max_files_per_chunk if max_files_per_chunk > 0 else len(files)
    effective_max_lines = max_lines_per_chunk if max_lines_per_chunk > 0 else 0

    chunked: List[List[Mapping[str, Any]]] = []
    current_chunk: List[Mapping[str, Any]] = []
    current_lines = 0

    for entry in files:
        entry_lines = _estimate_entry_lines(entry)
        should_split = (
            current_chunk
            and (
                (effective_max_files and len(current_chunk) >= effective_max_files)
                or (effective_max_lines and current_lines + entry_lines > effective_max_lines)
            )
        )
        if should_split:
            chunked.append(current_chunk)
            current_chunk = []
            current_lines = 0
        current_chunk.append(entry)
        current_lines += entry_lines

    if current_chunk:
        chunked.append(current_chunk)

    return chunked


def _compute_dynamic_line_limit(
    configured_limit: int,
    rule_text: str,
    files: Sequence[Mapping[str, Any]],
) -> int:
    """Derive a chunk line limit based on rule size and patch impact."""
    DEFAULT_LINES_LIMIT = 320
    base_limit = configured_limit if configured_limit > 0 else DEFAULT_LINES_LIMIT

    total_line_impact = sum(_estimate_entry_lines(entry) for entry in files)
    rule_token_estimate = max(len(rule_text) // 4, 0)

    if rule_token_estimate > 9_000 or total_line_impact > 800:
        base_limit = min(base_limit, 80)
    elif rule_token_estimate > 6_000 or total_line_impact > 600:
        base_limit = min(base_limit, 120)
    elif rule_token_estimate > 4_000 or total_line_impact > 400:
        base_limit = min(base_limit, 160)

    if configured_limit <= 0 and base_limit == DEFAULT_LINES_LIMIT:
        return 0
    return base_limit


def validate_review_response(
    response: Dict[str, Any],
    *,
    added_lines: Dict[str, Dict[int, str]],
    removed_lines: Optional[Dict[str, Dict[int, str]]] = None,
    parsed_files: Set[str],
) -> Dict[str, Any]:
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

    valid_violations: List[Dict[str, Any]] = []
    discarded_entries: List[str] = []

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

        sanitized: Dict[str, Any] = dict(entry)
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
    context_section: Optional[str] = None,
) -> str:
    context_block = ""
    if context_section:
        context_block = f"\nAdditional Context:\n{context_section}\n"
    return f"""Review the patch file {patch_rel} against the coding rule provided below.

Follow this workflow:
1. Review the rule below to understand:
   - Rule title and description
   - "Applies To" pattern (file scope)
   - Examples of violations and compliant code

2. Examine the pre-parsed Patch Dataset included at the end of this message.
   - Each file shows only added lines with their final line numbers.
   - Use this dataset as the single source of truth for code under review.

3. For each added line in scope:
   - Check if it violates the rule
   - Record file path, line number, and violation description

4. Output the results in the required JSON format with all violations found.

{context_block}
Patch Dataset:
{patch_dataset}
"""


def run_review(
    ctx: click.Context,
    *,
    patch_file: str,
    rule_file: str,
    json_output: bool,
    settings: Settings,
) -> Dict[str, Any]:
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

    try:
        review_workspace: Optional[Path] = None
        common_parts: List[Path] = []
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
            if (original_workspace_root is None or candidate_within_original) and not candidate_is_root:
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

        max_lines_setting = getattr(settings, "review_max_lines_per_chunk", 320)
        dynamic_line_limit = _compute_dynamic_line_limit(
            max_lines_setting,
            rule_content,
            original_files,
        )

        expanded_files = _split_large_file_entries(
            original_files,
            max_hunks_per_group=getattr(settings, "review_max_hunks_per_chunk", 8),
            max_lines_per_group=dynamic_line_limit if dynamic_line_limit > 0 else max_lines_setting,
        )

        max_files_per_chunk_setting = getattr(settings, "review_max_files_per_chunk", 10)
        if max_files_per_chunk_setting <= 0:
            max_files_per_chunk = len(expanded_files) or 1
        else:
            max_files_per_chunk = max_files_per_chunk_setting

        max_lines_per_chunk = dynamic_line_limit

        chunks = _chunk_patch_files(
            expanded_files,
            max_files_per_chunk=max_files_per_chunk,
            max_lines_per_chunk=max_lines_per_chunk,
        )
        total_chunks = max(1, len(chunks))
        use_chunking = len(chunks) > 1

        context_orchestrator = ContextOrchestrator(
            [SourceContextProvider(
                pad_lines=getattr(settings, "review_context_pad_lines", 20),
                max_lines_per_item=getattr(settings, "review_context_max_lines", 160),
            )],
            max_total_lines=getattr(settings, "review_context_max_total_lines", 320),
        )

        def build_fallback_summary(
            file_count: int,
            existing_summary: Optional[Mapping[str, Any]] = None,
        ) -> Dict[str, Any]:
            summary: Dict[str, Any] = dict(existing_summary or {})
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

        _record_invocation(ctx, overrides={"patch": patch_file, "rule": rule_file, "mode": "review"})

        if json_output:
            ctx.obj["silent_mode"] = True

        # Configure aggressive context management for review sessions
        # Reviews can accumulate large context through multiple iterations,
        # especially with large patches and verbose rules
        settings.max_tool_output_chars = 1_000_000
        settings.max_context_tokens = 100_000  # Conservative limit to prevent overflow
        settings.response_headroom_tokens = 10_000  # Leave room for response generation

        try:
            cli_pkg = import_module('ai_dev_agent.cli')
            llm_factory = getattr(cli_pkg, 'get_llm_client', get_llm_client)
        except ModuleNotFoundError:
            llm_factory = get_llm_client

        try:
            client = llm_factory(ctx)
        except click.ClickException as exc:
            raise click.ClickException(f'Failed to create LLM client: {exc}') from exc

        original_session_id = ctx.obj.get("_session_id")
        review_session_id = f"review-{uuid4()}"
        ctx.obj["_session_id"] = review_session_id

        def reset_session(previous: Optional[str]) -> None:
            if previous:
                ctx.obj["_session_id"] = previous
            else:
                ctx.obj.pop("_session_id", None)

        def execute_dataset(
            dataset_text: str,
            chunk_files: List[Mapping[str, Any]],
            *,
            chunk_index: Optional[int] = None,
            total_expected_chunks: Optional[int] = None,
        ) -> Tuple[Dict[str, Any], Set[str]]:
            if not chunk_files:
                summary = build_fallback_summary(0)
                return {"violations": [], "summary": summary}, set()

            subset_patch = {"files": chunk_files}
            added_lines, removed_lines, parsed_files = collect_patch_review_data(subset_patch)

            chunk_header = ""
            if chunk_index is not None and total_expected_chunks and total_expected_chunks > 1:
                chunk_header = f"(Chunk {chunk_index + 1} of {total_expected_chunks})\n\n"

            context_section = context_orchestrator.build_section(
                settings.workspace_root,
                chunk_files,
            )

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
                if "Assistant response did not contain valid JSON matching the required schema" in message_text:
                    LOGGER.debug("LLM output missing required JSON; returning fallback summary: %s", exc)
                    summary = build_fallback_summary(len(parsed_files))
                    return {"violations": [], "summary": summary}, parsed_files
                raise

            if not isinstance(execution_payload, dict):
                raise click.ClickException("Review execution did not return diagnostics for validation.")

            run_result = execution_payload.get("result")
            if run_result is None:
                raise click.ClickException("Review execution missing run metadata; cannot validate results.")

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
                existing_summary: Dict[str, Any] = {}
                if isinstance(final_json, Mapping):
                    summary_value = final_json.get("summary")
                    if isinstance(summary_value, Mapping):
                        existing_summary = {str(k): v for k, v in summary_value.items()}

                validated = {
                    "violations": [],
                    "summary": build_fallback_summary(len(parsed_files), existing_summary),
                }
                LOGGER.debug("Reviewer output discarded due to validation failure: %s", exc)

            return validated, parsed_files

        datasets: List[Tuple[str, List[Mapping[str, Any]], Optional[int]]] = []

        compiled_pattern: Optional[re.Pattern[str]] = None
        if filter_pattern:
            try:
                compiled_pattern = re.compile(filter_pattern)
            except re.error:
                LOGGER.warning("Invalid filter pattern '%s', processing without filter", filter_pattern)
                filter_pattern = None
                compiled_pattern = None

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

            datasets.append(
                (
                    dataset_text,
                    filtered_chunk,
                    idx if use_chunking else None,
                )
            )

        if not datasets:
            reset_session(original_session_id)
            summary = build_fallback_summary(0)
            return {"violations": [], "summary": summary}

        aggregated_violations: List[Dict[str, Any]] = []
        files_reviewed: Set[str] = set()
        last_summary: Optional[Dict[str, Any]] = None

        for dataset_text, chunk_subset, chunk_index in datasets:
            validated, parsed_files = execute_dataset(
                dataset_text,
                chunk_subset,
                chunk_index=chunk_index,
                total_expected_chunks=total_chunks if use_chunking else None,
            )
            aggregated_violations.extend(validated["violations"])
            files_reviewed.update(parsed_files)
            last_summary = validated["summary"]

        final_summary: Dict[str, Any] = dict(last_summary or {})
        final_summary["total_violations"] = len(aggregated_violations)
        final_summary["files_reviewed"] = len(files_reviewed)
        rule_name_value = final_summary.get("rule_name")
        if not isinstance(rule_name_value, str) or not rule_name_value.strip():
            final_summary["rule_name"] = rule_path.stem

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


__all__ = [
    "collect_patch_review_data",
    "format_patch_dataset",
    "parse_patch_file",
    "run_review",
    "validate_review_response",
]
