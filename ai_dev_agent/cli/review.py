"""Review command helpers and execution logic."""
from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import click

import re

from ai_dev_agent.agents.schemas import VIOLATION_SCHEMA
from ai_dev_agent.core.utils.config import Settings
from ai_dev_agent.core.utils.logger import get_logger
from ai_dev_agent.tools.patch_analysis import PatchParser

from .react.executor import _execute_react_assistant
from .utils import _record_invocation, get_llm_client

LOGGER = get_logger(__name__)

# Cache for parsed patches: key=(path, mtime) -> parsed_data
_PATCH_CACHE: Dict[Tuple[str, float], Dict[str, Any]] = {}


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
        cache_key = (str(patch_path), stat.st_mtime)

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

    parser = PatchParser(content, include_context=False)
    parsed = parser.parse()

    # Store in cache
    try:
        cache_key = (str(patch_path), stat.st_mtime)
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


def collect_patch_review_data(parsed_patch: Mapping[str, Any]) -> Tuple[Dict[str, Dict[int, str]], Set[str]]:
    """Collect lookup tables for added lines and parsed file set from parsed patch data."""

    added_lines: Dict[str, Dict[int, str]] = {}
    parsed_files: Set[str] = set()
    for file_entry in parsed_patch.get("files", []):
        path = file_entry.get("path")
        if not isinstance(path, str):
            continue
        parsed_files.add(path)
        line_lookup: Dict[int, str] = {}
        for hunk in file_entry.get("hunks", []):
            if not isinstance(hunk, Mapping):
                continue
            for added in hunk.get("added_lines", []):
                if not isinstance(added, Mapping):
                    continue
                line_no = added.get("line_number")
                content_value = added.get("content")
                if isinstance(line_no, int) and isinstance(content_value, str):
                    line_lookup[line_no] = content_value
        if line_lookup:
            added_lines[path] = line_lookup

    return added_lines, parsed_files


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
        lines.append(f"FILE: {path}")
        lines.append(f"  Change type: {change_type}")
        lines.append(f"  Language: {language}")

        added: List[Tuple[int, str]] = []
        for hunk in file_entry.get("hunks", []) or []:
            for line in hunk.get("added_lines", []) or []:
                line_number = line.get("line_number")
                content = line.get("content")
                if isinstance(line_number, int) and isinstance(content, str):
                    added.append((line_number, content))

        lines.append(f"  Total added lines: {len(added)}")
        lines.append("  ADDED LINES:")
        if added:
            for line_number, content in added:
                lines.append(f"    {line_number:4d} | {content}")
        else:
            lines.append("    (none)")
        lines.append("")  # Blank line between files

    return "\n".join(lines).rstrip()


def validate_review_response(
    response: Dict[str, Any],
    *,
    added_lines: Dict[str, Dict[int, str]],
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

    files_with_additions = set(added_lines.keys())
    invalid_entries: List[str] = []
    for entry in violations:
        if not isinstance(entry, Mapping):
            invalid_entries.append("non-object violation entry")
            continue
        file_path = entry.get("file")
        line_number = entry.get("line")
        snippet = entry.get("code_snippet")

        if not isinstance(file_path, str) or not isinstance(line_number, int):
            invalid_entries.append(str(entry))
            continue

        added_for_file = added_lines.get(file_path)
        if added_for_file is None:
            invalid_entries.append(f"{file_path}:{line_number} (file not in patch)")
            continue

        actual_line = added_for_file.get(line_number)
        if actual_line is None:
            invalid_entries.append(f"{file_path}:{line_number} (line not added)")
            continue

        if isinstance(snippet, str) and snippet.strip() != actual_line.strip():
            invalid_entries.append(f"{file_path}:{line_number} (content mismatch)")

    if invalid_entries:
        preview = ", ".join(invalid_entries[:5])
        if len(invalid_entries) > 5:
            preview += ", ..."
        raise click.ClickException(
            "Reviewer output referenced lines not present in the patch: "
            f"{preview}"
        )

    total_violations = summary.get("total_violations")
    if isinstance(total_violations, int) and total_violations != len(violations):
        LOGGER.debug(
            "Adjusting total_violations from %s to %s based on validated entries",
            total_violations,
            len(violations),
        )

    files_reviewed = summary.get("files_reviewed")
    if isinstance(files_reviewed, int) and files_reviewed != len(parsed_files):
        LOGGER.debug(
            "Adjusting files_reviewed from %s to %s based on parsed files",
            files_reviewed,
            len(parsed_files),
        )

    normalized_summary = dict(summary)
    normalized_summary["total_violations"] = len(violations)
    normalized_summary["files_reviewed"] = len(parsed_files)

    rule_name = normalized_summary.get("rule_name")
    if not isinstance(rule_name, str) or not rule_name.strip():
        normalized_summary["rule_name"] = normalized_summary.get("rule_name") or ""

    return {
        "violations": list(violations),
        "summary": normalized_summary,
    }


def _build_review_prompt(patch_rel: Path, patch_dataset: str) -> str:
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

    common_parts = []
    for p1, p2 in zip(patch_path.parents, rule_path.parents):
        if p1 == p2:
            common_parts.append(p1)
            break

    if common_parts:
        review_workspace = common_parts[0]
        settings.workspace_root = review_workspace
        patch_rel = patch_path.relative_to(review_workspace)
    else:
        patch_rel = patch_path

    # Extract "Applies To" pattern from rule to filter patch dataset
    filter_pattern = extract_applies_to_pattern(rule_content)

    parsed_patch = parse_patch_file(patch_path)
    patch_dataset = format_patch_dataset(parsed_patch, filter_pattern=filter_pattern)
    added_lines, parsed_files = collect_patch_review_data(parsed_patch)

    def build_fallback_summary(existing_summary: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        summary: Dict[str, Any] = dict(existing_summary or {})
        summary["total_violations"] = 0
        summary["files_reviewed"] = len(parsed_files)
        rule_name_value = summary.get("rule_name")
        if not isinstance(rule_name_value, str) or not rule_name_value.strip():
            summary["rule_name"] = rule_path.stem
        return summary

    prompt = _build_review_prompt(patch_rel, patch_dataset)

    system_extension_with_rule = f"""# Coding Rule: {rule_path.name}

{rule_content}

---

The above rule has been pre-loaded for your review. Do NOT use the read tool to access it.
Follow the workflow in the user prompt to analyze the patch against this rule."""

    _record_invocation(ctx, overrides={"patch": patch_file, "rule": rule_file, "mode": "review"})

    if json_output:
        ctx.obj["silent_mode"] = True

    settings.max_tool_output_chars = 1_000_000

    try:
        cli_pkg = import_module('ai_dev_agent.cli')
        llm_factory = getattr(cli_pkg, 'get_llm_client', get_llm_client)
    except ModuleNotFoundError:
        llm_factory = get_llm_client

    try:
        client = llm_factory(ctx)
    except click.ClickException as exc:
        raise click.ClickException(f'Failed to create LLM client: {exc}') from exc

    try:
        execution_payload = _execute_react_assistant(
            ctx, client, settings, prompt,
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
            return {
                "violations": [],
                "summary": build_fallback_summary(),
            }
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
            "summary": build_fallback_summary(existing_summary),
        }
        LOGGER.debug("Reviewer output discarded due to validation failure: %s", exc)

    return validated


__all__ = [
    "collect_patch_review_data",
    "format_patch_dataset",
    "parse_patch_file",
    "run_review",
    "validate_review_response",
]
