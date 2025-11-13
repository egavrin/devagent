"""Registry-backed intent handlers used by the CLI."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import click

from ai_dev_agent.core.utils.tool_utils import expand_tool_aliases
from ai_dev_agent.tools import EDIT, FIND, GREP, READ, RUN, SYMBOLS

from ..utils import _invoke_registry_tool, _normalize_argument_list, build_system_context

if TYPE_CHECKING:
    from ai_dev_agent.core.utils.config import Settings

_REGEX_HINT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?<!\\)\\[AbBdDsSwWZA]"),
    re.compile(r"(?<!\\)\\[pP]\{"),
    re.compile(r"(?<!\\)\\k<"),
    re.compile(r"(?<!\\)\\x[0-9A-Fa-f]{2}"),
    re.compile(r"(?<!\\)\\u[0-9A-Fa-f]{4}"),
    re.compile(r"(?<!\\)\[[^\]]+\]"),
    re.compile(r"(?<!\\)\(\?"),
    re.compile(r"(?<!\\)\{[0-9,\s]+\}"),
    re.compile(r"(?<!\\)\|"),
    re.compile(r"(?<!\\)\.\*"),
    re.compile(r"(?<!\\)\.\+"),
    re.compile(r"(?<!\\)\.\?"),
)


def _should_enable_regex(query: str) -> bool:
    if not query:
        return False

    stripped = query.strip()
    if stripped.startswith("^") and not stripped.startswith(r"\^"):
        return True
    if stripped.endswith("$") and not stripped.endswith(r"\$"):
        return True

    return any(pattern.search(query) for pattern in _REGEX_HINT_PATTERNS)


PayloadBuilder = Callable[[click.Context, dict[str, Any]], tuple[dict[str, Any], dict[str, Any]]]
ResultHandler = Callable[[click.Context, dict[str, Any], dict[str, Any], dict[str, Any]], None]
RecoveryHandler = Callable[
    [click.Context, dict[str, Any], dict[str, Any], dict[str, Any], Exception], dict[str, Any]
]


@dataclass
class RegistryIntent:
    """Reusable wrapper for registry-backed intent handlers."""

    tool_name: str
    payload_builder: PayloadBuilder
    result_handler: ResultHandler
    with_sandbox: bool = False
    recovery_handler: RecoveryHandler | None = None

    def __call__(self, ctx: click.Context, arguments: dict[str, Any]) -> Mapping[str, Any]:
        payload, context = self.payload_builder(ctx, arguments)
        extras = context or {}
        try:
            result = _invoke_registry_tool(
                ctx,
                self.tool_name,
                payload,
                with_sandbox=self.with_sandbox,
            )
        except Exception as exc:  # pragma: no cover - defensive
            if not self.recovery_handler:
                raise
            result = self.recovery_handler(ctx, arguments, payload, extras, exc)
        self.result_handler(ctx, arguments, result, extras)
        return result


def _build_find_payload(
    ctx: click.Context, arguments: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    query = str(arguments.get("query", "")).strip()
    if not query:
        raise click.ClickException("find requires a 'query' argument.")

    payload: dict[str, Any] = {"query": query}
    if "path" in arguments and arguments["path"] is not None:
        payload["path"] = str(arguments["path"])

    if "limit" in arguments and arguments["limit"] is not None:
        try:
            payload["limit"] = max(1, int(arguments["limit"]))
        except (TypeError, ValueError):
            raise click.ClickException("--limit must be an integer") from None

    fuzzy = arguments.get("fuzzy")
    if fuzzy is not None:
        payload["fuzzy"] = bool(fuzzy)

    return payload, {}


def _build_grep_payload(
    ctx: click.Context, arguments: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    pattern = str(arguments.get("pattern", "")).strip()
    if not pattern:
        raise click.ClickException("grep requires a 'pattern' argument.")

    payload: dict[str, Any] = {"pattern": pattern}
    if "path" in arguments and arguments["path"] is not None:
        payload["path"] = str(arguments["path"])

    if "regex" in arguments:
        payload["regex"] = bool(arguments["regex"])
    elif _should_enable_regex(pattern):
        payload["regex"] = True

    if "limit" in arguments and arguments["limit"] is not None:
        try:
            payload["limit"] = max(1, int(arguments["limit"]))
        except (TypeError, ValueError):
            raise click.ClickException("--limit must be an integer") from None

    return payload, {}


def _build_symbols_payload(
    ctx: click.Context, arguments: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    name = str(arguments.get("name", "")).strip()
    if not name:
        raise click.ClickException("symbols requires a 'name' argument.")

    payload: dict[str, Any] = {"name": name}

    if "path" in arguments and arguments["path"] is not None:
        payload["path"] = str(arguments["path"])

    if "limit" in arguments and arguments["limit"] is not None:
        try:
            payload["limit"] = max(1, int(arguments["limit"]))
        except (TypeError, ValueError):
            raise click.ClickException("--limit must be an integer") from None

    if arguments.get("kind"):
        payload["kind"] = str(arguments["kind"])

    if arguments.get("lang"):
        payload["lang"] = str(arguments["lang"])

    return payload, {}


def _build_read_payload(
    ctx: click.Context, arguments: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = _normalize_argument_list(arguments, plural_key="paths", singular_key="path")
    if not paths:
        raise click.ClickException("read requires 'paths' (or a single 'path').")

    payload: dict[str, Any] = {"paths": paths}
    if "context_lines" in arguments:
        try:
            payload["context_lines"] = int(arguments.get("context_lines"))
        except (TypeError, ValueError):
            pass

    if "byte_range" in arguments:
        byte_range = arguments.get("byte_range")
        if isinstance(byte_range, (list, tuple)) and len(byte_range) == 2:
            payload["byte_range"] = [int(byte_range[0]), int(byte_range[1])]

    return payload, {}


def _handle_read_result(
    ctx: click.Context,
    arguments: dict[str, Any],
    result: dict[str, Any],
    _: dict[str, Any],
) -> None:
    files = result.get("files", [])
    if not files:
        click.echo("No content returned.")
        return

    slicing_requested = any(key in arguments for key in ("start_line", "end_line", "max_lines"))

    try:
        start_line = int(arguments.get("start_line", 1) or 1)
    except (TypeError, ValueError):
        raise click.ClickException("start_line must be an integer.")
    if start_line < 1:
        start_line = 1

    end_line_value = arguments.get("end_line")
    if end_line_value is not None:
        try:
            end_line = int(end_line_value)
        except (TypeError, ValueError):
            raise click.ClickException("end_line must be an integer.")
        if end_line < start_line:
            raise click.ClickException("end_line must be greater than or equal to start_line.")
    else:
        end_line = None

    max_lines = None
    if end_line is None and arguments.get("max_lines") is not None:
        try:
            max_lines = int(arguments.get("max_lines"))
        except (TypeError, ValueError):
            raise click.ClickException("max_lines must be an integer.")
        if max_lines <= 0:
            max_lines = 200

    settings: Settings = ctx.obj["settings"]
    default_window = max(1, getattr(settings, "fs_read_default_max_lines", 200))

    for entry in files:
        rel_path = entry.get("path") or "(unknown)"
        content = entry.get("content", "")
        if not slicing_requested:
            lines = content.splitlines()
            total_lines = len(lines)
            if not lines:
                click.echo(f"== {rel_path} ==")
                click.echo("(empty)")
                continue

            window = min(default_window, total_lines)
            snippet = lines[:window]
            click.echo(f"== {rel_path} ==")
            for line_number, text in enumerate(snippet, start=1):
                click.echo(f"{line_number:5}: {text.rstrip()}")
            if window < total_lines:
                remaining = total_lines - window
                click.echo(
                    f"... ({remaining} more lines not shown; specify --start-line/--end-line to expand)"
                )
            continue

        lines = content.splitlines()
        total_lines = len(lines)
        if total_lines == 0:
            click.echo(f"{rel_path} is empty.")
            continue

        start_index = min(start_line - 1, total_lines)
        if start_index >= total_lines:
            click.echo(
                f"File {rel_path} has only {total_lines} lines; nothing to show from line {start_line}."
            )
            continue

        if end_line is not None:
            end_index = min(end_line, total_lines)
        else:
            window = max_lines or 200
            end_index = min(start_index + window, total_lines)
            if end_index == start_index:
                end_index = min(start_index + 200, total_lines)

        snippet = lines[start_index:end_index]
        if snippet:
            last_line = start_index + len(snippet)
            click.echo(
                f"Reading {rel_path} (lines {start_index + 1}-{last_line} of {total_lines}):"
            )
            for line_number, text in enumerate(snippet, start=start_index + 1):
                click.echo(f"{line_number:5}: {text.rstrip()}")
            if end_index < total_lines:
                remaining = total_lines - end_index
                click.echo(f"... ({remaining} more lines not shown)")
        else:
            click.echo(f"No content available in the requested range for {rel_path}.")


def _handle_symbols_index_result(
    _: click.Context,
    __: dict[str, Any],
    result: dict[str, Any],
    ___: dict[str, Any],
) -> None:
    stats = result.get("stats", {})
    files_indexed = stats.get("files_indexed")
    symbols = stats.get("symbols")
    db_path = result.get("db_path")

    click.echo("Symbol index updated.")
    if files_indexed is not None or symbols is not None:
        parts = [
            f"{files_indexed} file(s)" if files_indexed is not None else None,
            f"{symbols} symbol(s)" if symbols is not None else None,
        ]
        click.echo("Indexed " + ", ".join(part for part in parts if part))
    if db_path:
        click.echo(f"Index written to {db_path}")


def _build_exec_payload(
    ctx: click.Context, arguments: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    ctx_obj = ctx.obj if isinstance(ctx.obj, dict) else {}
    system_context = (
        ctx_obj.setdefault("_system_context", build_system_context())
        if isinstance(ctx_obj, dict)
        else build_system_context()
    )
    cmd_value = arguments.get("cmd") or arguments.get("command")
    cmd = str(cmd_value or "").strip()
    if not cmd:
        raise click.ClickException(
            f"{RUN} requires 'cmd'. Received arguments: "
            f"{arguments}. System: {system_context.get('os')}. "
            "Ensure the LLM provided a valid command for this platform."
        )

    if system_context.get("os") == "Windows" and cmd.startswith("ls"):
        cmd = cmd.replace("ls", "dir", 1)

    payload: dict[str, Any] = {"cmd": cmd}
    if arguments.get("args"):
        payload["args"] = [str(a) for a in (arguments.get("args") or [])]
    if arguments.get("cwd"):
        payload["cwd"] = str(arguments.get("cwd"))

    timeout_value = None
    if arguments.get("timeout_sec") is not None:
        timeout_value = arguments.get("timeout_sec")
    elif arguments.get("timeout") is not None:
        timeout_value = arguments.get("timeout")
    if timeout_value is not None:
        try:
            payload["timeout_sec"] = int(timeout_value)
        except (TypeError, ValueError):
            raise click.ClickException("timeout must be an integer.")

    return payload, {}


def _handle_exec_result(
    _: click.Context,
    __: dict[str, Any],
    result: dict[str, Any],
    ___: dict[str, Any],
) -> None:
    exit_code = result.get("exit_code", 0)
    click.echo(f"Exit: {exit_code}")
    if result.get("stdout_tail"):
        click.echo(result["stdout_tail"].rstrip())
    if result.get("stderr_tail"):
        click.echo(result["stderr_tail"].rstrip())


# WRITE tool has been removed - use EDIT instead which supports both
# SEARCH/REPLACE blocks and unified diff formats


def _handle_edit_result(
    _: click.Context,
    __: dict[str, Any],
    result: dict[str, Any],
    ___: dict[str, Any],
) -> None:
    """Handle EDIT tool results (both SEARCH/REPLACE and unified diff formats)."""
    success = result.get("success", False)
    changed = result.get("changed_files", [])
    new = result.get("new_files", [])

    if success:
        click.echo("Patch applied")
    else:
        click.echo("Patch failed to apply")
        errors = result.get("errors", [])
        if errors:
            click.echo("Errors:")
            for error in errors:
                click.echo(f"- {error}")

    all_files = list(set(changed + new))
    if all_files:
        label = "Changed files:" if changed else "New files:" if new else "Modified files:"
        click.echo(label)
        for filename in all_files:
            click.echo(f"- {filename}")


def _handle_simple_result(ctx, arguments, result, _) -> None:
    """Simple result handler for tools that return clean output."""
    # Echo file lists
    if "files" in result:
        for entry in result["files"]:
            if isinstance(entry, Mapping):
                path = entry.get("path", str(entry))
                score = entry.get("score")
                meta_parts = []
                if entry.get("lines") is not None:
                    meta_parts.append(f"{entry['lines']} lines")
                if entry.get("size_bytes") is not None:
                    size_kb = (
                        entry["size_bytes"] / 1024
                        if isinstance(entry["size_bytes"], (int, float))
                        else None
                    )
                    if size_kb is not None:
                        meta_parts.append(f"{size_kb:.1f} KB")
                if entry.get("mtime"):
                    meta_parts.append(entry["mtime"])
                if score is not None:
                    meta_parts.append(f"score {score}")
                suffix = f" ({', '.join(meta_parts)})" if meta_parts else ""
                click.echo(f"{path}{suffix}")
            else:
                click.echo(str(entry))

    # Echo grep matches
    elif "matches" in result:
        for match_group in result["matches"]:
            if isinstance(match_group, str):
                click.echo(match_group)
                continue

            file_path = match_group.get("file") if isinstance(match_group, dict) else None
            if file_path:
                click.echo(f"\n{file_path}:")

            entries = match_group.get("matches", []) if isinstance(match_group, dict) else []
            for entry in entries:
                if isinstance(entry, dict):
                    line_no = entry.get("line", "?")
                    text = entry.get("text", "")
                else:
                    line_no = "?"
                    text = str(entry)
                click.echo(f"  {line_no}: {text}")

    # Echo symbols
    elif "symbols" in result:
        for symbol in result["symbols"]:
            click.echo(
                f"{symbol['file']}:{symbol.get('line', '?')} - {symbol['name']} ({symbol.get('kind', 'unknown')})"
            )

    # Echo errors
    if "error" in result:
        click.secho(f"Error: {result['error']}", fg="red")


REGISTRY_INTENTS: dict[str, RegistryIntent] = {
    FIND: RegistryIntent(
        tool_name=FIND,
        payload_builder=_build_find_payload,
        result_handler=_handle_simple_result,
    ),
    GREP: RegistryIntent(
        tool_name=GREP,
        payload_builder=_build_grep_payload,
        result_handler=_handle_simple_result,
    ),
    SYMBOLS: RegistryIntent(
        tool_name=SYMBOLS,
        payload_builder=_build_symbols_payload,
        result_handler=_handle_simple_result,
    ),
    READ: RegistryIntent(
        tool_name=READ,
        payload_builder=_build_read_payload,
        result_handler=_handle_read_result,
    ),
    RUN: RegistryIntent(
        tool_name=RUN,
        payload_builder=_build_exec_payload,
        result_handler=_handle_exec_result,
        with_sandbox=False,
    ),
    EDIT: RegistryIntent(
        tool_name=EDIT,
        payload_builder=lambda ctx, args: (args, {}),  # Pass through arguments
        result_handler=_handle_edit_result,
    ),
    # WRITE has been removed - EDIT now handles both SEARCH/REPLACE and unified diffs
}

INTENT_HANDLERS: dict[str, RegistryIntent] = expand_tool_aliases(REGISTRY_INTENTS)
