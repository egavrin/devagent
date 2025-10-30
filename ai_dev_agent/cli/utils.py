"""Helper utilities shared across CLI commands."""

from __future__ import annotations

import os
import platform
import re
import shlex
import shutil
import tempfile
import uuid
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

from ai_dev_agent.core.approval.approvals import ApprovalManager
from ai_dev_agent.core.approval.policy import ApprovalPolicy
from ai_dev_agent.core.utils.constants import MAX_HISTORY_ENTRIES
from ai_dev_agent.core.utils.context_budget import BudgetedLLMClient, config_from_settings
from ai_dev_agent.core.utils.devagent_config import load_devagent_yaml
from ai_dev_agent.core.utils.keywords import extract_keywords
from ai_dev_agent.core.utils.logger import get_logger, set_correlation_id
from ai_dev_agent.core.utils.repo_outline import generate_repo_outline
from ai_dev_agent.core.utils.state import InMemoryStateStore, StateStore
from ai_dev_agent.providers.llm import DEEPSEEK_DEFAULT_BASE_URL, RetryConfig, create_client

# Sandbox execution removed - using direct execution only
from ai_dev_agent.session import SessionManager
from ai_dev_agent.session.context_service import ContextPruningConfig
from ai_dev_agent.session.summarizer import LLMConversationSummarizer
from ai_dev_agent.tools import ToolContext
from ai_dev_agent.tools import registry as tool_registry
from ai_dev_agent.tools.code.code_edit.tree_sitter_analysis import extract_symbols_from_outline

if TYPE_CHECKING:
    from ai_dev_agent.core.utils.config import Settings

LOGGER = get_logger(__name__)


_OS_FRIENDLY_NAMES = {
    "Darwin": "macOS",
    "Linux": "Linux",
    "Windows": "Windows",
}

_COMMAND_MAPPINGS = {
    "Darwin": {
        "list_files": "ls -la",
        "find_files": "find",
        "copy": "cp",
        "move": "mv",
        "delete": "rm",
        "open_file": "open",
    },
    "Linux": {
        "list_files": "ls -la",
        "find_files": "find",
        "copy": "cp",
        "move": "mv",
        "delete": "rm",
        "open_file": "xdg-open",
    },
    "Windows": {
        "list_files": "dir",
        "find_files": "where",
        "copy": "copy",
        "move": "move",
        "delete": "del",
        "open_file": "start",
    },
}

_PLATFORM_EXAMPLES = {
    "Darwin": "e.g. list files with 'ls -la', open with 'open README.md'",
    "Linux": "e.g. list files with 'ls -la', open with 'xdg-open README.md'",
    "Windows": "e.g. list files with 'dir', open with 'start README.md'",
}

_TOOL_CANDIDATES = (
    "gh",
    "git",
    "docker",
    "kubectl",
    "npm",
    "pip",
    "python",
    "node",
    "make",
    "cmake",
    "curl",
    "jq",
    "psql",
    "sqlite3",
    "mongosh",
    "grep",
    "rg",
    "sed",
    "awk",
    "brew",
    "bundle",
    "wget",
)


def build_system_context() -> dict[str, Any]:
    """Return runtime environment details useful for prompt construction."""

    system = platform.system() or "unknown"
    os_friendly = _OS_FRIENDLY_NAMES.get(system, system or "Unknown")

    if system == "Darwin":
        os_version = platform.mac_ver()[0] or platform.release()
    elif system == "Windows":
        os_version = platform.version()
    else:
        os_version = platform.release()

    architecture = platform.machine() or platform.processor() or "unknown"

    shell = os.environ.get("SHELL")
    if not shell:
        shell = os.environ.get("COMSPEC", "cmd.exe") if system == "Windows" else "/bin/sh"

    cwd = str(Path.cwd())
    home_dir = str(Path.home())
    python_version = platform.python_version()

    is_unix = system in {"Darwin", "Linux"}
    shell_type = "unix" if is_unix else "windows"
    path_separator = "/" if is_unix else "\\"
    command_separator = "&&" if is_unix else "&"
    null_device = "/dev/null" if is_unix else "NUL"
    temp_dir = tempfile.gettempdir()

    available_tools = [tool for tool in _TOOL_CANDIDATES if shutil.which(tool)]
    if not available_tools:
        available_tools = []

    command_mappings = _COMMAND_MAPPINGS.get(system, _COMMAND_MAPPINGS.get("Linux"))
    platform_examples = _PLATFORM_EXAMPLES.get(system, _PLATFORM_EXAMPLES.get("Linux"))

    return {
        "os": system,
        "os_friendly": os_friendly,
        "os_version": os_version,
        "shell": shell,
        "shell_type": shell_type,
        "architecture": architecture,
        "home_dir": home_dir,
        "cwd": cwd,
        "python_version": python_version,
        "available_tools": available_tools,
        "is_unix": is_unix,
        "path_separator": path_separator,
        "command_separator": command_separator,
        "null_device": null_device,
        "temp_dir": temp_dir,
        "command_mappings": command_mappings,
        "platform_examples": platform_examples,
    }


def _resolve_repo_path(path_value: str | None) -> Path:
    repo_root = Path.cwd().resolve()
    if not path_value or path_value.strip() in {"", "."}:
        return repo_root
    candidate = (repo_root / path_value).resolve()
    if repo_root not in candidate.parents and candidate != repo_root:
        raise click.ClickException(f"Path '{path_value}' escapes the repository root.")
    return candidate


def _collect_project_structure_outline(
    repo_root: Path,
    *,
    max_entries: int = 160,
    max_depth: int = 3,
) -> str | None:
    """Return a concise repository outline for prompt context."""

    outline = generate_repo_outline(repo_root, max_entries=max_entries, max_depth=max_depth)
    if not outline:
        return None
    return outline


def _get_structure_hints_state(ctx: click.Context) -> dict[str, Any]:
    state = ctx.obj.setdefault(
        "_structure_hints_state",
        {"symbols": set(), "files": {}, "project_summary": None},
    )

    symbols = state.get("symbols")
    if not isinstance(symbols, set):
        state["symbols"] = set()
    files = state.get("files")
    if not isinstance(files, dict):
        state["files"] = {}
    return state


def _export_structure_hints_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "symbols": sorted(state.get("symbols", set())),
        "files": dict(state.get("files", {})),
        "project_summary": state.get("project_summary"),
    }


def _merge_structure_hints_state(state: dict[str, Any], payload: dict[str, Any] | None) -> None:
    if not payload:
        return
    symbols = payload.get("symbols")
    if isinstance(symbols, Iterable):
        state.setdefault("symbols", set()).update(str(item) for item in symbols)
    files = payload.get("files") or {}
    if isinstance(files, Mapping):
        for path, info in files.items():
            if not isinstance(info, Mapping):
                continue
            existing = state.setdefault("files", {}).setdefault(path, {})
            outline = info.get("outline")
            if outline and "symbols" not in existing:
                symbols = set(extract_symbols_from_outline(outline))
                existing["symbols"] = sorted(symbols)
            if info.get("symbols") and "symbols" not in existing:
                symbols = set(info.get("symbols"))
                existing["symbols"] = sorted(symbols)
            if info.get("summary"):
                existing.setdefault("summaries", []).append(info["summary"])
    if payload.get("project_summary"):
        state["project_summary"] = payload["project_summary"]


def _update_files_discovered(files_discovered: set[str], payload: dict[str, Any] | None) -> None:
    if not payload:
        return
    files_value = payload.get("files", [])
    if isinstance(files_value, Mapping):
        iterable_files = files_value.values()
    elif isinstance(files_value, Iterable) and not isinstance(files_value, (str, bytes)):
        iterable_files = files_value
    else:
        iterable_files = ()

    for file_entry in iterable_files:
        path = file_entry.get("path") if isinstance(file_entry, Mapping) else file_entry
        if path:
            files_discovered.add(str(path))

    matches_value = payload.get("matches", [])
    if isinstance(matches_value, Iterable) and not isinstance(matches_value, (str, bytes)):
        matches_iterable = matches_value
    else:
        matches_iterable = ()

    for match in matches_iterable:
        path = match.get("path") if isinstance(match, Mapping) else match
        if path:
            files_discovered.add(str(path))

    summaries_value = payload.get("summaries", [])
    if isinstance(summaries_value, Iterable) and not isinstance(summaries_value, (str, bytes)):
        summaries_iterable = summaries_value
    else:
        summaries_iterable = ()

    for summary in summaries_iterable:
        path = summary.get("path") if isinstance(summary, Mapping) else summary
        if path:
            files_discovered.add(str(path))


_LANGUAGE_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hh": "cpp",
    ".hpp": "cpp",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "csharp",
    ".kt": "kotlin",
    ".swift": "swift",
    ".m": "objective-c",
    ".mm": "objective-c++",
    ".sh": "shell",
    ".bash": "shell",
}


def _detect_repository_language(
    repo_root: Path,
    *,
    max_files: int = 400,
) -> tuple[str | None, int | None]:
    counts: dict[str, int] = {}
    total = 0
    try:
        for path in repo_root.rglob("*"):
            if total >= max_files:
                break
            if not path.is_file():
                continue
            total += 1
            ext = path.suffix.lower()
            language = _LANGUAGE_EXTENSIONS.get(ext)
            if language:
                counts[language] = counts.get(language, 0) + 1
    except OSError:
        pass

    language = max(counts, key=counts.get) if counts else None
    return language, total if total else None


def infer_task_files(task: Mapping[str, Any], repo_root: Path) -> list[str]:
    """Infer target files for a task using commands, deliverables, and textual hints."""
    repo_root = repo_root.resolve()
    results: list[str] = []
    seen: set[str] = set()

    def _add(candidate: Path) -> None:
        try:
            resolved = candidate.resolve()
        except OSError:
            return
        if not resolved.is_file():
            return
        try:
            relative = resolved.relative_to(repo_root)
        except ValueError:
            return
        rel_str = str(relative)
        if rel_str not in seen:
            seen.add(rel_str)
            results.append(rel_str)

    for command in task.get("commands") or []:
        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = command.split()
        for idx, token in enumerate(tokens):
            if token == "--files":
                for follow in tokens[idx + 1 :]:
                    if follow.startswith("-"):
                        break
                    _add(repo_root / follow)
            elif "/" in token:
                _add(repo_root / token.strip("\"'"))

    for deliverable in task.get("deliverables") or []:
        if isinstance(deliverable, str):
            _add(repo_root / deliverable)

    text_blob = " ".join(filter(None, [task.get("title"), task.get("description")]))
    if text_blob:
        for match in re.findall(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+", text_blob):
            cleaned = match.strip(".,;:'\"()[]{}<>")
            _add(repo_root / cleaned)

        keywords = {kw.lower() for kw in extract_keywords(text_blob)}
    else:
        keywords = set()

    if keywords:
        examined = 0
        for path_candidate in repo_root.rglob("*"):
            if examined >= 200:
                break
            examined += 1
            if not path_candidate.is_file():
                continue
            stem = path_candidate.stem.lower()
            parent_names = {
                parent.name.lower() for parent in path_candidate.parents if parent != repo_root
            }
            if stem in keywords or keywords.intersection(parent_names):
                _add(path_candidate)

    return results


def update_task_state(
    store: StateStore,
    plan: dict[str, Any],
    task: dict[str, Any],
    updates: Mapping[str, Any],
    *,
    reasoning: Any | None = None,
) -> None:
    """Persist updated plan and task state to the store."""
    updated_task = dict(task)
    updated_task.update(updates)

    if reasoning is not None:
        apply = getattr(reasoning, "apply_to_task", None)
        if callable(apply):
            apply(updated_task)
        merge = getattr(reasoning, "merge_into_plan", None)
        if callable(merge):
            merge(plan)

    task_id = updated_task.get("id")
    tasks = plan.setdefault("tasks", [])
    if task_id and isinstance(tasks, list):
        for existing in tasks:
            if isinstance(existing, dict) and existing.get("id") == task_id:
                existing.clear()
                existing.update(updated_task)
                break
        else:
            tasks.append(updated_task)
    task.clear()
    task.update(updated_task)

    if "status" in updates:
        plan["status"] = updates["status"]

    snapshot = store.load() or {}
    snapshot["last_plan"] = plan
    snapshot["last_updated"] = datetime.utcnow().isoformat()
    store.save(snapshot)


def _create_sandbox(settings: Settings) -> None:
    # Sandbox execution removed - using direct execution only
    # For safety, we rely on user permissions and system security
    return None


def _make_tool_context(ctx: click.Context, *, with_sandbox: bool = False) -> ToolContext:
    settings: Settings = ctx.obj["settings"]

    sandbox = None
    if with_sandbox:
        sandbox = ctx.obj.get("sandbox_executor")
        if sandbox is None:
            sandbox = _create_sandbox(settings)
            ctx.obj["sandbox_executor"] = sandbox

    devagent_cfg = ctx.obj.get("devagent_config")
    if devagent_cfg is None:
        devagent_cfg = load_devagent_yaml()
        ctx.obj["devagent_config"] = devagent_cfg

    structure_hints_state = _get_structure_hints_state(ctx)

    extra: dict[str, Any] = {
        "structure_hints": _export_structure_hints_state(structure_hints_state)
    }

    shell_manager = ctx.obj.get("_shell_session_manager") if isinstance(ctx.obj, dict) else None
    shell_session_id = ctx.obj.get("_shell_session_id") if isinstance(ctx.obj, dict) else None
    if shell_manager and shell_session_id:
        extra["shell_session_manager"] = shell_manager
        extra["shell_session_id"] = shell_session_id

    return ToolContext(
        repo_root=Path.cwd(),
        settings=settings,
        sandbox=sandbox,
        devagent_config=devagent_cfg,
        metrics_collector=None,
        extra=extra,
    )


def _invoke_registry_tool(
    ctx: click.Context,
    tool_name: str,
    payload: dict[str, Any] | None = None,
    *,
    with_sandbox: bool = False,
) -> dict[str, Any]:
    """Invoke a registry tool with a lazily created ToolContext."""

    context = _make_tool_context(ctx, with_sandbox=with_sandbox)
    return tool_registry.invoke(tool_name, payload or {}, context)


def _normalize_argument_list(
    arguments: dict[str, Any],
    *,
    plural_key: str,
    singular_key: str | None = None,
) -> list[str]:
    """Return a normalized list of string arguments for registry handlers."""

    values = arguments.get(plural_key)
    if not values and singular_key and arguments.get(singular_key):
        values = [arguments[singular_key]]
    if values is None:
        return []
    if isinstance(values, (str, os.PathLike)):
        values = [values]
    return [str(item) for item in values]


def _build_context(settings: Settings) -> dict[str, Any]:
    state_store = StateStore(settings.state_file)
    policy = ApprovalPolicy(
        auto_approve_plan=settings.auto_approve_plan,
        auto_approve_code=settings.auto_approve_code,
        auto_approve_shell=settings.auto_approve_shell,
        auto_approve_adr=settings.auto_approve_adr,
        emergency_override=settings.emergency_override,
        audit_file=settings.audit_approvals,
    )
    return {
        "settings": settings,
        "state": state_store,
        "llm_client": None,
        "approval_policy": policy,
        "sandbox_executor": None,
        "devagent_config": None,
    }


def _prompt_yes_no(
    ctx: click.Context,
    purpose: str,
    message: str,
    *,
    default: bool = False,
) -> bool:
    """Consistently request user confirmation using the approval manager."""

    policy: ApprovalPolicy = ctx.obj["approval_policy"]
    audit_file = Path.cwd() / ".devagent" / "audit.log" if policy.audit_file else None
    approvals = ApprovalManager(policy, audit_file)
    return approvals.require(purpose, default=default, prompt=message)


def _record_invocation(ctx: click.Context, overrides: dict[str, Any] | None = None) -> None:
    """Persist command invocation details for history and replay."""
    if not ctx or not ctx.command_path:
        return
    set_correlation_id(uuid.uuid4().hex[:12])
    state_obj = ctx.obj.get("state") if ctx.obj else None
    if not isinstance(state_obj, InMemoryStateStore):
        return
    root_ctx = ctx.find_root()
    root_name = root_ctx.info_name if root_ctx else None
    parts = ctx.command_path.split()
    if root_name and parts and parts[0] == root_name:
        parts = parts[1:]
    if not parts or parts[0] == "history":
        return
    params = dict(ctx.params)
    if overrides:
        params.update(overrides)
    entry = {
        "command_path": parts,
        "params": params,
        "timestamp": datetime.utcnow().isoformat(),
    }
    state_obj.append_history(entry, limit=MAX_HISTORY_ENTRIES)


def get_llm_client(ctx: click.Context):
    client = ctx.obj.get("llm_client")
    if client:
        return client
    settings: Settings = ctx.obj["settings"]
    if not settings.api_key:
        raise click.ClickException(
            "No API key configured. Set DEVAGENT_API_KEY or update config file."
        )

    retry_config = RetryConfig(
        max_retries=3,
        initial_delay=0.5,
        max_delay=5.0,
        backoff_multiplier=2.0,
        jitter_ratio=0.3,
    )

    provider_key = settings.provider.lower()
    base_url_override: str | None = settings.base_url or None
    provider_kwargs: dict[str, Any] = {}

    if provider_key in {"openrouter", "cerebras"}:
        if settings.provider_only:
            provider_kwargs["provider_only"] = tuple(settings.provider_only)
        if settings.provider_config:
            provider_kwargs["provider_config"] = dict(settings.provider_config)
        if settings.request_headers:
            provider_kwargs["default_headers"] = dict(settings.request_headers)
        if not base_url_override or base_url_override == DEEPSEEK_DEFAULT_BASE_URL:
            base_url_override = None

    client = create_client(
        provider=settings.provider,
        api_key=settings.api_key,
        model=settings.model,
        base_url=base_url_override,
        **provider_kwargs,
    )

    if hasattr(client, "configure_timeout"):
        client.configure_timeout(120.0)
    if hasattr(client, "configure_retry"):
        client.configure_retry(retry_config)

    budget_config = config_from_settings(settings)
    disabled = bool(getattr(settings, "disable_context_pruner", False))
    wrapped_client = BudgetedLLMClient(client, budget_config, disabled=disabled)
    ctx.obj["_raw_llm_client"] = client
    ctx.obj["llm_client"] = wrapped_client

    session_manager = SessionManager.get_instance()
    pruning_config = _build_context_pruning_config_from_settings(settings)

    # Try to use enhanced summarizer if available
    summarizer = None
    try:
        from ai_dev_agent.session.enhanced_summarizer_wrapper import create_enhanced_summarizer

        summarizer = create_enhanced_summarizer(wrapped_client)
    except ImportError:
        pass

    # Fall back to standard LLM summarizer if enhanced not available
    if summarizer is None:
        summarizer = LLMConversationSummarizer(wrapped_client)

    session_manager.configure_context_service(config=pruning_config, summarizer=summarizer)
    return wrapped_client


__all__ = [
    "_build_context",
    "_collect_project_structure_outline",
    "_create_sandbox",
    "_detect_repository_language",
    "_export_structure_hints_state",
    "_get_structure_hints_state",
    "_invoke_registry_tool",
    "_make_tool_context",
    "_merge_structure_hints_state",
    "_normalize_argument_list",
    "_prompt_yes_no",
    "_record_invocation",
    "_resolve_repo_path",
    "_update_files_discovered",
    "build_system_context",
    "get_llm_client",
    "infer_task_files",
    "update_task_state",
]


def _build_context_pruning_config_from_settings(settings: Settings) -> ContextPruningConfig:
    """Translate CLI settings into a context pruning configuration."""

    max_total = max(int(settings.context_pruner_max_total_tokens or 0), 1)

    trigger_tokens = settings.context_pruner_trigger_tokens
    if trigger_tokens is None:
        ratio = settings.context_pruner_trigger_ratio
        if ratio <= 0:
            ratio = 0.8
        # Cap within (0, 1] to keep trigger sensible relative to the max budget.
        ratio = max(min(ratio, 1.0), 0.05)
        trigger_tokens = int(max_total * ratio)

    trigger_tokens = max(min(int(trigger_tokens), max_total), 1)
    keep_recent = max(int(settings.context_pruner_keep_recent_messages or 0), 2)
    summary_max_chars = max(int(settings.context_pruner_summary_max_chars or 0), 1)
    max_events = max(int(settings.context_pruner_max_event_history or 0), 1)

    # Get max_tool_messages from settings (defaults to 10 if not set)
    max_tool_msgs = max(int(getattr(settings, "max_tool_messages_kept", 10)), 3)

    return ContextPruningConfig(
        max_total_tokens=max_total,
        trigger_tokens=trigger_tokens,
        keep_recent_messages=keep_recent,
        summary_max_chars=summary_max_chars,
        max_event_history=max_events,
        max_tool_messages=max_tool_msgs,  # Now configurable!
    )
