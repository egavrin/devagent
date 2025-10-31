"""Configuration loading utilities for the dev agent."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - fallback for <3.11
    import tomli as tomllib


CONFIG_FILENAMES: tuple[str, ...] = (".devagent.toml", "devagent.toml")
DEFAULT_CONFIG_PATHS = (
    Path.home() / ".config" / "devagent" / "config.toml",
    Path.home() / ".devagent.toml",
    Path.home() / "devagent.toml",
)


def find_config_in_parents(
    start_path: Path, config_name: str | Sequence[str] = ".devagent.toml"
) -> Path | None:
    """Search parent directories starting from ``start_path`` for configuration files."""

    if isinstance(config_name, str):
        candidate_names: tuple[str, ...] = (config_name,)
    else:
        candidate_names = tuple(config_name)

    current = start_path.resolve()
    if current.is_file():
        current = current.parent

    while True:
        for name in candidate_names:
            candidate = current / name
            if candidate.is_file():
                return candidate.resolve()
        if current.parent == current:
            break
        current = current.parent
    return None


@dataclass
class Settings:
    """Runtime configuration for the CLI agent."""

    provider: str = "deepseek"
    model: str = "deepseek-coder"
    api_key: str | None = None
    base_url: str = "https://api.deepseek.com/v1"
    provider_only: Sequence[str] = ()
    provider_config: dict[str, Any] = field(default_factory=dict)
    request_headers: dict[str, str] = field(default_factory=dict)
    workspace_root: Path = Path()
    auto_approve_plan: bool = False
    auto_approve_code: bool = False
    auto_approve_shell: bool = False
    auto_approve_adr: bool = False
    emergency_override: bool = False
    audit_approvals: bool = False
    state_file: Path = Path(".devagent/state.json")
    log_level: str = "INFO"
    structured_logging: bool = False
    repomap_debug_stdout: bool = False  # Enable RepoMap debug logging
    diff_limit_lines: int = 400
    diff_limit_files: int = 10
    patch_coverage_target: float = 0.8
    stuck_threshold: int = 3
    steps_budget: int = 25
    max_iterations: int = 120
    lint_command: str | None = None
    format_command: str | None = None
    typecheck_command: str | None = None
    compile_command: str | None = None
    test_command: str = "pytest"
    coverage_xml_path: Path = Path("coverage.xml")
    sandbox_allowlist: tuple[str, ...] = ()
    sandbox_memory_limit_mb: int = 2048
    sandbox_cpu_time_limit: int = 120
    shell_executable: str | None = None
    shell_session_timeout: float | None = None
    shell_session_cpu_time_limit: int | None = None
    shell_session_memory_limit_mb: int | None = None
    # Extras / Phase 4
    flake_check_runs: int = 0
    perf_command: str | None = None
    max_context_tokens: int = 100_000
    response_headroom_tokens: int = 2_000
    max_tool_output_chars: int = 4_000
    max_tool_messages_kept: int = 10
    keep_last_assistant_messages: int = 4
    fs_read_default_max_lines: int = 200
    search_max_results: int = 100
    disable_context_pruner: bool = False
    always_use_planning: bool = False
    # Cost tracking settings
    enable_cost_tracking: bool = True
    cost_warning_threshold: float = 1.0  # USD
    # Cache management settings
    enable_cache_management: bool = True
    cache_ttl_seconds: int = 300  # 5 minutes
    # Reflection settings
    enable_reflection: bool = True
    max_reflections: int = 3
    # Summarization settings
    enable_summarization: bool = True
    summarization_model: str | None = None  # Use cheaper model if specified
    # Adaptive budget settings
    adaptive_budget_scaling: bool = True
    enable_two_tier_pruning: bool = True
    # Retry settings
    enable_retry: bool = True
    retry_max_attempts: int = 5
    retry_initial_delay: float = 0.125
    retry_max_delay: float = 60.0
    context_pruner_max_total_tokens: int = 12_000
    context_pruner_trigger_tokens: int | None = None
    context_pruner_trigger_ratio: float = 0.8
    context_pruner_keep_recent_messages: int = 10
    context_pruner_summary_max_chars: int = 1_200
    context_pruner_max_event_history: int = 10

    # Agent context synthesis settings (unified mode is now always enabled)
    agent_context_synthesis: bool = True  # Synthesize context from previous steps
    agent_max_context_chars: int = 2000  # Max chars for context synthesis

    # Memory Bank settings (ReasoningBank pattern)
    enable_memory_bank: bool = True  # Enable memory system
    memory_retrieval_limit: int = 5  # Max memories to retrieve
    memory_similarity_threshold: float = 0.3  # Minimum similarity for retrieval
    memory_prune_threshold: float = 0.2  # Prune memories below this effectiveness
    memory_max_per_type: int = 100  # Max memories per task type

    # Playbook and dynamic instructions settings removed - features deleted
    # Review settings
    review_max_files_per_chunk: int = 50  # Increased from 10 for fewer LLM calls
    review_max_hunks_per_chunk: int = 50  # Increased from 8 to handle larger files
    review_max_lines_per_chunk: int = 1500  # Increased from 320 to use model capacity
    review_token_budget: int = 80_000  # Token budget for chunk sizing (increased to reduce splits)
    review_chunk_overlap_lines: int = 100  # Lines to overlap between chunks (quality improvement)
    review_context_pad_lines: int = 40  # Lines of context before/after hunks (quality improvement)
    review_context_max_lines: int = 600  # Max lines per context item (quality improvement)
    review_context_max_total_lines: int = 1500  # Total context budget (quality improvement)

    # Global message settings (--system and --context CLI options)
    global_system_message: str | None = None  # System message (inline or file path)
    global_context_message: str | None = None  # Context message (inline or file path)

    def ensure_state_dir(self) -> None:
        """Ensure the directory for the state file exists."""
        if not self.state_file.parent.exists():
            self.state_file.parent.mkdir(parents=True, exist_ok=True)


DEFAULT_MAX_ITERATIONS: int = Settings.__dataclass_fields__["max_iterations"].default  # type: ignore[assignment]


def _cast_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "on", "yes", "y"}
    return bool(value)


def _load_from_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    return {k.replace("-", "_"): v for k, v in data.items()}


def _load_from_env(prefix: str = "DEVAGENT_") -> dict[str, Any]:
    env: dict[str, Any] = {}
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        field = key[len(prefix) :].lower()
        if field in {
            "auto_approve_plan",
            "auto_approve_code",
            "auto_approve_shell",
            "auto_approve_adr",
            "emergency_override",
            "audit_approvals",
            "structured_logging",
            "disable_context_pruner",
            "always_use_planning",
            "agent_context_synthesis",
            "enable_memory_bank",
        }:
            env[field] = _cast_bool(value)
        elif field in {
            "flake_check_runs",
        } or field in {
            "diff_limit_lines",
            "diff_limit_files",
            "stuck_threshold",
            "steps_budget",
            "max_iterations",
            "sandbox_memory_limit_mb",
            "sandbox_cpu_time_limit",
            "max_context_tokens",
            "response_headroom_tokens",
            "max_tool_output_chars",
            "max_tool_messages_kept",
            "keep_last_assistant_messages",
            "fs_read_default_max_lines",
            "search_max_results",
            "context_pruner_max_total_tokens",
            "context_pruner_trigger_tokens",
            "context_pruner_keep_recent_messages",
            "context_pruner_summary_max_chars",
            "context_pruner_max_event_history",
            "agent_max_context_chars",
            "memory_retrieval_limit",
            "memory_max_per_type",
        }:
            env[field] = int(value)
        elif (
            field
            in {
                "patch_coverage_target",
                "context_pruner_trigger_ratio",
                "memory_similarity_threshold",
                "memory_prune_threshold",
            }
            or field == "shell_session_timeout"
        ):
            env[field] = float(value)
        elif field == "state_file" or field == "coverage_xml_path":
            env[field] = Path(value)
        elif field == "sandbox_allowlist" or field == "provider_only":
            env[field] = tuple(filter(None, (item.strip() for item in value.split(","))))
        elif field in {"provider_config", "request_headers"}:
            try:
                env[field] = json.loads(value)
            except json.JSONDecodeError:
                env[field] = {}
        elif field in {"shell_session_cpu_time_limit", "shell_session_memory_limit_mb"}:
            env[field] = int(value)
        elif field == "system":
            env["global_system_message"] = value  # Map DEVAGENT_SYSTEM to global_system_message
        elif field == "context":
            env["global_context_message"] = value  # Map DEVAGENT_CONTEXT to global_context_message
        else:
            env[field] = value
    return env


def load_settings(explicit_path: Path | None = None) -> Settings:
    """Load configuration, merging file and environment sources."""

    file_data: dict[str, Any] = {}
    if explicit_path:
        file_data.update(_load_from_file(explicit_path))
    else:
        search_paths = []
        cwd = Path.cwd()
        project_config = find_config_in_parents(cwd, CONFIG_FILENAMES)
        if project_config:
            search_paths.append(project_config)
        for name in CONFIG_FILENAMES:
            search_paths.append(cwd / name)
        search_paths.extend(DEFAULT_CONFIG_PATHS)
        seen_paths = set()
        for candidate in search_paths:
            if candidate in seen_paths:
                continue
            seen_paths.add(candidate)
            file_data = _load_from_file(candidate)
            if file_data:
                break

    env_data = _load_from_env()
    merged: dict[str, Any] = {**file_data, **env_data}

    # Normalize state_file path relative to cwd if provided as string
    if "workspace_root" in merged and isinstance(merged["workspace_root"], str):
        merged["workspace_root"] = Path(merged["workspace_root"])
    if "state_file" in merged and isinstance(merged["state_file"], str):
        merged["state_file"] = Path(merged["state_file"])
    if "coverage_xml_path" in merged and isinstance(merged["coverage_xml_path"], str):
        merged["coverage_xml_path"] = Path(merged["coverage_xml_path"])
    allowlist = merged.get("sandbox_allowlist")
    if allowlist is not None and not isinstance(allowlist, tuple):
        if isinstance(allowlist, str):
            merged["sandbox_allowlist"] = tuple(
                filter(None, (item.strip() for item in allowlist.split(",")))
            )
        else:
            merged["sandbox_allowlist"] = tuple(allowlist)

    provider_only = merged.get("provider_only")
    if provider_only is not None and not isinstance(provider_only, tuple):
        if isinstance(provider_only, str):
            merged["provider_only"] = tuple(
                filter(None, (item.strip() for item in provider_only.split(",")))
            )
        else:
            merged["provider_only"] = tuple(provider_only)

    for key in ("provider_config", "request_headers"):
        value = merged.get(key)
        if value is None:
            continue
        if isinstance(value, dict):
            continue
        if isinstance(value, str):
            try:
                merged[key] = json.loads(value)
            except json.JSONDecodeError:
                merged[key] = {}
            continue
        merged[key] = dict(value)

    # Only pass known fields to the dataclass constructor; attach extras afterward so
    # advanced/toggle flags (e.g., experimental features) remain available as attributes.
    known_fields = set(Settings.__dataclass_fields__)
    init_kwargs = {key: value for key, value in merged.items() if key in known_fields}
    settings = Settings(**init_kwargs)
    for key, value in merged.items():
        if key not in known_fields:
            setattr(settings, key, value)
    if not settings.workspace_root.is_absolute():
        settings.workspace_root = (Path.cwd() / settings.workspace_root).resolve()
    settings.ensure_state_dir()
    return settings


__all__ = ["DEFAULT_MAX_ITERATIONS", "Settings", "load_settings"]
