"""Reader for devagent.yaml (CI/build/test/gates configuration).

This intentionally supports a minimal subset of the example schema described by
the user: build/test/lint/type/format/coverage commands and gate thresholds.
If the YAML file is missing, callers should fall back to Settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore


@dataclass
class DevAgentConfig:
    build_cmd: str | None = None
    test_cmd: str | None = None
    lint_cmd: str | None = None
    type_cmd: str | None = None
    format_cmd: str | None = None
    coverage_cmd: str | None = None
    threshold_diff: float | None = None
    threshold_project: float | None = None
    diff_limit_lines: int | None = None
    diff_limit_files: int | None = None
    sandbox_allowlist: tuple[str, ...] = ()
    sandbox_time_limit: int | None = None
    sandbox_memory_limit: int | None = None
    ctags_cmd: Any | None = None
    ctags_db: str | None = None
    ctags_refresh_sec: int | None = None
    react_iteration_global_cap: int | None = None
    react_phase_overrides: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    budget_control: dict[str, Any] = field(default_factory=dict)


def load_devagent_yaml(path: Path | None = None) -> DevAgentConfig | None:
    """Load devagent.yaml into a DevAgentConfig or return None if unavailable."""
    candidate = path or (Path.cwd() / "devagent.yaml")
    if not candidate.is_file():
        return None
    if yaml is None:
        return None
    try:
        data = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
    except Exception:  # pragma: no cover - best effort
        return None
    if not isinstance(data, dict):
        return None

    cfg = DevAgentConfig()
    build = data.get("build") or {}
    tests = data.get("tests") or {}
    coverage = data.get("coverage") or {}
    lint = data.get("lint") or {}
    types = data.get("types") or {}
    fmt = data.get("format") or {}
    gates = data.get("gates") or []
    sandbox = data.get("sandbox") or {}
    index = data.get("index") or {}
    ctags = index.get("ctags") or {}
    react = data.get("react") or {}
    react_iteration = react.get("iteration") or {}
    budget_control_cfg = data.get("budget_control") or {}

    def g(name: str, d: dict[str, Any]) -> str | None:
        v = d.get("cmd")
        return str(v) if v else None

    cfg.build_cmd = g("build", build)
    cfg.test_cmd = g("tests", tests)
    cfg.lint_cmd = g("lint", lint)
    cfg.type_cmd = g("types", types)
    cfg.format_cmd = g("format", fmt)
    cfg.coverage_cmd = g("coverage", coverage)

    # Thresholds
    if isinstance(coverage.get("threshold"), dict):
        cfg.threshold_diff = _as_float(coverage["threshold"].get("diff"))
        cfg.threshold_project = _as_float(coverage["threshold"].get("project"))

    # Gates diff limits (if present)
    for gate in gates if isinstance(gates, list) else []:
        if isinstance(gate, dict) and gate.get("name") == "diff.size":
            cfg.diff_limit_lines = _as_int(gate.get("lte_lines"))
            cfg.diff_limit_files = _as_int(gate.get("lte_files"))
            break

    # Sandbox configuration
    allowlist = sandbox.get("shell_allowlist") or []
    if isinstance(allowlist, list):
        cfg.sandbox_allowlist = tuple(str(item) for item in allowlist)
    cfg.sandbox_time_limit = _as_int(sandbox.get("time_limit_sec"))
    cfg.sandbox_memory_limit = _as_int(sandbox.get("memory_limit_mb"))

    # Index / ctags configuration
    if ctags:
        cfg.ctags_cmd = ctags.get("cmd")
        db_value = ctags.get("db")
        if db_value:
            cfg.ctags_db = str(db_value)
        refresh_value = ctags.get("refresh_sec")
        if refresh_value is not None:
            cfg.ctags_refresh_sec = _as_int(refresh_value)

    cfg.react_iteration_global_cap = _as_int(react_iteration.get("global_cap"))

    phases_cfg = react_iteration.get("phases")
    if isinstance(phases_cfg, dict):
        parsed_overrides: dict[str, list[dict[str, Any]]] = {}
        for task_name, overrides in phases_cfg.items():
            if not isinstance(overrides, list):
                continue
            normalized: list[dict[str, Any]] = []
            for item in overrides:
                if not isinstance(item, dict) or not item.get("name"):
                    continue
                entry: dict[str, Any] = {"name": str(item["name"])}
                for key in ("description", "max_iterations", "weight", "min_iterations"):
                    if key in item:
                        entry[key] = item[key]
                normalized.append(entry)
            if normalized:
                parsed_overrides[str(task_name)] = normalized
        if parsed_overrides:
            cfg.react_phase_overrides = parsed_overrides

    if isinstance(budget_control_cfg, dict):
        cfg.budget_control = budget_control_cfg

    return cfg


def _as_int(v: Any) -> int | None:
    try:
        return int(v) if v is not None else None
    except Exception:
        return None


def _as_float(v: Any) -> float | None:
    try:
        return float(v) if v is not None else None
    except Exception:
        return None


__all__ = ["DevAgentConfig", "load_devagent_yaml"]
