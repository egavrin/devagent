"""Helpers for persisting large tool outputs as artifacts."""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

ARTIFACTS_ROOT = Path(".devagent") / "artifacts"


def _sanitize_suffix(suffix: str | None) -> str:
    """Return a filesystem-safe suffix while preserving extensions."""
    if not suffix:
        return ".txt"

    candidate = str(suffix).strip()
    if not candidate:
        return ".txt"

    name = Path(candidate).name  # Drop any directory traversal attempts
    suffixes = Path(name).suffixes
    if suffixes:
        safe = "".join(suffixes)
    else:
        safe = name if name.startswith(".") else f".{name.lstrip('.')}"

    safe = safe.replace("/", "_").replace("\\", "_")
    if safe in {"", ".", "_"}:
        return ".txt"
    return safe


def write_artifact(content: str, *, suffix: str = ".txt", root: Path | None = None) -> Path:
    """Persist content to an artifact file and return the path.

    Files are stored under `.devagent/artifacts` by default using a timestamp and
    hash-based naming scheme to avoid collisions."""

    base_dir = (root or Path.cwd()) / ARTIFACTS_ROOT
    base_dir.mkdir(parents=True, exist_ok=True)

    data = content.encode("utf-8", errors="replace")
    digest = hashlib.sha1(data).hexdigest()[:12]
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_suffix = _sanitize_suffix(suffix)
    filename = f"artifact_{timestamp}_{digest}{safe_suffix}"

    path = base_dir / filename
    resolved_base = base_dir.resolve()
    resolved_path = path.resolve()
    if resolved_path.parent != resolved_base:
        try:
            resolved_path.relative_to(resolved_base)
        except ValueError as exc:
            raise ValueError("Unsafe artifact path resolved outside artifact directory.") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


__all__ = ["ARTIFACTS_ROOT", "write_artifact"]
