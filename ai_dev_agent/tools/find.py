"""Simple file finding tool using ripgrep."""
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .registry import ToolSpec, ToolContext, registry

SCHEMA_DIR = Path(__file__).resolve().parent / "schemas" / "tools"


def find(payload: Mapping[str, Any], context: ToolContext) -> Mapping[str, Any]:
    """Find files by pattern using ripgrep."""
    # Normalize: accept both 'query' and 'pattern'
    query = payload.get("query") or payload.get("pattern", "")
    path = payload.get("path", ".")
    limit = min(payload.get("limit", 20), 100)  # Reduced default to 20, cap at 100

    # Convert path to absolute if relative
    if not os.path.isabs(path):
        path = os.path.join(context.repo_root, path)

    # Use ripgrep to list files matching the pattern
    cmd = ["rg", "--files", "--hidden", "--no-ignore-vcs", "-g", query, path]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(context.repo_root),
            timeout=10
        )

        if result.returncode not in {0, 1}:
            # Error occurred
            return {"error": result.stderr.strip() if result.stderr else "Unknown error", "files": []}

        files: List[str] = []
        if result.stdout:
            all_files = result.stdout.strip().split('\n')
            # Make paths relative to repo root
            for f in all_files:
                if f:
                    try:
                        abs_path = (context.repo_root / f).resolve()
                        rel_path = abs_path.relative_to(context.repo_root)
                        files.append(str(rel_path))
                    except ValueError:
                        # If can't make relative, use absolute
                        files.append(f)

        # Sort by modification time (newest first)
        files_with_time: List[tuple[str, float]] = []
        for f in files:
            full_path = context.repo_root / f
            if full_path.exists():
                mtime = full_path.stat().st_mtime
                files_with_time.append((f, mtime))

        # Sort by mtime descending
        files_with_time.sort(key=lambda x: x[1], reverse=True)

        # Track if we're truncating results
        total_files = len(files_with_time)
        truncated = total_files > limit

        results: List[Dict[str, Any]] = []
        now = datetime.now(timezone.utc).timestamp()
        for rel_path, mtime in files_with_time[:limit]:
            full_path = (context.repo_root / rel_path).resolve()
            try:
                size_bytes = full_path.stat().st_size
            except OSError:
                size_bytes = 0
            try:
                lines = sum(1 for _ in full_path.open(encoding="utf-8", errors="ignore"))
            except OSError:
                lines = 0

            age_days = max((now - mtime) / 86400.0, 0.0)
            score = 10.0 / (1.0 + age_days)
            depth = rel_path.count(os.sep)
            if depth <= 2:
                score *= 1.5

            results.append(
                {
                    "path": rel_path,
                    "score": round(score, 2),
                    "lines": lines,
                    "size_bytes": size_bytes,
                    "mtime": datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat(),
                }
            )

        result_dict = {"files": results}

        # Add truncation info if results were limited
        if truncated:
            result_dict["truncated"] = True
            result_dict["total_files"] = total_files
            result_dict["message"] = f"Showing {limit} of {total_files} matching files. Use more specific pattern to narrow results."

        return result_dict

    except subprocess.TimeoutExpired:
        return {"error": "Search timeout", "files": []}
    except Exception as e:
        return {"error": str(e), "files": []}


# Register the tool
registry.register(
    ToolSpec(
        name="find",
        handler=find,
        request_schema_path=SCHEMA_DIR / "find.request.json",
        response_schema_path=SCHEMA_DIR / "find.response.json",
        description=(
            "Find files by pattern using ripgrep glob syntax.\n"
            "Examples:\n"
            "  find('*.py') - all Python files\n"
            "  find('**/test_*.js') - test files in any directory\n"
            "  find('src/**/*.ts') - TypeScript files under src\n"
            "Results are sorted by modification time (newest first)."
        ),
        category="search",
    )
)
