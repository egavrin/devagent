"""Simple content search tool using ripgrep."""
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .registry import ToolSpec, ToolContext, registry

SCHEMA_DIR = Path(__file__).resolve().parent / "schemas" / "tools"


def grep(payload: Mapping[str, Any], context: ToolContext) -> Mapping[str, Any]:
    """Search content in files using ripgrep."""
    pattern = payload.get("pattern", "")
    path = payload.get("path", ".")
    regex = payload.get("regex", False)
    limit = min(payload.get("limit", 200), 1000)  # Cap at 1000
    max_files = payload.get("max_files", 20)  # Limit number of files returned

    if not pattern:
        return {"matches": []}

    # Convert path to absolute if relative
    if not os.path.isabs(path):
        path = os.path.join(context.repo_root, path)

    # Build ripgrep command
    cmd = ["rg", "-n", "-H", "--no-heading", "--max-count", str(limit)]

    if not regex:
        cmd.append("-F")  # Fixed string search

    cmd.extend([pattern, path])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(context.repo_root),
            timeout=30
        )

        if result.returncode not in {0, 1}:
            # Error occurred
            return {"error": result.stderr.strip() if result.stderr else "Unknown error", "matches": []}

        matches = []
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                if ':' in line and line:
                    # Parse ripgrep output: file:line:content
                    parts = line.split(':', 2)
                    if len(parts) >= 3:
                        file_path = parts[0]
                        try:
                            abs_path = (context.repo_root / file_path).resolve()
                            rel_path = abs_path.relative_to(context.repo_root)
                            file_path = str(rel_path)
                        except ValueError:
                            file_path = str(Path(file_path).resolve())

                        matches.append({
                            "file": file_path,
                            "line": int(parts[1]) if parts[1].isdigit() else 0,
                            "text": parts[2].strip()
                        })

        # Group matches by file for better readability
        files_dict = {}
        for match in matches:
            file_path = match["file"]
            if file_path not in files_dict:
                files_dict[file_path] = []
            files_dict[file_path].append({
                "line": match["line"],
                "text": match["text"]
            })

        # Sort files by modification time
        files_with_time = []
        for file_path in files_dict.keys():
            full_path = context.repo_root / file_path
            if full_path.exists():
                mtime = full_path.stat().st_mtime
                files_with_time.append((file_path, mtime))
            else:
                files_with_time.append((file_path, 0))

        # Sort by mtime descending
        files_with_time.sort(key=lambda x: x[1], reverse=True)

        # Build final result with file limit
        total_files = len(files_with_time)
        truncated = total_files > max_files

        result_matches = []
        for file_path, _ in files_with_time[:max_files]:
            result_matches.append({
                "file": file_path,
                "matches": files_dict[file_path]
            })

        result = {"matches": result_matches}

        # Add truncation info if results were limited
        if truncated:
            result["truncated"] = True
            result["total_files"] = total_files
            result["message"] = f"Showing {max_files} of {total_files} files with matches. Use more specific search pattern to narrow results."

        return result

    except subprocess.TimeoutExpired:
        return {"error": "Search timeout", "matches": []}
    except Exception as e:
        return {"error": str(e), "matches": []}


# Register the tool
registry.register(
    ToolSpec(
        name="grep",
        handler=grep,
        request_schema_path=SCHEMA_DIR / "grep.request.json",
        response_schema_path=SCHEMA_DIR / "grep.response.json",
        description=(
            "Search content in files using ripgrep.\n"
            "Examples:\n"
            "  grep('TODO') - find all TODO comments\n"
            "  grep('func.*name', regex=True) - regex search\n"
            "  grep('error', path='src/') - search only in src directory\n"
            "Results grouped by file, sorted by modification time."
        ),
        category="search",
    )
)
