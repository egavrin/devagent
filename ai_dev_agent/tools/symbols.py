"""Simple symbol finding tool using universal ctags."""
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .registry import ToolSpec, ToolContext, registry

SCHEMA_DIR = Path(__file__).resolve().parent / "schemas" / "tools"


def symbols(payload: Mapping[str, Any], context: ToolContext) -> Mapping[str, Any]:
    """Find symbol definitions using universal ctags."""
    name = payload.get("name", "")
    path = payload.get("path", ".")
    limit = min(payload.get("limit", 100), 500)  # Cap at 500

    if not name:
        return {"symbols": []}

    # Convert path to absolute if relative
    if not os.path.isabs(path):
        path = os.path.join(context.repo_root, path)

    # Check if ctags is available
    ctags_cmd = "ctags"
    try:
        subprocess.run([ctags_cmd, "--version"], capture_output=True, timeout=1)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        # Try universal-ctags
        ctags_cmd = "universal-ctags"
        try:
            subprocess.run([ctags_cmd, "--version"], capture_output=True, timeout=1)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return {"error": "ctags not found. Install universal-ctags.", "symbols": []}

    # Generate or update tags file
    tags_file = os.path.join(context.repo_root, ".tags")

    # Check if we need to regenerate tags (if file is older than 5 minutes)
    should_regenerate = True
    if os.path.exists(tags_file):
        age_seconds = time.time() - os.path.getmtime(tags_file)
        should_regenerate = age_seconds > 300  # 5 minutes

    if should_regenerate:
        try:
            # Generate tags for the specified path
            cmd = [ctags_cmd, "-R", "-f", tags_file, "--languages=all", path]
            subprocess.run(
                cmd,
                capture_output=True,
                cwd=str(context.repo_root),
                timeout=30
            )
        except subprocess.TimeoutExpired:
            return {"error": "ctags generation timeout", "symbols": []}
        except Exception as e:
            return {"error": f"ctags generation failed: {e}", "symbols": []}

    # Read and search the tags file
    symbols_list = []
    if os.path.exists(tags_file):
        try:
            with open(tags_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if line.startswith('!'):  # Skip comments
                        continue

                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        symbol_name = parts[0]
                        file_path = parts[1]
                        pattern = parts[2]

                        # Check if name matches (case-insensitive substring match)
                        if name.lower() in symbol_name.lower():
                            # Make path relative to repo root
                            try:
                                abs_path = Path(file_path).absolute()
                                rel_path = abs_path.relative_to(context.repo_root)
                                file_path = str(rel_path)
                            except ValueError:
                                pass  # Keep as is if can't make relative

                            # Extract metadata if available
                            kind = "unknown"
                            line_no: Optional[int] = None
                            for part in parts[3:]:
                                if part.startswith("kind:"):
                                    kind = part[5:]
                                elif part.startswith("line:"):
                                    try:
                                        line_no = int(part[5:])
                                    except ValueError:
                                        line_no = None

                            symbols_list.append({
                                "name": symbol_name,
                                "file": file_path,
                                "pattern": pattern,
                                "kind": kind,
                                "line": line_no,
                            })

                            if len(symbols_list) >= limit:
                                break

        except Exception as e:
            return {"error": f"Failed to read tags file: {e}", "symbols": []}

    # Sort by exact match first, then alphabetically
    symbols_list.sort(key=lambda s: (s["name"].lower() != name.lower(), s["name"].lower()))

    return {"symbols": symbols_list[:limit]}


# Register the tool
registry.register(
    ToolSpec(
        name="symbols",
        handler=symbols,
        request_schema_path=SCHEMA_DIR / "symbols.request.json",
        response_schema_path=SCHEMA_DIR / "symbols.response.json",
        description=(
            "Find symbol definitions using universal ctags.\n"
            "Examples:\n"
            "  symbols('MyClass') - find class definition\n"
            "  symbols('process_data') - find function definition\n"
            "  symbols('CONFIG', path='src/') - search only in src\n"
            "Requires ctags/universal-ctags to be installed.\n"
            "Searches are case-insensitive substring matches."
        ),
        category="symbol",
    )
)
