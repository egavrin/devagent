"""Fast workspace-wide symbol search via LSP."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

from ai_dev_agent.lsp import ServerRegistry
from ai_dev_agent.tools.registry import ToolSpec, registry, ToolContext

@registry.register
class WorkspaceSymbolTool:
    """Search for symbols across entire workspace using LSP."""

    spec = ToolSpec(
        name="workspace_symbols",
        description=(
            "Search for functions, classes, variables across the entire workspace. "
            "Much faster than grep for finding symbol definitions. "
            "Returns symbol locations with type information when available."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Symbol name or partial name to search for"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 20)"
                }
            },
            "required": ["query"]
        }
    )

    def __init__(self):
        self.registry: Optional[ServerRegistry] = None

    def _ensure_registry(self, context: ToolContext) -> ServerRegistry:
        """Ensure registry is initialized."""
        if self.registry is None:
            self.registry = ServerRegistry(context.workspace, auto_install=True)
        return self.registry

    def execute(self, context: ToolContext) -> Dict[str, Any]:
        """Execute workspace symbol search synchronously."""
        # Run async method in sync context
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._execute_async(context))
        finally:
            loop.close()

    async def _execute_async(self, context: ToolContext) -> Dict[str, Any]:
        """Execute workspace symbol search asynchronously."""
        query = context.args.get("query", "")
        limit = context.args.get("limit", 20)

        registry = self._ensure_registry(context)

        # Search across all active LSP clients
        all_symbols = await registry.workspace_symbol_search(query)

        # If no LSP clients active, try to start one for common files
        if not all_symbols and not registry.active_clients():
            # Try to find and open a common file to start an LSP server
            for ext in [".py", ".js", ".ts", ".go", ".rs", ".java"]:
                sample_files = list(context.workspace.rglob(f"*{ext}"))[:1]
                if sample_files:
                    await registry.get_or_create_client(sample_files[0])
                    break

            # Retry search
            all_symbols = await registry.workspace_symbol_search(query)

        # Format results
        formatted = []
        for symbol in all_symbols[:limit]:
            # Extract file path from URI
            uri = symbol.get("location", {}).get("uri", "")
            file_path = uri.replace("file://", "") if uri.startswith("file://") else uri

            # Make path relative to workspace
            try:
                rel_path = Path(file_path).relative_to(context.workspace)
            except (ValueError, RuntimeError):
                rel_path = Path(file_path)

            formatted.append({
                "name": symbol.get("name", ""),
                "kind": self._get_symbol_kind_name(symbol.get("kind", 0)),
                "file": str(rel_path),
                "line": symbol.get("location", {}).get("range", {}).get("start", {}).get("line", 0) + 1,
                "column": symbol.get("location", {}).get("range", {}).get("start", {}).get("character", 0) + 1
            })

        return {
            "success": True,
            "query": query,
            "results": formatted,
            "total": len(formatted),
            "truncated": len(all_symbols) > limit
        }

    def _get_symbol_kind_name(self, kind: int) -> str:
        """Convert LSP SymbolKind number to name."""
        kinds = {
            1: "file",
            2: "module",
            3: "namespace",
            4: "package",
            5: "class",
            6: "method",
            7: "property",
            8: "field",
            9: "constructor",
            10: "enum",
            11: "interface",
            12: "function",
            13: "variable",
            14: "constant",
            15: "string",
            16: "number",
            17: "boolean",
            18: "array",
            19: "object",
            20: "key",
            21: "null",
            22: "enum_member",
            23: "struct",
            24: "event",
            25: "operator",
            26: "type_parameter"
        }
        return kinds.get(kind, "unknown")