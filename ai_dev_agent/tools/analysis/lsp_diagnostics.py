"""LSP-powered diagnostics tool for real-time error detection."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

from ai_dev_agent.lsp import LSPClient, ServerRegistry
from ai_dev_agent.tools.registry import ToolSpec, registry, ToolContext

@registry.register
class LSPDiagnosticsTool:
    """Get real-time diagnostics from Language Server Protocol."""

    spec = ToolSpec(
        name="lsp_diagnostics",
        description=(
            "Check for errors, warnings, and hints in files using Language Server Protocol. "
            "Provides IDE-quality diagnostics including type errors, syntax issues, and linting warnings. "
            "Faster and more accurate than running tests or linters manually."
        ),
        parameters={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to file to check (relative to workspace root)"
                },
                "wait_for_update": {
                    "type": "boolean",
                    "description": "Whether to wait for diagnostics to update (default: true)"
                }
            },
            "required": ["file_path"]
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
        """Execute LSP diagnostics check synchronously."""
        # Run async method in sync context
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._execute_async(context))
        finally:
            loop.close()

    async def _execute_async(self, context: ToolContext) -> Dict[str, Any]:
        """Execute LSP diagnostics check asynchronously."""
        file_path = context.args.get("file_path")
        wait_for_update = context.args.get("wait_for_update", True)

        if not file_path:
            return {
                "success": False,
                "error": "file_path is required"
            }

        registry = self._ensure_registry(context)
        path = context.workspace / file_path

        if not path.exists():
            return {
                "success": False,
                "error": f"File not found: {file_path}"
            }

        # Get LSP client for this file type
        client = await registry.get_or_create_client(path)

        if not client:
            # Fallback message when no LSP available
            return {
                "success": True,
                "file": file_path,
                "diagnostics": [],
                "summary": "No LSP server available for this file type",
                "errors": 0,
                "warnings": 0,
                "info": 0,
                "hints": 0
            }

        # Open file in LSP (if not already)
        await client.did_open(path)

        # Wait for diagnostics if requested
        if wait_for_update:
            diagnostics = await client.wait_for_diagnostics(path, timeout=3.0)
        else:
            diagnostics = await client.get_diagnostics(path)

        # Format for LLM
        formatted = []
        severity_counts = {1: 0, 2: 0, 3: 0, 4: 0}

        for diag in diagnostics:
            severity_name = ["error", "warning", "info", "hint"][diag.severity - 1]
            severity_counts[diag.severity] += 1

            formatted.append({
                "severity": severity_name,
                "line": diag.line + 1,  # 1-indexed for humans
                "column": diag.column + 1,
                "message": diag.message,
                "source": diag.source or "lsp",
                "code": diag.code
            })

        # Sort by severity and line
        formatted.sort(key=lambda x: (["error", "warning", "info", "hint"].index(x["severity"]), x["line"]))

        return {
            "success": True,
            "file": file_path,
            "diagnostics": formatted,
            "summary": f"{len(formatted)} diagnostic(s) found",
            "errors": severity_counts[1],
            "warnings": severity_counts[2],
            "info": severity_counts[3],
            "hints": severity_counts[4]
        }