"""LSP client implementation with JSON-RPC communication."""
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from urllib.parse import quote

# We'll implement a minimal JSON-RPC client without external dependencies
# from pylsp_jsonrpc.dispatchers import MethodDispatcher
# from pylsp_jsonrpc.endpoint import Endpoint
# from pylsp_jsonrpc.streams import JsonRpcStreamReader, JsonRpcStreamWriter

LOGGER = logging.getLogger(__name__)


# Minimal JSON-RPC implementation
class JsonRpcEndpoint:
    """Minimal JSON-RPC 2.0 endpoint for LSP communication."""

    def __init__(self, reader, writer):
        self.reader = reader
        self.writer = writer
        self.request_id = 0
        self.pending_requests = {}

    async def request(self, method: str, params: Any = None) -> Any:
        """Send a request and wait for response."""
        self.request_id += 1
        req_id = self.request_id

        message = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params or {}
        }

        # Create a future for this request
        future = asyncio.Future()
        self.pending_requests[req_id] = future

        # Send request
        await self._write_message(message)

        # Wait for response (with timeout)
        try:
            result = await asyncio.wait_for(future, timeout=30.0)
            return result
        except asyncio.TimeoutError:
            LOGGER.warning("Request %s timed out", method)
            self.pending_requests.pop(req_id, None)
            return {}

    async def notify(self, method: str, params: Any = None) -> None:
        """Send a notification (no response expected)."""
        message = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {}
        }
        await self._write_message(message)

    async def _read_message(self) -> Optional[Dict]:
        """Read a JSON-RPC message from the reader."""
        try:
            # Read headers
            content_length = None
            while True:
                line = await self.reader.readline()
                if not line:
                    return None

                line = line.decode('utf-8').strip()
                if not line:
                    # Empty line marks end of headers
                    break

                if line.startswith('Content-Length:'):
                    content_length = int(line.split(':')[1].strip())

            if content_length is None:
                return None

            # Read content
            content = await self.reader.readexactly(content_length)
            message = json.loads(content.decode('utf-8'))
            return message

        except (asyncio.IncompleteReadError, json.JSONDecodeError) as e:
            LOGGER.debug("Error reading message: %s", e)
            return None

    async def _write_message(self, message: Dict) -> None:
        """Write a JSON-RPC message."""
        content = json.dumps(message)
        header = f"Content-Length: {len(content)}\r\n\r\n"
        data = header.encode() + content.encode()
        self.writer.write(data)
        await self.writer.drain()

    def consume(self, message: Dict, client=None) -> None:
        """Process incoming message.

        Args:
            message: The JSON-RPC message
            client: The LSPClient instance for handling notifications
        """
        # Handle responses to requests
        if "id" in message and "method" not in message:
            req_id = message["id"]
            if req_id in self.pending_requests:
                future = self.pending_requests.pop(req_id)
                if "error" in message:
                    # Request failed
                    future.set_exception(Exception(message["error"]))
                else:
                    # Request succeeded
                    future.set_result(message.get("result", {}))

        # Handle notifications
        elif "method" in message and "id" not in message:
            # This is a notification from server
            if client:
                method = message["method"]
                params = message.get("params", {})

                # Convert method name to handler (e.g., "textDocument/publishDiagnostics" -> "textDocument_publishDiagnostics")
                handler_name = method.replace("/", "_")
                handler = getattr(client, handler_name, None)

                if handler:
                    try:
                        handler(**params)
                    except Exception as e:
                        LOGGER.debug("Error in notification handler %s: %s", handler_name, e)


@dataclass
class Diagnostic:
    """LSP diagnostic message."""
    message: str
    severity: int  # 1=Error, 2=Warning, 3=Info, 4=Hint
    line: int
    column: int
    source: Optional[str] = None
    code: Optional[str] = None

    @classmethod
    def from_lsp(cls, lsp_diagnostic: Dict[str, Any]) -> "Diagnostic":
        """Create from LSP diagnostic format."""
        return cls(
            message=lsp_diagnostic.get("message", ""),
            severity=lsp_diagnostic.get("severity", 1),
            line=lsp_diagnostic["range"]["start"]["line"],
            column=lsp_diagnostic["range"]["start"]["character"],
            source=lsp_diagnostic.get("source"),
            code=str(lsp_diagnostic.get("code")) if "code" in lsp_diagnostic else None
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "message": self.message,
            "severity": self.severity,
            "line": self.line,
            "column": self.column
        }
        if self.source:
            result["source"] = self.source
        if self.code:
            result["code"] = self.code
        return result


class LSPClient:
    """Async LSP client with JSON-RPC communication."""

    def __init__(
        self,
        process: asyncio.subprocess.Process,
        root: Path,
        server_id: str = "unknown"
    ):
        """Initialize LSP client.

        Args:
            process: The LSP server subprocess
            root: Workspace root directory
            server_id: Identifier for the LSP server
        """
        self.process = process
        self.root = Path(root)
        self.server_id = server_id
        self.diagnostics: Dict[Path, List[Diagnostic]] = {}

        # Setup JSON-RPC communication
        self.endpoint = JsonRpcEndpoint(process.stdout, process.stdin)

        # Track open files
        self._open_files: Dict[Path, int] = {}  # Path -> version

        # Initialization state
        self._initialized = False
        self._capabilities: Dict[str, Any] = {}
        self._shutdown = False

        # Start listening task to process server messages
        self._listen_task = asyncio.create_task(self._listen())

    async def _listen(self) -> None:
        """Listen for server messages and dispatch them."""
        try:
            while not self._shutdown:
                message = await self.endpoint._read_message()
                if message is None:
                    # Connection closed
                    break

                # Process the message through the endpoint
                self.endpoint.consume(message, client=self)

        except asyncio.CancelledError:
            # Task was cancelled during shutdown
            pass
        except Exception as e:
            LOGGER.error("Error in listen loop: %s", e)

    async def initialize(
        self,
        initialization_options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Send initialize request to server.

        Args:
            initialization_options: Server-specific initialization options

        Returns:
            Server capabilities
        """
        if self._initialized:
            return self._capabilities

        params = {
            "processId": os.getpid(),
            "rootUri": self._path_to_uri(self.root),
            "rootPath": str(self.root),
            "capabilities": {
                "textDocument": {
                    "synchronization": {
                        "dynamicRegistration": False,
                        "willSave": False,
                        "didSave": True,
                        "willSaveWaitUntil": False
                    },
                    "completion": {
                        "dynamicRegistration": False,
                        "completionItem": {
                            "snippetSupport": False
                        }
                    },
                    "hover": {
                        "dynamicRegistration": False
                    },
                    "signatureHelp": {
                        "dynamicRegistration": False
                    },
                    "references": {
                        "dynamicRegistration": False
                    },
                    "documentSymbol": {
                        "dynamicRegistration": False,
                        "hierarchicalDocumentSymbolSupport": True
                    },
                    "definition": {
                        "dynamicRegistration": False
                    },
                    "codeAction": {
                        "dynamicRegistration": False
                    },
                    "publishDiagnostics": {
                        "relatedInformation": True
                    }
                },
                "workspace": {
                    "applyEdit": False,
                    "workspaceEdit": {
                        "documentChanges": False
                    },
                    "symbol": {
                        "dynamicRegistration": False
                    },
                    "configuration": False
                }
            },
            "trace": "off"
        }

        if initialization_options:
            params["initializationOptions"] = initialization_options

        result = await self.endpoint.request("initialize", params)
        self._capabilities = result.get("capabilities", {})

        # Send initialized notification
        await self.endpoint.notify("initialized", {})

        self._initialized = True
        return self._capabilities

    async def did_open(self, file_path: Path, language_id: Optional[str] = None) -> None:
        """Notify server of file open.

        Args:
            file_path: Path to the file
            language_id: Language identifier (auto-detected if None)
        """
        if not file_path.exists():
            return

        # Auto-detect language if not provided
        if not language_id:
            language_id = self._detect_language_id(file_path)

        content = file_path.read_text(encoding="utf-8")

        # Track version
        self._open_files[file_path] = 1

        await self.endpoint.notify(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": self._path_to_uri(file_path),
                    "languageId": language_id,
                    "version": 1,
                    "text": content
                }
            }
        )

    async def did_change(self, file_path: Path, version: Optional[int] = None) -> None:
        """Notify server of file change.

        Args:
            file_path: Path to the file
            version: Document version (auto-incremented if None)
        """
        if not file_path.exists():
            return

        if file_path not in self._open_files:
            await self.did_open(file_path)
            return

        # Auto-increment version
        if version is None:
            version = self._open_files[file_path] + 1

        self._open_files[file_path] = version
        content = file_path.read_text(encoding="utf-8")

        await self.endpoint.notify(
            "textDocument/didChange",
            {
                "textDocument": {
                    "uri": self._path_to_uri(file_path),
                    "version": version
                },
                "contentChanges": [
                    {
                        "text": content
                    }
                ]
            }
        )

    async def did_save(self, file_path: Path) -> None:
        """Notify server of file save.

        Args:
            file_path: Path to the file
        """
        if file_path not in self._open_files:
            return

        await self.endpoint.notify(
            "textDocument/didSave",
            {
                "textDocument": {
                    "uri": self._path_to_uri(file_path)
                }
            }
        )

    async def did_close(self, file_path: Path) -> None:
        """Notify server of file close.

        Args:
            file_path: Path to the file
        """
        if file_path not in self._open_files:
            return

        del self._open_files[file_path]

        await self.endpoint.notify(
            "textDocument/didClose",
            {
                "textDocument": {
                    "uri": self._path_to_uri(file_path)
                }
            }
        )

    async def get_diagnostics(self, file_path: Path) -> List[Diagnostic]:
        """Get current diagnostics for a file.

        Args:
            file_path: Path to the file

        Returns:
            List of diagnostics
        """
        return self.diagnostics.get(file_path, [])

    async def wait_for_diagnostics(
        self,
        file_path: Path,
        timeout: float = 3.0
    ) -> List[Diagnostic]:
        """Wait for diagnostics to be updated.

        Args:
            file_path: Path to the file
            timeout: Maximum time to wait in seconds

        Returns:
            List of diagnostics
        """
        # Clear current diagnostics to detect update
        old_count = len(self.diagnostics.get(file_path, []))

        # Wait for update
        start_time = asyncio.get_event_loop().time()
        while (asyncio.get_event_loop().time() - start_time) < timeout:
            await asyncio.sleep(0.1)

            current = self.diagnostics.get(file_path, [])
            if len(current) != old_count:
                # Diagnostics updated
                return current

        # Timeout, return current diagnostics
        return self.diagnostics.get(file_path, [])

    async def get_hover(
        self,
        file_path: Path,
        line: int,
        character: int
    ) -> Optional[Dict[str, Any]]:
        """Get hover information at position.

        Args:
            file_path: Path to the file
            line: Line number (0-indexed)
            character: Character position (0-indexed)

        Returns:
            Hover information or None
        """
        if not self._initialized:
            return None

        try:
            result = await self.endpoint.request(
                "textDocument/hover",
                {
                    "textDocument": {
                        "uri": self._path_to_uri(file_path)
                    },
                    "position": {
                        "line": line,
                        "character": character
                    }
                }
            )
            return result
        except Exception as e:
            LOGGER.debug("Hover request failed: %s", e)
            return None

    async def get_document_symbols(self, file_path: Path) -> List[Dict[str, Any]]:
        """Get document symbols.

        Args:
            file_path: Path to the file

        Returns:
            List of symbols
        """
        if not self._initialized:
            return []

        try:
            result = await self.endpoint.request(
                "textDocument/documentSymbol",
                {
                    "textDocument": {
                        "uri": self._path_to_uri(file_path)
                    }
                }
            )
            return result or []
        except Exception as e:
            LOGGER.debug("Document symbols request failed: %s", e)
            return []

    async def workspace_symbol(self, query: str = "") -> List[Dict[str, Any]]:
        """Search for symbols across workspace.

        Args:
            query: Search query

        Returns:
            List of matching symbols
        """
        if not self._initialized:
            return []

        try:
            result = await self.endpoint.request(
                "workspace/symbol",
                {
                    "query": query
                }
            )
            return result or []
        except Exception as e:
            LOGGER.debug("Workspace symbol request failed: %s", e)
            return []

    async def get_references(
        self,
        file_path: Path,
        line: int,
        character: int,
        include_declaration: bool = False
    ) -> List[Dict[str, Any]]:
        """Find references to symbol at position.

        Args:
            file_path: Path to the file
            line: Line number (0-indexed)
            character: Character position (0-indexed)
            include_declaration: Whether to include the declaration

        Returns:
            List of reference locations
        """
        if not self._initialized:
            return []

        try:
            result = await self.endpoint.request(
                "textDocument/references",
                {
                    "textDocument": {
                        "uri": self._path_to_uri(file_path)
                    },
                    "position": {
                        "line": line,
                        "character": character
                    },
                    "context": {
                        "includeDeclaration": include_declaration
                    }
                }
            )
            return result or []
        except Exception as e:
            LOGGER.debug("References request failed: %s", e)
            return []

    async def shutdown(self) -> None:
        """Shutdown the LSP server."""
        if not self._initialized:
            return

        try:
            # Signal shutdown
            self._shutdown = True

            # Request shutdown
            await self.endpoint.request("shutdown", {})
            # Send exit notification
            await self.endpoint.notify("exit", {})
        except Exception as e:
            LOGGER.debug("Error during shutdown: %s", e)
        finally:
            # Cancel listen task
            if hasattr(self, '_listen_task') and not self._listen_task.done():
                self._listen_task.cancel()
                try:
                    await self._listen_task
                except asyncio.CancelledError:
                    pass

            # Terminate process
            if self.process.returncode is None:
                self.process.terminate()
                try:
                    await asyncio.wait_for(self.process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    self.process.kill()
                    await self.process.wait()

            self._initialized = False

    # LSP notification handlers
    def textDocument_publishDiagnostics(self, **params) -> None:
        """Handle diagnostic notifications from server."""
        uri = params.get("uri", "")
        diagnostics = params.get("diagnostics", [])

        # Convert URI to path
        path = self._uri_to_path(uri)
        if path:
            # Convert to our Diagnostic format
            self.diagnostics[path] = [
                Diagnostic.from_lsp(d) for d in diagnostics
            ]

    def window_logMessage(self, **params) -> None:
        """Handle log messages from server."""
        message = params.get("message", "")
        message_type = params.get("type", 4)  # 1=Error, 2=Warning, 3=Info, 4=Log

        log_levels = {
            1: logging.ERROR,
            2: logging.WARNING,
            3: logging.INFO,
            4: logging.DEBUG
        }
        level = log_levels.get(message_type, logging.DEBUG)
        LOGGER.log(level, "LSP server %s: %s", self.server_id, message)

    def window_showMessage(self, **params) -> None:
        """Handle show message notifications from server."""
        # Similar to logMessage but meant for user display
        self.window_logMessage(**params)

    # Helper methods
    def _path_to_uri(self, path: Path) -> str:
        """Convert file path to URI."""
        path_str = str(path.absolute()).replace('\\', '/')
        return f"file://{quote(path_str, safe='/')}"

    def _uri_to_path(self, uri: str) -> Optional[Path]:
        """Convert URI to file path."""
        if not uri.startswith("file://"):
            return None

        path_str = uri[7:]  # Remove "file://"
        # Handle Windows paths
        if len(path_str) > 2 and path_str[0] == "/" and path_str[2] == ":":
            path_str = path_str[1:]

        return Path(path_str)

    def _detect_language_id(self, file_path: Path) -> str:
        """Detect language ID from file extension."""
        extension_map = {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascriptreact",
            ".ts": "typescript",
            ".tsx": "typescriptreact",
            ".java": "java",
            ".go": "go",
            ".rs": "rust",
            ".c": "c",
            ".cpp": "cpp",
            ".cc": "cpp",
            ".cxx": "cpp",
            ".h": "c",
            ".hpp": "cpp",
            ".cs": "csharp",
            ".rb": "ruby",
            ".php": "php",
            ".swift": "swift",
            ".kt": "kotlin",
            ".scala": "scala",
            ".r": "r",
            ".lua": "lua",
            ".dart": "dart",
            ".elm": "elm",
            ".ex": "elixir",
            ".exs": "elixir",
            ".clj": "clojure",
            ".hs": "haskell",
            ".ml": "ocaml",
            ".vim": "vim",
            ".sh": "shellscript",
            ".bash": "shellscript",
            ".zsh": "shellscript",
            ".fish": "fish",
            ".ps1": "powershell",
            ".yml": "yaml",
            ".yaml": "yaml",
            ".json": "json",
            ".xml": "xml",
            ".html": "html",
            ".css": "css",
            ".scss": "scss",
            ".sass": "sass",
            ".less": "less",
            ".vue": "vue",
            ".svelte": "svelte",
            ".md": "markdown",
            ".markdown": "markdown",
            ".tex": "latex",
            ".rst": "restructuredtext",
            ".toml": "toml",
            ".ini": "ini",
            ".cfg": "ini",
            ".conf": "ini",
            ".sql": "sql",
            ".dockerfile": "dockerfile",
            ".makefile": "makefile",
            ".mk": "makefile",
            ".cmake": "cmake"
        }

        suffix = file_path.suffix.lower()
        return extension_map.get(suffix, "plaintext")

    @property
    def notify(self):
        """Convenience property for notification methods."""
        class NotificationMethods:
            def __init__(self, client):
                self.client = client

            async def open(self, params: Dict[str, Any]) -> None:
                path = params.get("path")
                if path:
                    await self.client.did_open(Path(path))

            async def change(self, params: Dict[str, Any]) -> None:
                path = params.get("path")
                if path:
                    await self.client.did_change(Path(path))

            async def save(self, params: Dict[str, Any]) -> None:
                path = params.get("path")
                if path:
                    await self.client.did_save(Path(path))

            async def close(self, params: Dict[str, Any]) -> None:
                path = params.get("path")
                if path:
                    await self.client.did_close(Path(path))

        return NotificationMethods(self)

    @property
    def connection(self):
        """Compatibility property for connection-based operations."""
        class ConnectionMethods:
            def __init__(self, client):
                self.client = client

            async def sendRequest(self, method: str, params: Any) -> Any:
                return await self.client.endpoint.request(method, params)

            async def sendNotification(self, method: str, params: Any) -> None:
                await self.client.endpoint.notify(method, params)

        return ConnectionMethods(self)