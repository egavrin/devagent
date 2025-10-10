"""LSP server registry and auto-installation management."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from ai_dev_agent.lsp.client import LSPClient

LOGGER = logging.getLogger(__name__)


@dataclass
class ServerInfo:
    """LSP server configuration."""
    id: str
    name: str
    extensions: List[str]
    root_markers: List[str]
    spawn: Callable[[Path], asyncio.subprocess.Process]
    auto_install: bool = True
    initialization_options: Optional[Dict[str, Any]] = None


class ServerRegistry:
    """Manages LSP server definitions, lifecycle, and auto-installation."""

    def __init__(self, workspace: Path, auto_install: bool = True):
        """Initialize server registry.

        Args:
            workspace: Workspace root directory
            auto_install: Whether to auto-install missing servers
        """
        self.workspace = Path(workspace)
        self.auto_install = auto_install
        self.servers: Dict[str, ServerInfo] = {}
        self.clients: Dict[str, LSPClient] = {}  # server_id:root -> client

        # Binary directory for installed servers
        self.bin_dir = Path.home() / ".devagent" / "lsp" / "bin"
        self.bin_dir.mkdir(parents=True, exist_ok=True)

        # Add bin directory to PATH
        os.environ["PATH"] = f"{self.bin_dir}:{os.environ.get('PATH', '')}"

        # Register built-in servers
        self._register_builtin_servers()

    def _register_builtin_servers(self) -> None:
        """Register commonly used LSP servers."""

        # Python (Pyright)
        self.servers["python"] = ServerInfo(
            id="pyright",
            name="Pyright (Python)",
            extensions=[".py", ".pyi"],
            root_markers=["pyproject.toml", "setup.py", "requirements.txt", "Pipfile"],
            spawn=self._spawn_pyright,
            initialization_options={}
        )

        # TypeScript/JavaScript
        self.servers["typescript"] = ServerInfo(
            id="typescript-language-server",
            name="TypeScript Language Server",
            extensions=[".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"],
            root_markers=["tsconfig.json", "jsconfig.json", "package.json"],
            spawn=self._spawn_typescript
        )

        # Go
        self.servers["go"] = ServerInfo(
            id="gopls",
            name="gopls (Go)",
            extensions=[".go"],
            root_markers=["go.mod", "go.sum", "go.work"],
            spawn=self._spawn_gopls
        )

        # Rust
        self.servers["rust"] = ServerInfo(
            id="rust-analyzer",
            name="rust-analyzer (Rust)",
            extensions=[".rs"],
            root_markers=["Cargo.toml", "Cargo.lock"],
            spawn=self._spawn_rust_analyzer
        )

        # C/C++
        self.servers["cpp"] = ServerInfo(
            id="clangd",
            name="clangd (C/C++)",
            extensions=[".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".hxx"],
            root_markers=["compile_commands.json", "compile_flags.txt", ".clangd", "CMakeLists.txt"],
            spawn=self._spawn_clangd
        )

        # Java
        self.servers["java"] = ServerInfo(
            id="jdtls",
            name="Eclipse JDT Language Server (Java)",
            extensions=[".java"],
            root_markers=["pom.xml", "build.gradle", "build.gradle.kts", ".project"],
            spawn=self._spawn_jdtls
        )

        # Ruby
        self.servers["ruby"] = ServerInfo(
            id="solargraph",
            name="Solargraph (Ruby)",
            extensions=[".rb", ".rake", ".gemspec"],
            root_markers=["Gemfile", "Gemfile.lock", ".solargraph.yml"],
            spawn=self._spawn_solargraph
        )

        # PHP
        self.servers["php"] = ServerInfo(
            id="intelephense",
            name="Intelephense (PHP)",
            extensions=[".php"],
            root_markers=["composer.json", "composer.lock"],
            spawn=self._spawn_intelephense
        )

        # C#
        self.servers["csharp"] = ServerInfo(
            id="omnisharp",
            name="OmniSharp (C#)",
            extensions=[".cs", ".csx"],
            root_markers=[".csproj", ".sln", "project.json"],
            spawn=self._spawn_omnisharp
        )

    # Server spawn methods
    async def _spawn_pyright(self, root: Path) -> asyncio.subprocess.Process:
        """Start Pyright LSP server."""
        binary = await self._ensure_binary("pyright-langserver")

        if not binary:
            # Try to install via npm
            if self.auto_install and shutil.which("npm"):
                LOGGER.info("Installing Pyright...")
                result = subprocess.run(
                    ["npm", "install", "-g", "pyright"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    binary = shutil.which("pyright-langserver")

        if not binary:
            raise RuntimeError("Pyright not found. Install with: npm install -g pyright")

        # Detect virtual environment
        venv_path = self._find_python_venv(root)
        env = os.environ.copy()

        if venv_path:
            python_path = venv_path / ("Scripts" if sys.platform == "win32" else "bin") / "python"
            if python_path.exists():
                env["PYRIGHT_PYTHON_PATH"] = str(python_path)

        return await asyncio.create_subprocess_exec(
            binary, "--stdio",
            cwd=root,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

    async def _spawn_typescript(self, root: Path) -> asyncio.subprocess.Process:
        """Start TypeScript Language Server."""
        binary = await self._ensure_binary("typescript-language-server")

        if not binary:
            if self.auto_install and shutil.which("npm"):
                LOGGER.info("Installing TypeScript Language Server...")
                result = subprocess.run(
                    ["npm", "install", "-g", "typescript-language-server", "typescript"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    binary = shutil.which("typescript-language-server")

        if not binary:
            raise RuntimeError("TypeScript Language Server not found. Install with: npm install -g typescript-language-server")

        return await asyncio.create_subprocess_exec(
            binary, "--stdio",
            cwd=root,
            stdout=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

    async def _spawn_gopls(self, root: Path) -> asyncio.subprocess.Process:
        """Start gopls LSP server."""
        binary = await self._ensure_binary("gopls")

        if not binary:
            if self.auto_install and shutil.which("go"):
                LOGGER.info("Installing gopls...")
                result = subprocess.run(
                    ["go", "install", "golang.org/x/tools/gopls@latest"],
                    capture_output=True,
                    text=True,
                    env={**os.environ, "GOBIN": str(self.bin_dir)}
                )
                if result.returncode == 0:
                    binary = self.bin_dir / "gopls"

        if not binary:
            raise RuntimeError("gopls not found. Install with: go install golang.org/x/tools/gopls@latest")

        return await asyncio.create_subprocess_exec(
            str(binary),
            cwd=root,
            stdout=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

    async def _spawn_rust_analyzer(self, root: Path) -> asyncio.subprocess.Process:
        """Start rust-analyzer LSP server."""
        binary = await self._ensure_binary("rust-analyzer")

        if not binary:
            if self.auto_install:
                LOGGER.info("Installing rust-analyzer...")
                # Try rustup first
                if shutil.which("rustup"):
                    result = subprocess.run(
                        ["rustup", "component", "add", "rust-analyzer"],
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0:
                        # Find rust-analyzer in rustup toolchain
                        rustup_home = os.environ.get("RUSTUP_HOME", Path.home() / ".rustup")
                        for rust_analyzer in Path(rustup_home).rglob("rust-analyzer"):
                            if rust_analyzer.is_file() and os.access(rust_analyzer, os.X_OK):
                                binary = rust_analyzer
                                break

        if not binary:
            raise RuntimeError("rust-analyzer not found. Install with: rustup component add rust-analyzer")

        return await asyncio.create_subprocess_exec(
            str(binary),
            cwd=root,
            stdout=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

    async def _spawn_clangd(self, root: Path) -> asyncio.subprocess.Process:
        """Start clangd LSP server."""
        binary = await self._ensure_binary("clangd")

        if not binary:
            raise RuntimeError("clangd not found. Install via your package manager or download from https://clangd.llvm.org")

        return await asyncio.create_subprocess_exec(
            binary, "--background-index",
            cwd=root,
            stdout=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

    async def _spawn_jdtls(self, root: Path) -> asyncio.subprocess.Process:
        """Start Eclipse JDT Language Server."""
        # JDTLS requires more complex setup
        jdtls_home = self.bin_dir / "jdtls"

        if not jdtls_home.exists() and self.auto_install:
            LOGGER.info("Installing Eclipse JDT Language Server...")
            # Download and extract JDTLS
            # This is simplified - real implementation would download from Eclipse
            pass

        if not jdtls_home.exists():
            raise RuntimeError("Eclipse JDT Language Server not found")

        java = shutil.which("java")
        if not java:
            raise RuntimeError("Java not found. JDTLS requires Java 11 or later")

        return await asyncio.create_subprocess_exec(
            java,
            "-jar", str(jdtls_home / "plugins" / "org.eclipse.equinox.launcher_*.jar"),
            "-configuration", str(jdtls_home / "config_linux"),  # Or config_win/config_mac
            "-data", str(root / ".jdtls-workspace"),
            cwd=root,
            stdout=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

    async def _spawn_solargraph(self, root: Path) -> asyncio.subprocess.Process:
        """Start Solargraph LSP server."""
        binary = await self._ensure_binary("solargraph")

        if not binary:
            if self.auto_install and shutil.which("gem"):
                LOGGER.info("Installing Solargraph...")
                result = subprocess.run(
                    ["gem", "install", "solargraph"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    binary = shutil.which("solargraph")

        if not binary:
            raise RuntimeError("Solargraph not found. Install with: gem install solargraph")

        return await asyncio.create_subprocess_exec(
            binary, "stdio",
            cwd=root,
            stdout=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

    async def _spawn_intelephense(self, root: Path) -> asyncio.subprocess.Process:
        """Start Intelephense LSP server."""
        binary = await self._ensure_binary("intelephense")

        if not binary:
            if self.auto_install and shutil.which("npm"):
                LOGGER.info("Installing Intelephense...")
                result = subprocess.run(
                    ["npm", "install", "-g", "intelephense"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    binary = shutil.which("intelephense")

        if not binary:
            raise RuntimeError("Intelephense not found. Install with: npm install -g intelephense")

        return await asyncio.create_subprocess_exec(
            binary, "--stdio",
            cwd=root,
            stdout=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

    async def _spawn_omnisharp(self, root: Path) -> asyncio.subprocess.Process:
        """Start OmniSharp LSP server."""
        binary = await self._ensure_binary("omnisharp")

        if not binary:
            # OmniSharp requires complex installation
            raise RuntimeError("OmniSharp not found. See https://github.com/OmniSharp/omnisharp-roslyn")

        return await asyncio.create_subprocess_exec(
            binary, "--lsp", "--hosting:stdio",
            cwd=root,
            stdout=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

    # Helper methods
    async def _ensure_binary(self, name: str) -> Optional[Path]:
        """Ensure a binary exists, checking standard locations.

        Args:
            name: Binary name

        Returns:
            Path to binary or None if not found
        """
        # Check PATH first
        binary = shutil.which(name)
        if binary:
            return Path(binary)

        # Check our bin directory
        local_binary = self.bin_dir / name
        if local_binary.exists() and os.access(local_binary, os.X_OK):
            return local_binary

        # Platform-specific locations
        if sys.platform == "win32":
            # Check common Windows locations
            for location in [
                Path(os.environ.get("PROGRAMFILES", "")) / name,
                Path(os.environ.get("LOCALAPPDATA", "")) / name,
            ]:
                if location.exists():
                    exe = location / f"{name}.exe"
                    if exe.exists():
                        return exe

        return None

    def _find_python_venv(self, root: Path) -> Optional[Path]:
        """Find Python virtual environment.

        Args:
            root: Project root

        Returns:
            Path to virtual environment or None
        """
        # Check VIRTUAL_ENV environment variable
        venv = os.environ.get("VIRTUAL_ENV")
        if venv:
            return Path(venv)

        # Check common venv locations
        for venv_name in [".venv", "venv", "env", ".env"]:
            venv_path = root / venv_name
            if venv_path.exists() and (venv_path / "bin" / "python").exists():
                return venv_path

        return None

    # Public API
    def find_server(self, file_path: Path) -> Optional[ServerInfo]:
        """Find appropriate LSP server for file.

        Args:
            file_path: Path to file

        Returns:
            Server info or None if no server available
        """
        ext = file_path.suffix.lower()

        for server in self.servers.values():
            if ext in server.extensions:
                return server

        return None

    def find_root(self, file_path: Path, markers: List[str]) -> Path:
        """Find project root by walking up and looking for markers.

        Args:
            file_path: Starting file path
            markers: Root marker files/directories

        Returns:
            Project root path
        """
        current = file_path.parent if file_path.is_file() else file_path

        while current != current.parent:
            for marker in markers:
                if (current / marker).exists():
                    return current
            current = current.parent

        # Default to workspace root
        return self.workspace

    async def get_or_create_client(self, file_path: Path) -> Optional[LSPClient]:
        """Get or create LSP client for a file.

        Args:
            file_path: Path to file

        Returns:
            LSP client or None if no server available
        """
        # Find appropriate server
        server_info = self.find_server(file_path)
        if not server_info:
            return None

        # Find project root
        root = self.find_root(file_path, server_info.root_markers)

        # Check for existing client
        client_key = f"{server_info.id}:{root}"
        if client_key in self.clients:
            return self.clients[client_key]

        # Create new client
        try:
            process = await server_info.spawn(root)
            client = LSPClient(process, root, server_info.id)

            # Initialize
            await client.initialize(server_info.initialization_options)

            # Cache client
            self.clients[client_key] = client

            LOGGER.info("Started LSP server %s for %s", server_info.name, root)
            return client

        except Exception as e:
            LOGGER.error("Failed to start LSP server %s: %s", server_info.name, e)
            return None

    async def workspace_symbol_search(self, query: str) -> List[Any]:
        """Search for symbols across all active LSP clients.

        Args:
            query: Search query

        Returns:
            List of symbols from all servers
        """
        results = []

        for client in self.clients.values():
            try:
                symbols = await client.workspace_symbol(query)
                results.extend(symbols)
            except Exception as e:
                LOGGER.debug("Workspace symbol search failed for %s: %s", client.server_id, e)

        return results

    def active_clients(self) -> List[LSPClient]:
        """Get all active LSP clients.

        Returns:
            List of active clients
        """
        return list(self.clients.values())

    async def shutdown_all(self) -> None:
        """Shutdown all LSP servers."""
        for client in self.clients.values():
            try:
                await client.shutdown()
            except Exception as e:
                LOGGER.error("Error shutting down LSP client: %s", e)

        self.clients.clear()