"""Unified context builder - single source of truth for context building."""

import logging
import os
import platform
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ai_dev_agent.core.utils.constants import DEFAULT_IGNORED_REPO_DIRS
from ai_dev_agent.core.utils.repo_outline import generate_repo_outline

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Unified context builder for all context needs."""

    # OS friendly names
    _OS_FRIENDLY_NAMES = {
        "Darwin": "macOS",
        "Linux": "Linux",
        "Windows": "Windows",
    }

    # Common development tools to check for
    _TOOL_CANDIDATES = [
        "gh",
        "git",
        "docker",
        "kubectl",
        "npm",
        "pip",
        "python",
        "python3",
        "node",
        "make",
        "cmake",
        "curl",
        "jq",
        "psql",
        "sqlite3",
        "mongosh",
        "gcc",
        "clang",
        "cargo",
        "go",
        "java",
        "mvn",
        "gradle",
    ]

    # Platform-specific command mappings
    _COMMAND_MAPPINGS = {
        "Darwin": {
            "list_files": "ls -la",
            "find_files": "find",
            "copy": "cp",
            "move": "mv",
            "delete": "rm",
            "open_file": "open",
        },
        "Linux": {
            "list_files": "ls -la",
            "find_files": "find",
            "copy": "cp",
            "move": "mv",
            "delete": "rm",
            "open_file": "xdg-open",
        },
        "Windows": {
            "list_files": "dir",
            "find_files": "dir /s",
            "copy": "copy",
            "move": "move",
            "delete": "del",
            "open_file": "start",
        },
    }

    # Platform-specific usage examples
    _PLATFORM_EXAMPLES = {
        "Darwin": "e.g. list files with 'ls -la', open with 'open README.md'",
        "Linux": "e.g. list files with 'ls -la', open with 'xdg-open README.md'",
        "Windows": "e.g. list files with 'dir', open with 'start README.md'",
    }

    def __init__(self, workspace: Optional[Path] = None):
        """Initialize the context builder.

        Args:
            workspace: The workspace path (defaults to current directory)
        """
        self.workspace = workspace or Path.cwd()

    def build_system_context(self) -> Dict[str, Any]:
        """Build system/environment context.

        Returns:
            Dictionary containing system information
        """
        system = platform.system() or "unknown"
        os_friendly = self._OS_FRIENDLY_NAMES.get(system, system or "Unknown")

        if system == "Darwin":
            os_version = platform.mac_ver()[0] or platform.release()
        elif system == "Windows":
            os_version = platform.version()
        else:
            os_version = platform.release()

        architecture = platform.machine() or platform.processor() or "unknown"

        shell = os.environ.get("SHELL")
        if not shell:
            shell = os.environ.get("COMSPEC", "cmd.exe") if system == "Windows" else "/bin/sh"

        cwd = str(self.workspace)
        home_dir = str(Path.home())
        python_version = platform.python_version()

        is_unix = system in {"Darwin", "Linux"}
        shell_type = "unix" if is_unix else "windows"
        path_separator = "/" if is_unix else "\\"
        command_separator = "&&" if is_unix else "&"
        null_device = "/dev/null" if is_unix else "NUL"
        temp_dir = tempfile.gettempdir()

        available_tools = [tool for tool in self._TOOL_CANDIDATES if shutil.which(tool)]

        # Get platform-specific command mappings and examples
        command_mappings = self._COMMAND_MAPPINGS.get(system, self._COMMAND_MAPPINGS.get("Linux"))
        platform_examples = self._PLATFORM_EXAMPLES.get(
            system, self._PLATFORM_EXAMPLES.get("Linux")
        )

        return {
            "os": system,
            "os_friendly": os_friendly,
            "os_version": os_version,
            "architecture": architecture,
            "shell": shell,
            "shell_type": shell_type,
            "cwd": cwd,
            "home_dir": home_dir,
            "python_version": python_version,
            "path_separator": path_separator,
            "command_separator": command_separator,
            "null_device": null_device,
            "temp_dir": temp_dir,
            "available_tools": available_tools,
            "is_unix": is_unix,
            "command_mappings": command_mappings,
            "platform_examples": platform_examples,
        }

    def build_project_context(self, include_outline: bool = False) -> Dict[str, Any]:
        """Build project-specific context.

        Args:
            include_outline: Whether to include project structure outline

        Returns:
            Dictionary containing project information
        """
        context = {
            "workspace": str(self.workspace),
            "workspace_name": self.workspace.name,
        }

        # Check for common project files
        if (self.workspace / ".git").exists():
            context["has_git"] = True
            try:
                # Get current branch
                git_head = self.workspace / ".git" / "HEAD"
                if git_head.exists():
                    content = git_head.read_text().strip()
                    if content.startswith("ref: refs/heads/"):
                        context["git_branch"] = content.replace("ref: refs/heads/", "")
            except Exception as e:
                logger.debug(f"Failed to read git info: {e}")

        # Check for common config files
        for config_file in ["pyproject.toml", "setup.py", "package.json", "Cargo.toml"]:
            if (self.workspace / config_file).exists():
                context[f"has_{config_file.replace('.', '_')}"] = True

        # Count files
        context["python_files_count"] = self._count_python_files()

        # Add project structure outline if requested
        if include_outline:
            outline = self.get_project_structure_outline()
            if outline:
                context["project_outline"] = outline

        return context

    def _count_python_files(self) -> int:
        """Count Python files while skipping ignored locations."""
        if not self.workspace.exists():
            return 0

        base = self.workspace.resolve()
        gitignore_dirs, gitignore_files = self._load_gitignore_entries()

        count = 0
        stack = [base]

        while stack:
            current = stack.pop()
            try:
                entries = list(current.iterdir())
            except OSError:
                continue

            for entry in entries:
                try:
                    relative = entry.relative_to(base)
                except ValueError:
                    continue

                if self._is_hidden_path(relative):
                    continue

                if self._is_in_default_ignored(relative):
                    continue

                if self._is_gitignored(relative, gitignore_dirs, gitignore_files):
                    continue

                if entry.is_dir():
                    stack.append(entry)
                    continue

                if entry.is_file() and entry.suffix == ".py":
                    count += 1

        return count

    def _load_gitignore_entries(self) -> Tuple[set[Path], set[Path]]:
        """Load directory and file entries from .gitignore."""
        ignored_dirs: set[Path] = set()
        ignored_files: set[Path] = set()

        gitignore = self.workspace / ".gitignore"
        if not gitignore.is_file():
            return ignored_dirs, ignored_files

        try:
            lines = gitignore.read_text(encoding="utf-8").splitlines()
        except OSError:
            return ignored_dirs, ignored_files

        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("!"):
                continue
            if any(char in line for char in "*?["):
                continue

            if line.startswith("/"):
                line = line[1:]

            is_directory = line.endswith("/")
            if is_directory:
                line = line.rstrip("/")

            if not line:
                continue

            path = Path(line)
            path = Path(*[part for part in path.parts if part not in {".", ""}])
            if not path.parts:
                continue

            if is_directory or (self.workspace / path).is_dir():
                ignored_dirs.add(path)
            else:
                ignored_files.add(path)

        return ignored_dirs, ignored_files

    @staticmethod
    def _is_hidden_path(relative: Path) -> bool:
        return any(part.startswith(".") and part not in {".", ".."} for part in relative.parts)

    def _is_in_default_ignored(self, relative: Path) -> bool:
        return any(part in DEFAULT_IGNORED_REPO_DIRS for part in relative.parts if part)

    @staticmethod
    def _is_gitignored(relative: Path, ignored_dirs: set[Path], ignored_files: set[Path]) -> bool:
        if relative in ignored_files:
            return True

        for directory in ignored_dirs:
            if ContextBuilder._path_is_within(relative, directory):
                return True

        return False

    @staticmethod
    def _path_is_within(path: Path, ancestor: Path) -> bool:
        ancestor_parts = ancestor.parts
        path_parts = path.parts
        if len(ancestor_parts) > len(path_parts):
            return False
        return path_parts[: len(ancestor_parts)] == ancestor_parts

    def get_project_structure_outline(
        self, max_entries: int = 160, max_depth: int = 3, directories_only: bool = False
    ) -> Optional[str]:
        """Get a concise project structure outline.

        Args:
            max_entries: Maximum number of entries to include
            max_depth: Maximum directory depth
            directories_only: Whether to show only directories

        Returns:
            Project outline string or None if generation fails
        """
        return generate_repo_outline(
            self.workspace,
            max_entries=max_entries,
            max_depth=max_depth,
            directories_only=directories_only,
        )

    def build_agent_context(self, agent_type: str, **kwargs) -> Dict[str, Any]:
        """Build context for a specific agent.

        Args:
            agent_type: Type of agent (design, implementation, review, test)
            **kwargs: Additional context parameters

        Returns:
            Dictionary containing agent-specific context
        """
        base_context = {
            "agent_type": agent_type,
            "workspace": str(self.workspace),
        }

        # Add system context
        base_context.update(self.build_system_context())

        # Add project context
        base_context.update(self.build_project_context())

        # Add any additional kwargs
        base_context.update(kwargs)

        return base_context

    def build_tool_context(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Build context for a specific tool.

        Args:
            tool_name: Name of the tool
            **kwargs: Additional context parameters

        Returns:
            Dictionary containing tool-specific context
        """
        context = {
            "tool_name": tool_name,
            "workspace": str(self.workspace),
        }

        # Add system info for tools that need it
        if tool_name in ["bash", "shell", "terminal"]:
            context.update(self.build_system_context())

        # Add any additional kwargs
        context.update(kwargs)

        return context

    def build_full_context(
        self, include_system: bool = True, include_project: bool = True
    ) -> Dict[str, Any]:
        """Build complete context with all available information.

        Args:
            include_system: Whether to include system context
            include_project: Whether to include project context

        Returns:
            Dictionary containing full context
        """
        context = {}

        if include_system:
            context.update(self.build_system_context())

        if include_project:
            context.update(self.build_project_context())

        return context


# Singleton instance for convenience
_default_builder: Optional[ContextBuilder] = None


def get_context_builder(workspace: Optional[Path] = None) -> ContextBuilder:
    """Get or create the default context builder.

    Args:
        workspace: Optional workspace path

    Returns:
        ContextBuilder instance
    """
    global _default_builder
    if _default_builder is None or (workspace and workspace != _default_builder.workspace):
        _default_builder = ContextBuilder(workspace)
    return _default_builder
