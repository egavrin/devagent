"""Simplified context gathering with text-based heuristics."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ai_dev_agent.core.repo_map import RepoMapManager
from ai_dev_agent.core.utils.constants import DEFAULT_IGNORED_REPO_DIRS
from ai_dev_agent.core.utils.logger import get_logger

from .tree_sitter_analysis import TreeSitterProjectAnalyzer, extract_symbols_from_outline

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

LOGGER = get_logger(__name__)

_EXCLUDED_DIR_PATTERNS = tuple(sorted({f"{name}/*" for name in DEFAULT_IGNORED_REPO_DIRS}))


@dataclass
class FileContext:
    path: Path
    content: str
    relevance_score: float = 0.0
    reason: str = "explicitly_requested"
    structure_outline: list[str] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)
    size_bytes: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.size_bytes = len(self.content.encode("utf-8"))


@dataclass
class ContextGatheringOptions:
    max_files: int = 20
    max_total_size: int = 100_000
    include_structure_summary: bool = True
    include_related_files: bool = True
    keyword_match_limit: int = 5
    use_repo_map: bool = True  # NEW: Enable RepoMap ranking
    exclude_patterns: Sequence[str] = (
        "*.pyc",
        "*.pyo",
        "*.min.js",
        "*.bundle.js",
        "*.log",
        "*.tmp",
        ".env",
        *_EXCLUDED_DIR_PATTERNS,
    )


class ContextGatherer:
    """Gather file contexts using lightweight discovery."""

    def __init__(
        self,
        repo_root: Path,
        options: ContextGatheringOptions | None = None,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.options = options or ContextGatheringOptions()
        self._structure_analyzer = TreeSitterProjectAnalyzer(self.repo_root)
        self._rg_available = self._check_command("rg")
        self._git_available = self._check_command("git")

        # NEW: Initialize RepoMap if enabled
        if self.options.use_repo_map:
            try:
                self.repo_map = RepoMapManager.get_instance(self.repo_root)
            except Exception as e:
                LOGGER.debug("RepoMap initialization failed, falling back: %s", e)
                self.repo_map = None
        else:
            self.repo_map = None

    def gather_contexts(
        self,
        files: Iterable[str],
        task_description: str | None = None,
        keywords: list[str] | None = None,
        chat_files: list[Path] | None = None,  # NEW: For PageRank personalization
    ) -> list[FileContext]:
        """Load requested files and optionally augment with keyword matches or RepoMap ranking."""
        requested = {self._normalize_rel_path(path) for path in files}
        contexts: list[FileContext] = []
        loaded_paths = set()

        # Load explicitly requested files first
        for rel_path in requested:
            context = self._load_file_context(rel_path, "explicitly_requested", 1.0)
            if context:
                contexts.append(context)
                loaded_paths.add(rel_path)

        if self.options.include_related_files:
            discovered = []

            # Try keyword-based discovery first (higher priority)
            if keywords:
                discovered.extend(
                    self._discover_related_files(loaded_paths, task_description, keywords)
                )

            # Then add RepoMap ranking if available (fills remaining slots)
            if self.repo_map and self.options.use_repo_map:
                # Update loaded_paths with discovered files
                current_loaded = loaded_paths.copy()
                current_loaded.update(rel_path for rel_path, _, _ in discovered)

                repo_map_discovered = self._discover_via_repo_map(
                    current_loaded, task_description, chat_files
                )
                discovered.extend(repo_map_discovered)

            for rel_path, reason, score in discovered:
                if rel_path in loaded_paths:
                    continue
                context = self._load_file_context(rel_path, reason, score)
                if context:
                    contexts.append(context)
                    loaded_paths.add(rel_path)

        if self.options.include_structure_summary:
            summary_context = self._build_structure_summary(contexts)
            if summary_context:
                contexts.append(summary_context)

        contexts.sort(key=lambda ctx: ctx.relevance_score, reverse=True)
        return self._apply_size_limits(contexts)

    def search_files(self, pattern: str, file_types: Sequence[str] | None = None) -> list[str]:
        """Search for files containing a pattern using rg or git grep."""
        if self._rg_available:
            return self._rg_search(pattern, file_types)
        if self._git_available:
            return self._git_grep_search(pattern, file_types)
        return self._fallback_search(pattern, file_types)

    def find_symbol_references(self, symbol: str) -> list[tuple[str, int]]:
        """Return files and line numbers containing the symbol."""
        pattern = rf"\b{re.escape(symbol)}\b"
        if self._rg_available:
            return self._rg_symbol_search(pattern)
        if self._git_available:
            return self._git_symbol_search(pattern)
        return self._fallback_symbol_search(pattern)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _discover_via_repo_map(
        self,
        existing: set[str],
        task_description: str | None,
        chat_files: list[Path] | None,
    ) -> list[tuple[str, str, float]]:
        """Discover related files using RepoMap PageRank ranking."""
        if not self.repo_map:
            return []

        discovered: list[tuple[str, str, float]] = []
        limit = max(0, self.options.max_files - len(existing))
        if limit == 0:
            return []

        # Extract mentioned symbols from task description
        mentioned_symbols = set()
        if task_description:
            # Simple pattern matching for potential symbols
            import re

            pattern = r"\b[a-zA-Z_][a-zA-Z0-9_]*\b"
            potential_symbols = re.findall(pattern, task_description)
            # Filter out common English words
            stopwords = {
                "the",
                "this",
                "that",
                "with",
                "from",
                "have",
                "should",
                "could",
                "and",
                "or",
                "but",
            }
            mentioned_symbols = {
                s for s in potential_symbols if s.lower() not in stopwords and len(s) > 2
            }

        # Get ranked files from RepoMap
        mentioned_files = set(existing)

        # Add chat files to mentioned_files to boost their connected files
        if chat_files:
            for chat_file in chat_files:
                rel_path = str(chat_file.relative_to(self.repo_root))
                mentioned_files.add(rel_path)

        # Use the existing RepoMap instance which should already be computed
        ranked_files = self.repo_map.get_ranked_files(
            mentioned_files, mentioned_symbols, limit * 2  # Get extra for filtering
        )

        for file_path, score in ranked_files:
            # file_path is already a string relative path from RepoMap
            if file_path not in existing:
                discovered.append((file_path, f"repomap_rank({score:.2f})", score))
                if len(discovered) >= limit:
                    break

        return discovered

    def _discover_related_files(
        self,
        existing: set[str],
        task_description: str | None,
        keywords: list[str] | None,
    ) -> list[tuple[str, str, float]]:
        if not keywords:
            return []

        discovered: list[tuple[str, str, float]] = []
        seen = set(existing)
        limit = max(0, self.options.max_files - len(existing))
        if limit == 0:
            return []

        for keyword in keywords[: self.options.keyword_match_limit]:
            matches = self.search_files(keyword)
            for rel_path in matches:
                if rel_path in seen:
                    continue
                seen.add(rel_path)
                score = 0.45
                if task_description and keyword.lower() in task_description.lower():
                    score = 0.6
                discovered.append((rel_path, f"keyword_match({keyword})", score))
                if len(discovered) >= limit:
                    return discovered
        return discovered

    def _load_file_context(self, rel_path: str, reason: str, score: float) -> FileContext | None:
        full_path = (self.repo_root / rel_path).resolve()
        if not full_path.exists():
            LOGGER.debug("File not found: %s", rel_path)
            return None
        if not self._should_include_file(full_path):
            return None

        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            LOGGER.warning("Failed to read %s: %s", rel_path, exc)
            return None

        context = FileContext(
            path=full_path,
            content=content,
            relevance_score=score,
            reason=reason,
        )
        outline = self._structure_analyzer.summarize_content(rel_path, content)
        if outline:
            context.structure_outline = outline
            context.symbols = extract_symbols_from_outline(outline)
        return context

    def _build_structure_summary(self, contexts: list[FileContext]) -> FileContext | None:
        outlines = []
        for context in contexts:
            if not context.structure_outline:
                continue
            try:
                rel_path = str(context.path.relative_to(self.repo_root))
            except ValueError:
                continue
            outlines.append((rel_path, context.structure_outline))

        if not outlines:
            return None

        lines: list[str] = [
            "# Project Structure",
            "",
            "Key definitions discovered in the requested files.",
            "",
        ]
        for rel_path, outline in outlines:
            lines.append(f"### {rel_path}")
            lines.extend(outline[: self._structure_analyzer.max_lines_per_file])
            lines.append("")

        content = "\n".join(lines).strip()
        synthetic_path = (self.repo_root / "__project_structure__.md").resolve()
        return FileContext(
            path=synthetic_path,
            content=content,
            relevance_score=0.95,
            reason="project_structure_summary",
        )

    def _apply_size_limits(self, contexts: list[FileContext]) -> list[FileContext]:
        limited: list[FileContext] = []
        total = 0
        for context in contexts:
            if len(limited) >= self.options.max_files:
                break
            if total + context.size_bytes > self.options.max_total_size:
                if not limited:
                    limited.append(context)
                break
            limited.append(context)
            total += context.size_bytes
        return limited

    def _should_include_file(self, file_path: Path) -> bool:
        try:
            rel_path = file_path.relative_to(self.repo_root)
        except ValueError:
            return False

        for pattern in self.options.exclude_patterns:
            if rel_path.match(pattern):
                return False

        try:
            if file_path.stat().st_size > 200_000:
                return False
        except OSError:
            return False

        try:
            with file_path.open("rb") as fh:
                chunk = fh.read(1024)
                if b"\x00" in chunk:
                    return False
        except OSError:
            return False
        return True

    def _normalize_rel_path(self, path: str) -> str:
        full_path = (self.repo_root / path).resolve()
        try:
            return str(full_path.relative_to(self.repo_root))
        except ValueError:
            return path

    def _check_command(self, name: str) -> bool:
        path = os.environ.get("PATH", "")
        for directory in path.split(os.pathsep):
            if not directory:
                continue
            candidate = Path(directory) / name
            if candidate.exists() and os.access(candidate, os.X_OK):
                return True
        return False

    def _rg_search(self, pattern: str, file_types: Sequence[str] | None) -> list[str]:
        cmd = ["rg", "--files-with-matches", "--no-messages", pattern]
        if file_types:
            for file_type in file_types:
                cmd.extend(["-g", f"*.{file_type}"])
        try:
            result = subprocess.run(
                cmd,
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if result.stdout:
                return [line.strip() for line in result.stdout.splitlines() if line.strip()]
        except Exception as exc:
            LOGGER.debug("rg search failed for %s: %s", pattern, exc)
        return []

    def _git_grep_search(self, pattern: str, file_types: Sequence[str] | None) -> list[str]:
        cmd = ["git", "grep", "-l", pattern]
        if file_types:
            for file_type in file_types:
                cmd.extend(["--", f"*.{file_type}"])
        try:
            result = subprocess.run(
                cmd,
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if result.stdout:
                return [line.strip() for line in result.stdout.splitlines() if line.strip()]
        except Exception as exc:
            LOGGER.debug("git grep failed for %s: %s", pattern, exc)
        return []

    def _fallback_search(self, pattern: str, file_types: Sequence[str] | None) -> list[str]:
        try:
            regex = re.compile(pattern)
        except re.error:
            return []

        matches: list[str] = []
        suffixes = {f".{ft}" for ft in file_types} if file_types else None
        for path in self.repo_root.rglob("*"):
            if not path.is_file():
                continue
            if suffixes and path.suffix not in suffixes:
                continue
            try:
                if regex.search(path.read_text(encoding="utf-8", errors="ignore")):
                    matches.append(str(path.relative_to(self.repo_root)))
            except OSError:
                continue
        return matches

    def _rg_symbol_search(self, pattern: str) -> list[tuple[str, int]]:
        cmd = ["rg", "--line-number", "--no-messages", pattern]
        try:
            result = subprocess.run(
                cmd,
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            hits: list[tuple[str, int]] = []
            for line in result.stdout.splitlines():
                parts = line.split(":", 2)
                if len(parts) >= 2 and parts[1].isdigit():
                    hits.append((parts[0], int(parts[1])))
            return hits
        except Exception as exc:
            LOGGER.debug("rg symbol search failed: %s", exc)
            return []

    def _git_symbol_search(self, pattern: str) -> list[tuple[str, int]]:
        cmd = ["git", "grep", "-n", pattern]
        try:
            result = subprocess.run(
                cmd,
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            hits: list[tuple[str, int]] = []
            for line in result.stdout.splitlines():
                parts = line.split(":", 2)
                if len(parts) >= 2 and parts[1].isdigit():
                    hits.append((parts[0], int(parts[1])))
            return hits
        except Exception as exc:
            LOGGER.debug("git symbol search failed: %s", exc)
            return []

    def _fallback_symbol_search(self, pattern: str) -> list[tuple[str, int]]:
        try:
            regex = re.compile(pattern)
        except re.error:
            return []

        hits: list[tuple[str, int]] = []
        for path in self.repo_root.rglob("*"):
            if not path.is_file():
                continue
            try:
                for line_no, line in enumerate(
                    path.read_text(encoding="utf-8", errors="ignore").splitlines(),
                    1,
                ):
                    if regex.search(line):
                        rel = str(path.relative_to(self.repo_root))
                        hits.append((rel, line_no))
                        break
            except OSError:
                continue
        return hits


def gather_file_contexts(
    repo_root: Path,
    files: Iterable[str],
    *,
    task_description: str | None = None,
    keywords: list[str] | None = None,
    options: ContextGatheringOptions | None = None,
) -> list[FileContext]:
    """Convenience wrapper mirroring the historical public API."""

    gatherer = ContextGatherer(repo_root, options)
    return gatherer.gather_contexts(
        files,
        task_description=task_description,
        keywords=keywords,
    )
