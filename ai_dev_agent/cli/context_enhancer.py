"""Automatic context enhancement using RepoMap and Memory Bank for all queries."""

import logging
import re
from pathlib import Path
from typing import Any, Iterable, Optional

from ai_dev_agent.cli.memory_provider import MemoryProvider
from ai_dev_agent.core.repo_map import RepoMapManager
from ai_dev_agent.core.utils.config import Settings

logger = logging.getLogger(__name__)

# Playbook and dynamic instructions have been removed from the codebase


class ContextEnhancer:
    """Automatically enhances queries with RepoMap context, like Aider does."""

    FILE_MENTION_LIMIT = 40
    DIRECTORY_MENTION_LIMIT = 8

    def __init__(self, workspace: Optional[Path] = None, settings: Optional[Settings] = None):
        self.workspace = workspace or Path.cwd()
        self.settings = settings or Settings()
        self._repo_map = None
        self._initialized = False

        # Initialize memory provider
        enable_memory = getattr(settings, "enable_memory_bank", True) if settings else True
        self._memory_provider = MemoryProvider(self.workspace, enable_memory)

    @property
    def repo_map(self):
        """Lazy load RepoMap."""
        if self._repo_map is None:
            self._repo_map = RepoMapManager.get_instance(self.workspace)
            # Note: RepoMapManager.get_instance already scans on first access
            # so we just need to check if files are populated
            if self._repo_map.context.files:
                # RepoMap is ready (either from our scan or RepoMapManager's auto-scan)
                self._initialized = True
            elif not self._initialized:
                # If still not populated, scan explicitly
                logger.info("Initializing RepoMap for context enhancement...")
                self._repo_map.scan_repository()
                self._initialized = True
        return self._repo_map

    def _get_important_files(self, all_files: list[str], max_files: int) -> list[tuple[str, float]]:
        """Get important files using Aider-style prioritization.

        Returns files with scores based on their importance in typical codebases.
        """
        # Comprehensive list of important file patterns (inspired by Aider)
        important_patterns = {
            # Documentation (highest priority)
            "README.md": 20.0,
            "readme.md": 20.0,
            "README.rst": 20.0,
            "README.txt": 20.0,
            "README": 18.0,
            "CONTRIBUTING.md": 15.0,
            "CHANGELOG.md": 12.0,
            "CHANGES.md": 12.0,
            "AUTHORS.md": 10.0,
            "LICENSE": 15.0,
            "LICENSE.md": 15.0,
            "LICENSE.txt": 15.0,
            # Python configuration
            "pyproject.toml": 18.0,
            "setup.py": 18.0,
            "setup.cfg": 16.0,
            "requirements.txt": 15.0,
            "requirements.in": 14.0,
            "Pipfile": 14.0,
            "poetry.lock": 12.0,
            "tox.ini": 10.0,
            "pytest.ini": 10.0,
            ".coveragerc": 8.0,
            # JavaScript/TypeScript configuration
            "package.json": 18.0,
            "tsconfig.json": 16.0,
            "webpack.config.js": 14.0,
            "babel.config.js": 12.0,
            "jest.config.js": 10.0,
            ".eslintrc.js": 8.0,
            "vite.config.js": 12.0,
            "next.config.js": 12.0,
            "nuxt.config.js": 12.0,
            # Build files
            "Makefile": 16.0,
            "makefile": 16.0,
            "CMakeLists.txt": 16.0,
            "build.gradle": 14.0,
            "pom.xml": 14.0,
            "Cargo.toml": 14.0,
            "go.mod": 14.0,
            "go.sum": 12.0,
            # Docker/Containers
            "Dockerfile": 16.0,
            "docker-compose.yml": 14.0,
            "docker-compose.yaml": 14.0,
            ".dockerignore": 8.0,
            "kubernetes.yml": 12.0,
            # CI/CD
            ".github/workflows/ci.yml": 12.0,
            ".github/workflows/main.yml": 12.0,
            ".gitlab-ci.yml": 12.0,
            ".travis.yml": 10.0,
            "azure-pipelines.yml": 10.0,
            "Jenkinsfile": 10.0,
            ".circleci/config.yml": 10.0,
            # Entry points (lower than config but still important)
            "main.py": 10.0,
            "app.py": 10.0,
            "server.py": 10.0,
            "index.py": 9.0,
            "index.js": 10.0,
            "index.ts": 10.0,
            "main.js": 10.0,
            "main.ts": 10.0,
            "app.js": 9.0,
            "app.ts": 9.0,
            "server.js": 9.0,
            "server.ts": 9.0,
            "cli.py": 8.0,
            "__main__.py": 8.0,
            "__init__.py": 6.0,
            # Configuration files
            ".env.example": 8.0,
            "config.py": 8.0,
            "settings.py": 8.0,
            "config.json": 7.0,
            "config.yaml": 7.0,
            "config.yml": 7.0,
            ".editorconfig": 5.0,
            ".gitignore": 5.0,
            ".gitattributes": 5.0,
        }

        # Also check for patterns in paths (not just exact names)
        path_patterns = {
            "src/index": 9.0,
            "src/main": 9.0,
            "src/app": 8.0,
            "lib/index": 9.0,
            "lib/main": 9.0,
            "cmd/main": 9.0,  # Go entry points
            "bin/": 7.0,  # Binary/script directories
            "scripts/": 6.0,
        }

        important_files = []
        seen_files = set()

        # First pass: exact filename matches
        for file_path in all_files:
            file_name = Path(file_path).name
            if file_name in important_patterns and file_path not in seen_files:
                score = important_patterns[file_name]
                important_files.append((file_path, score))
                seen_files.add(file_path)
                if len(important_files) >= max_files:
                    break

        # Second pass: path pattern matches
        if len(important_files) < max_files:
            for file_path in all_files:
                if file_path in seen_files:
                    continue
                for pattern, score in path_patterns.items():
                    if pattern in file_path:
                        important_files.append((file_path, score))
                        seen_files.add(file_path)
                        break
                if len(important_files) >= max_files:
                    break

        # Third pass: implementation files (non-test, non-generated)
        if len(important_files) < max_files:
            for file_path in all_files:
                if file_path in seen_files:
                    continue
                # Skip test files, generated files, vendored code
                lower_path = file_path.lower()
                if any(
                    skip in lower_path
                    for skip in [
                        "test",
                        "spec",
                        "vendor",
                        "node_modules",
                        ".min.",
                        "dist/",
                        "build/",
                        "__pycache__",
                        ".egg-info",
                    ]
                ):
                    continue
                # Add with lower score
                important_files.append((file_path, 3.0))
                seen_files.add(file_path)
                if len(important_files) >= max_files:
                    break

        # Sort by score (highest first) and return
        important_files.sort(key=lambda x: x[1], reverse=True)
        return important_files[:max_files]

    def extract_symbols_and_files(self, text: str) -> tuple[set[str], set[str]]:
        """Extract symbols and file mentions from text."""
        symbols: set[str] = set()
        files_in_order: list[str] = []
        files_seen: set[str] = set()
        directory_mentions = 0

        # Common English words to exclude
        stop_words = {
            "what",
            "where",
            "when",
            "how",
            "why",
            "who",
            "which",
            "the",
            "find",
            "files",
            "related",
            "implement",
            "implements",
            "for",
            "all",
            "about",
            "with",
            "from",
            "that",
            "this",
            "does",
            "have",
            "many",
            "lines",
            "line",
            "file",
            "any",
            "type",
            "types",
            "check",
            "case",
            "cases",
            "could",
            "please",
            "tell",
            "there",
            "some",
            "more",
            "will",
            "emit",
            "emitted",
            "union",
            "generics",
            "without",
            "such",
            "fragile",
            "constructs",
            "variants",
            "cover",
            "covers",
            "these",
            "those",
            "like",
        }

        generic_symbols = {
            "any",
            "type",
            "types",
            "check",
            "checking",
            "case",
            "cases",
            "generic",
            "generics",
            "union",
            "optional",
            "undefined",
            "please",
            "could",
            "would",
            "should",
            "tell",
            "these",
            "those",
            "this",
            "that",
            "where",
            "when",
            "what",
            "with",
            "emit",
            "emitted",
            "emitting",
            "cover",
            "covers",
            "fragile",
            "construct",
            "constructs",
            "variants",
        }

        def _should_keep_symbol(token: str) -> bool:
            if not token:
                return False
            lowered = token.lower()
            if lowered in stop_words or lowered in generic_symbols:
                return False
            if lowered.isnumeric():
                return False
            return not (len(lowered) <= 2 and lowered not in {"c", "go", "js", "ts", "py"})

        # Common file extensions
        file_extensions = {
            "py",
            "js",
            "ts",
            "tsx",
            "jsx",
            "cpp",
            "cc",
            "cxx",
            "c",
            "h",
            "hpp",
            "java",
            "go",
            "rs",
            "rb",
            "php",
            "cs",
            "swift",
            "kt",
            "scala",
            "ets",
            "yaml",
            "yml",
            "json",
            "xml",
            "md",
            "txt",
            "sh",
            "bash",
        }

        # Extract file paths (existing patterns)
        # Matches: path/to/file.ext, ./file.ext, ../path/file.ext
        potential_files = re.findall(r"[\w./\-]+\.\w+", text)
        for pf in potential_files:
            if "." in pf and not pf.startswith("http") and pf not in files_seen:
                files_seen.add(pf)
                files_in_order.append(pf)

        # NEW: Extract bare filenames (e.g., "main.py", "helpers.h")
        # This catches filenames mentioned without paths
        words = re.findall(r"\b[\w\-]+\b", text)
        for word in words:
            if "." in word:
                parts = word.split(".")
                if (
                    len(parts) == 2
                    and parts[1].lower() in file_extensions
                    and word not in files_seen
                ):
                    files_seen.add(word)
                    files_in_order.append(word)

        # NEW: Match against actual repo files (most powerful)
        if self._initialized and self.repo_map.context.files:
            text_lower = text.lower()

            # Iterate with items() to get FileInfo for potential caching
            for file_path, _file_info in self.repo_map.context.files.items():
                # Create Path object once and reuse (aider's approach)
                path_obj = Path(file_path)
                file_name = path_obj.name.lower()

                # Check if exact filename is mentioned
                if file_name in text_lower:
                    if file_path not in files_seen:
                        files_seen.add(file_path)
                        files_in_order.append(file_path)
                    if self.settings.repomap_debug_stdout:
                        logger.debug(f"Matched repo file: {file_path} (from filename: {file_name})")
                    continue

                # Check if stem (without extension) is mentioned
                stem = path_obj.stem.lower()  # Reuse path_obj
                if (
                    len(stem) > 3
                    and stem in text_lower
                    and re.search(r"\b" + re.escape(stem) + r"\b", text_lower)
                ) and file_path not in files_seen:
                    files_seen.add(file_path)
                    files_in_order.append(file_path)
                    if self.settings.repomap_debug_stdout:
                        logger.debug(f"Matched repo file: {file_path} (from stem: {stem})")

                # NEW: Check if directory name is mentioned (e.g., "bytecode_optimizer")
                # This helps with queries like "files in bytecode_optimizer"
                parts = path_obj.parts  # Reuse path_obj
                for part in parts:
                    part_lower = part.lower()
                    if (
                        len(part_lower) > 6
                        and part_lower in text_lower
                        and re.search(r"\b" + re.escape(part_lower) + r"\b", text_lower)
                    ):
                        # Add this as a "directory mention" - will boost all files in that dir
                        if (
                            part not in files_seen
                            and directory_mentions < self.DIRECTORY_MENTION_LIMIT
                        ):
                            files_seen.add(part)
                            files_in_order.append(part)
                            directory_mentions += 1
                            if self.settings.repomap_debug_stdout:
                                logger.debug(
                                    f"Matched directory: {part} (matches files in {file_path})"
                                )
                            break

        # Extract CamelCase and snake_case identifiers
        # Pattern-based extraction filters out English prose automatically
        # CamelCase (at least 2 capital letters or mixed case)
        camel_matches = re.findall(r"\b[A-Z][a-z]*[A-Z][A-Za-z]*\b", text)
        symbols.update(match for match in camel_matches if _should_keep_symbol(match))

        # PascalCase (Capital followed by lowercase)
        pascal_matches = re.findall(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b", text)
        # Filter out common English words
        symbols.update(w for w in pascal_matches if _should_keep_symbol(w))

        # snake_case
        symbols.update(
            token
            for token in re.findall(r"\b[a-z]+(?:_[a-z]+)+\b", text)
            if _should_keep_symbol(token)
        )
        # CONSTANT_CASE
        symbols.update(
            token
            for token in re.findall(r"\b[A-Z]+(?:_[A-Z]+)+\b", text)
            if _should_keep_symbol(token)
        )

        # Single uppercase words that are likely class names (but not stop words)
        single_caps = re.findall(r"\b[A-Z][a-z]{2,}\b", text)
        symbols.update(w for w in single_caps if _should_keep_symbol(w))

        prioritized_files: list[str]
        if self._initialized and self.repo_map.context.files:
            repo_files: list[str] = []
            extra_entries: list[str] = []
            repo_index = self.repo_map.context.files
            for entry in files_in_order:
                if entry in repo_index:
                    repo_files.append(entry)
                else:
                    extra_entries.append(entry)
            prioritized_files = repo_files + extra_entries
        else:
            prioritized_files = files_in_order

        if len(prioritized_files) > self.FILE_MENTION_LIMIT:
            prioritized_files = prioritized_files[: self.FILE_MENTION_LIMIT]

        return symbols, set(prioritized_files)

    def enhance_query_with_context(self, query: str, max_files: int = 15) -> str:
        """Enhance a query with automatic RepoMap context."""
        # Don't enhance if RepoMap isn't available
        if not self.workspace.exists():
            return query

        try:
            # Extract symbols and files from query
            symbols, mentioned_files = self.extract_symbols_and_files(query)

            if not symbols and not mentioned_files:
                return query

            repo_files = self.repo_map.context.files
            repo_file_names = {
                info.file_name.lower() for info in repo_files.values() if info.file_name
            }
            repo_file_stems = {
                info.file_stem.lower() for info in repo_files.values() if info.file_stem
            }

            # Get relevant files from RepoMap
            relevant_files = self.repo_map.get_ranked_files(
                mentioned_files=mentioned_files, mentioned_symbols=symbols, max_files=max_files
            )

            unmatched_mentions: list[str] = []
            for candidate in mentioned_files:
                candidate.lower()
                if candidate in repo_files:
                    continue
                name_lower = Path(candidate).name.lower()
                stem_lower = Path(candidate).stem.lower()
                if name_lower in repo_file_names or (stem_lower and stem_lower in repo_file_stems):
                    continue
                unmatched_mentions.append(candidate)

            missing_symbols = [
                sym for sym in symbols if sym not in self.repo_map.context.symbol_index
            ]

            if not relevant_files:
                notice_lines = []
                if unmatched_mentions:
                    notice_lines.append(
                        "[RepoMap Notice] No files in this workspace match: "
                        + ", ".join(sorted(unmatched_mentions)[:8])
                    )
                if missing_symbols:
                    notice_lines.append(
                        "[RepoMap Notice] No symbols found for: "
                        + ", ".join(sorted(missing_symbols)[:8])
                    )
                if notice_lines:
                    return query + "\n" + "\n".join(notice_lines)
                return query

            # Build context string
            context_lines = [
                "\n[Automatic Context from RepoMap]",
                f"Found {len(relevant_files)} relevant files (ranked by importance):",
            ]

            # Debug output
            if self.settings.repomap_debug_stdout:
                logger.debug(f"RepoMap found {len(relevant_files)} files for symbols: {symbols}")
                for file_path, score in relevant_files[:5]:
                    logger.debug(f"  - {file_path}: score={score:.2f}")

            # Group by score ranges for better readability
            high_relevance = []
            medium_relevance = []
            low_relevance = []

            for file_path, score in relevant_files:
                file_info = self.repo_map.context.files.get(file_path)
                lang = file_info.language if file_info else "unknown"

                # Show full relative path for clarity
                entry = f"  • {file_path} [{lang}]"

                if score > 10:
                    high_relevance.append(entry + f" (score: {score:.1f})")
                elif score > 5:
                    medium_relevance.append(entry)
                else:
                    low_relevance.append(entry)

            if high_relevance:
                context_lines.append("\nHigh relevance:")
                context_lines.extend(high_relevance)

            if medium_relevance:
                context_lines.append("\nMedium relevance:")
                context_lines.extend(medium_relevance[:5])

            if low_relevance and len(high_relevance) < 5:
                context_lines.append("\nOther relevant files:")
                context_lines.extend(low_relevance[:3])

            if unmatched_mentions or missing_symbols:
                context_lines.append("\nRepoMap diagnostic:")
                if unmatched_mentions:
                    context_lines.append(
                        "  • No files matched: " + ", ".join(sorted(unmatched_mentions)[:8])
                    )
                if missing_symbols:
                    context_lines.append(
                        "  • Unknown symbols: " + ", ".join(sorted(missing_symbols)[:8])
                    )

            # Add the context to the query
            enhanced = query + "\n" + "\n".join(context_lines)

            # Add a hint to the LLM
            enhanced += "\n\n[Note: The above files were automatically identified as relevant. You can read them directly without searching.]"

            return enhanced

        except Exception as e:
            logger.warning(f"Failed to enhance query with RepoMap context: {e}")
            return query

    def get_context_for_files(
        self, files: list[str], symbols: Optional[set[str]] = None
    ) -> list[str]:
        """Get additional context files for the given files (like Aider's approach)."""
        try:
            # Use the files as "chat files" for personalization
            mentioned_files = set(files)

            # Get ranked related files
            related = self.repo_map.get_ranked_files(
                mentioned_files=mentioned_files, mentioned_symbols=symbols or set(), max_files=10
            )

            # Return just the file paths
            return [f for f, _ in related if f not in mentioned_files]

        except Exception as e:
            logger.warning(f"Failed to get context files: {e}")
            return []

    def get_repomap_messages(
        self,
        query: str,
        max_files: int = 15,
        additional_files: Optional[set[str]] = None,
        additional_symbols: Optional[set[str]] = None,
    ) -> tuple[str, Optional[list[dict]]]:
        """Get RepoMap context as conversation messages (Aider's approach).

        Args:
            query: The user query
            max_files: Maximum files to include
            additional_files: Files discovered in previous steps (for dynamic updates)
            additional_symbols: Symbols discovered in previous steps (for dynamic updates)

        Returns:
            Tuple of (original_query, repomap_messages)
            repomap_messages is a list of user/assistant message dicts, or None if no context
        """
        # Don't enhance if RepoMap isn't available
        if not self.workspace.exists():
            return query, None

        try:
            # Extract symbols and files from query
            symbols, mentioned_files = self.extract_symbols_and_files(query)

            # Merge with additional context from previous steps
            if additional_files:
                mentioned_files.update(additional_files)
            if additional_symbols:
                symbols.update(additional_symbols)

            # Implement Aider-style 3-tier fallback
            fallback_tier = 1
            effective_max_files = max_files  # May be expanded for exploratory queries

            # Tier 1: Standard approach with mentioned files/symbols
            relevant_files = (
                self.repo_map.get_ranked_files(
                    mentioned_files=mentioned_files,
                    mentioned_symbols=symbols,
                    max_files=effective_max_files,
                )
                if (symbols or mentioned_files)
                else []
            )

            # Tier 2: If no results, try with important files boosted
            if not relevant_files and (symbols or mentioned_files):
                fallback_tier = 2
                # Expand budget moderately for broader search
                effective_max_files = min(max_files * 2, 30)  # 2x expansion, cap at 30

                if self.settings.repomap_debug_stdout:
                    logger.debug(
                        f"Tier 1 failed, trying Tier 2 with important files boost (max_files={effective_max_files})"
                    )

                # Get important files to boost their priority
                all_files_list = list(self.repo_map.context.files.keys())
                important = self._get_important_files(all_files_list, effective_max_files // 2)
                important_set = {f for f, _ in important}

                # Combine with mentioned files for personalization
                boosted_files = mentioned_files | important_set

                relevant_files = self.repo_map.get_ranked_files(
                    mentioned_files=boosted_files,  # Include important files
                    mentioned_symbols=symbols,  # Still use symbols for weighting
                    max_files=effective_max_files,
                )

            # Tier 3: Only for TRULY exploratory queries where we have additional context
            # Don't use Tier 3 for initial queries with no context - return None instead
            if not relevant_files and (additional_files or additional_symbols):
                fallback_tier = 3
                # Aider-style: 8x expansion for exploratory queries (no direct matches)
                effective_max_files = min(max_files * 8, 120)  # 8x expansion like Aider, cap at 120
                if self.settings.repomap_debug_stdout:
                    logger.debug(
                        f"Tier 2 failed, trying Tier 3 with pure PageRank (max_files={effective_max_files})"
                    )
                # Get top files by pure PageRank (no personalization)
                all_files = list(self.repo_map.context.files.keys())
                if all_files:
                    # Get PageRank scores for all files

                    # Create a minimal RepoMap call with no hints
                    relevant_files = self.repo_map.get_ranked_files(
                        mentioned_files=set(),  # No file hints
                        mentioned_symbols=set(),  # No symbol hints
                        max_files=effective_max_files,  # Use expanded budget
                    )

                    # If still nothing (shouldn't happen), fall back to important files
                    if not relevant_files:
                        # Get important files first using Aider-style patterns
                        important_files = self._get_important_files(all_files, effective_max_files)
                        relevant_files = important_files

            if not relevant_files:
                # No relevant files found - return None (don't force context)
                return query, None

            # Build context string (Aider's style - more direct)
            if fallback_tier == 3:
                # Exploratory context with many files
                context_lines = [
                    "Here is an overview of the repository structure (showing key files):",
                    "",
                ]
            else:
                context_lines = ["Here are the relevant files in the git repository:", ""]

            # Group by score ranges and display based on tier
            if fallback_tier == 3:
                # For exploratory queries, show all files (no filtering)
                for file_path, _score in relevant_files:
                    entry = f"  • {file_path}"
                    context_lines.append(entry)
            else:
                # For targeted queries, group by relevance
                high_relevance = []
                medium_relevance = []
                low_relevance = []

                for file_path, score in relevant_files:
                    self.repo_map.context.files.get(file_path)

                    # Show full relative path
                    entry = f"  • {file_path}"

                    if score > 10:
                        high_relevance.append((entry, score))
                    elif score > 3:
                        medium_relevance.append((entry, score))
                    else:
                        low_relevance.append((entry, score))

                # Show high relevance files prominently
                if high_relevance:
                    for entry, _score in high_relevance:
                        context_lines.append(f"{entry}")

                # Add medium relevance if there's space
                if medium_relevance and len(high_relevance) < 10:
                    context_lines.append("")
                    for entry, _score in medium_relevance[:10]:
                        context_lines.append(f"{entry}")

                # In Tier 2, also include some low relevance files
                if (
                    fallback_tier == 2
                    and low_relevance
                    and len(high_relevance) + len(medium_relevance) < 20
                ):
                    context_lines.append("")
                    remaining = 20 - len(high_relevance) - len(medium_relevance)
                    for entry, _score in low_relevance[:remaining]:
                        context_lines.append(f"{entry}")

            repomap_content = "\n".join(context_lines)

            # Debug output
            if self.settings.repomap_debug_stdout:
                logger.debug(f"RepoMap found {len(relevant_files)} files for symbols: {symbols}")
                for file_path, score in relevant_files[:5]:
                    logger.debug(f"  - {file_path}: score={score:.2f}")

            # Show what the LLM will see
            if self.settings.repomap_debug_stdout:
                logger.debug("RepoMap message preview:")
                preview = repomap_content[:300] if len(repomap_content) > 300 else repomap_content
                for line in preview.split("\n"):
                    logger.debug(f"  {line}")

            # Return as conversation messages (Aider's approach)
            # Make the assistant response appropriate for the tier
            if fallback_tier == 3:
                # No direct matches - provide exploratory context like Aider
                assistant_msg = (
                    "I can see a high-level view of your repository structure. "
                    "Tell me which files you'd like me to examine for this task, "
                    "or I can search for relevant code based on your query."
                )
            elif len(high_relevance) <= 3:
                assistant_msg = (
                    "I can see the relevant files. I'll read them directly to answer your question."
                )
            else:
                assistant_msg = f"I can see {len(high_relevance)} relevant files. I'll focus on the most relevant ones."

            repomap_messages = [
                {"role": "user", "content": repomap_content},
                {"role": "assistant", "content": assistant_msg},
            ]

            return query, repomap_messages

        except Exception as e:
            logger.warning(f"Failed to get RepoMap messages: {e}")
            if self.settings.repomap_debug_stdout:
                logger.debug(f"Failed to get RepoMap messages: {e}")
            return query, None

    def get_memory_context(
        self, query: str, task_type: Optional[str] = None, limit: int = 5, threshold: float = 0.3
    ) -> tuple[list[dict[str, Any]], Optional[list[str]]]:
        """Retrieve relevant memories for the query - delegates to MemoryProvider."""
        if not self.has_memory_support:
            return [], None

        memories = self._memory_provider.retrieve_relevant_memories(query, task_type, limit)
        if not memories:
            return [], None

        # Format memories into context messages
        memory_content = self._memory_provider.format_memories_for_context(memories)
        if not memory_content:
            return [], None

        memory_messages = [
            {"role": "system", "content": memory_content},
            {
                "role": "assistant",
                "content": f"I've retrieved {len(memories)} relevant memories from past experiences.",
            },
        ]

        # Extract memory IDs if available
        memory_ids = [m.get("id") for m in memories if m.get("id")]
        return memory_messages, memory_ids if memory_ids else None

    def store_memory(
        self, query: str, response: str, task_type: Optional[str] = None, success: bool = True
    ) -> Optional[str]:
        """Store a new memory - delegates to MemoryProvider."""
        metadata = {"workspace": str(self.workspace)}
        return self._memory_provider.store_memory(query, response, task_type, success, metadata)

    # Memory provider helpers -------------------------------------------------

    @property
    def has_memory_support(self) -> bool:
        """Return True when the memory bank is available."""
        return self._memory_provider.has_store

    def distill_and_store_memory(
        self,
        session_id: str,
        messages: Iterable[Any],
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[str]:
        """Distill a session transcript and persist it using the memory provider."""
        return self._memory_provider.distill_and_store_memory(session_id, messages, metadata)

    def track_memory_effectiveness(
        self, memory_ids: Iterable[str], success: bool, feedback: Optional[str] = None
    ) -> None:
        """Update effectiveness metrics for retrieved memories."""
        self._memory_provider.track_memory_effectiveness(memory_ids, success, feedback)

    def record_query_outcome(
        self,
        *,
        session_id: str,
        success: bool,
        tools_used: list[str],
        task_type: str,
        error_type: Optional[str] = None,
        duration_seconds: Optional[float] = None,
    ) -> None:
        """Record lightweight outcome metadata for telemetry."""
        self._memory_provider.record_query_outcome(
            session_id=session_id,
            success=success,
            tools_used=tools_used,
            task_type=task_type,
            error_type=error_type,
            duration_seconds=duration_seconds,
        )

    def collect_memory_statistics(self) -> dict[str, Any]:
        """Expose aggregated memory statistics for diagnostics."""
        return self._memory_provider.collect_statistics()

    # Playbook and dynamic instruction methods have been removed


# Singleton instance
_context_enhancer = None


def get_context_enhancer(
    workspace: Optional[Path] = None, settings: Optional[Settings] = None
) -> ContextEnhancer:
    """Get or create the global context enhancer."""
    global _context_enhancer
    if _context_enhancer is None:
        _context_enhancer = ContextEnhancer(workspace, settings)
    return _context_enhancer


def enhance_query(query: str, workspace: Optional[Path] = None) -> tuple[str, Optional[list[dict]]]:
    """Convenience function to enhance a query with RepoMap context.

    Returns:
        Tuple of (original_query, repomap_messages)
        repomap_messages is None if no context, or a list of message dicts
    """
    enhancer = get_context_enhancer(workspace)
    return enhancer.get_repomap_messages(query)
