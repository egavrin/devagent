"""Automatic context enhancement using RepoMap for all queries."""

import re
from pathlib import Path
from typing import Set, List, Optional, Tuple
import logging

from ai_dev_agent.core.repo_map import RepoMapManager
from ai_dev_agent.core.utils.config import Settings

logger = logging.getLogger(__name__)


class ContextEnhancer:
    """Automatically enhances queries with RepoMap context, like Aider does."""

    FILE_MENTION_LIMIT = 40
    DIRECTORY_MENTION_LIMIT = 8

    def __init__(self, workspace: Optional[Path] = None, settings: Optional[Settings] = None):
        self.workspace = workspace or Path.cwd()
        self.settings = settings or Settings()
        self._repo_map = None
        self._initialized = False

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

    def extract_symbols_and_files(self, text: str) -> Tuple[Set[str], Set[str]]:
        """Extract symbols and file mentions from text."""
        symbols: Set[str] = set()
        files_in_order: List[str] = []
        files_seen: Set[str] = set()
        directory_mentions = 0

        # Common English words to exclude
        stop_words = {'what', 'where', 'when', 'how', 'why', 'who', 'which', 'the',
                      'find', 'files', 'related', 'implement', 'implements', 'for',
                      'all', 'about', 'with', 'from', 'that', 'this', 'does', 'have',
                      'many', 'lines', 'line', 'file'}

        # Common file extensions
        FILE_EXTENSIONS = {
            'py', 'js', 'ts', 'tsx', 'jsx', 'cpp', 'cc', 'cxx', 'c', 'h', 'hpp',
            'java', 'go', 'rs', 'rb', 'php', 'cs', 'swift', 'kt', 'scala', 'ets',
            'yaml', 'yml', 'json', 'xml', 'md', 'txt', 'sh', 'bash'
        }

        # Extract file paths (existing patterns)
        # Matches: path/to/file.ext, ./file.ext, ../path/file.ext
        potential_files = re.findall(r'[\w./\-]+\.\w+', text)
        for pf in potential_files:
            if '.' in pf and not pf.startswith('http'):
                if pf not in files_seen:
                    files_seen.add(pf)
                    files_in_order.append(pf)

        # NEW: Extract bare filenames (e.g., "commands.py", "helpers.h")
        # This catches filenames mentioned without paths
        words = re.findall(r'\b[\w\-]+\b', text)
        for word in words:
            if '.' in word:
                parts = word.split('.')
                if len(parts) == 2 and parts[1].lower() in FILE_EXTENSIONS:
                    if word not in files_seen:
                        files_seen.add(word)
                        files_in_order.append(word)

        # NEW: Match against actual repo files (most powerful)
        if self._initialized and self.repo_map.context.files:
            text_lower = text.lower()

            for file_path in self.repo_map.context.files.keys():
                file_name = Path(file_path).name.lower()

                # Check if exact filename is mentioned
                if file_name in text_lower:
                    if file_path not in files_seen:
                        files_seen.add(file_path)
                        files_in_order.append(file_path)
                    if self.settings.repomap_debug_stdout:
                        logger.debug(f"Matched repo file: {file_path} (from filename: {file_name})")
                    continue

                # Check if stem (without extension) is mentioned
                stem = Path(file_path).stem.lower()
                if len(stem) > 3 and stem in text_lower:
                    # Only if stem appears as a word boundary
                    if re.search(r'\b' + re.escape(stem) + r'\b', text_lower):
                        if file_path not in files_seen:
                            files_seen.add(file_path)
                            files_in_order.append(file_path)
                        if self.settings.repomap_debug_stdout:
                            logger.debug(f"Matched repo file: {file_path} (from stem: {stem})")

                # NEW: Check if directory name is mentioned (e.g., "bytecode_optimizer")
                # This helps with queries like "files in bytecode_optimizer"
                parts = Path(file_path).parts
                for part in parts:
                    part_lower = part.lower()
                    if len(part_lower) > 6 and part_lower in text_lower:
                        if re.search(r'\b' + re.escape(part_lower) + r'\b', text_lower):
                            # Add this as a "directory mention" - will boost all files in that dir
                            if part not in files_seen and directory_mentions < self.DIRECTORY_MENTION_LIMIT:
                                files_seen.add(part)
                                files_in_order.append(part)
                                directory_mentions += 1
                            if self.settings.repomap_debug_stdout:
                                logger.debug(f"Matched directory: {part} (matches files in {file_path})")
                            break

        # Extract CamelCase and snake_case identifiers
        # CamelCase (at least 2 capital letters or mixed case)
        camel_matches = re.findall(r'\b[A-Z][a-z]*[A-Z][A-Za-z]*\b', text)
        symbols.update(camel_matches)

        # PascalCase (Capital followed by lowercase)
        pascal_matches = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b', text)
        # Filter out common English words
        symbols.update(w for w in pascal_matches if w.lower() not in stop_words)

        # snake_case
        symbols.update(re.findall(r'\b[a-z]+(?:_[a-z]+)+\b', text))
        # CONSTANT_CASE
        symbols.update(re.findall(r'\b[A-Z]+(?:_[A-Z]+)+\b', text))

        # Single uppercase words that are likely class names (but not stop words)
        single_caps = re.findall(r'\b[A-Z][a-z]{2,}\b', text)
        symbols.update(w for w in single_caps if w.lower() not in stop_words)

        prioritized_files: List[str]
        if self._initialized and self.repo_map.context.files:
            repo_files: List[str] = []
            extra_entries: List[str] = []
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
            prioritized_files = prioritized_files[:self.FILE_MENTION_LIMIT]

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

            # Get relevant files from RepoMap
            relevant_files = self.repo_map.get_ranked_files(
                mentioned_files=mentioned_files,
                mentioned_symbols=symbols,
                max_files=max_files
            )

            if not relevant_files:
                return query

            # Build context string
            context_lines = [
                "\n[Automatic Context from RepoMap]",
                f"Found {len(relevant_files)} relevant files (ranked by importance):"
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

            # Add the context to the query
            enhanced = query + "\n" + "\n".join(context_lines)

            # Add a hint to the LLM
            enhanced += "\n\n[Note: The above files were automatically identified as relevant. You can read them directly without searching.]"

            return enhanced

        except Exception as e:
            logger.warning(f"Failed to enhance query with RepoMap context: {e}")
            return query

    def get_context_for_files(self, files: List[str], symbols: Set[str] = None) -> List[str]:
        """Get additional context files for the given files (like Aider's approach)."""
        try:
            # Use the files as "chat files" for personalization
            mentioned_files = set(files)

            # Get ranked related files
            related = self.repo_map.get_ranked_files(
                mentioned_files=mentioned_files,
                mentioned_symbols=symbols or set(),
                max_files=10
            )

            # Return just the file paths
            return [f for f, _ in related if f not in mentioned_files]

        except Exception as e:
            logger.warning(f"Failed to get context files: {e}")
            return []

    def get_repomap_messages(self, query: str, max_files: int = 15,
                            additional_files: Optional[Set[str]] = None,
                            additional_symbols: Optional[Set[str]] = None) -> Tuple[str, Optional[List[dict]]]:
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

            if not symbols and not mentioned_files:
                return query, None

            # Get relevant files from RepoMap
            relevant_files = self.repo_map.get_ranked_files(
                mentioned_files=mentioned_files,
                mentioned_symbols=symbols,
                max_files=max_files
            )

            if not relevant_files:
                return query, None

            # Build context string (Aider's style - more direct)
            context_lines = [
                "Here are the relevant files in the git repository:",
                ""
            ]

            # Group by score ranges
            high_relevance = []
            medium_relevance = []

            for file_path, score in relevant_files:
                file_info = self.repo_map.context.files.get(file_path)
                lang = file_info.language if file_info else "unknown"

                # Show full relative path
                entry = f"  • {file_path}"

                if score > 10:
                    high_relevance.append((entry, score))
                elif score > 3:
                    medium_relevance.append((entry, score))

            # Show high relevance files prominently
            if high_relevance:
                for entry, score in high_relevance:
                    context_lines.append(f"{entry}")

            # Add medium relevance if there's space
            if medium_relevance and len(high_relevance) < 10:
                context_lines.append("")
                for entry, score in medium_relevance[:5]:
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
                for line in preview.split('\n'):
                    logger.debug(f"  {line}")

            # Return as conversation messages (Aider's approach)
            # Make the assistant response more actionable
            if len(high_relevance) <= 3:
                assistant_msg = f"I can see the relevant files. I'll read them directly to answer your question."
            else:
                assistant_msg = f"I can see {len(high_relevance)} relevant files. I'll focus on the most relevant ones."

            repomap_messages = [
                {
                    "role": "user",
                    "content": repomap_content
                },
                {
                    "role": "assistant",
                    "content": assistant_msg
                }
            ]

            return query, repomap_messages

        except Exception as e:
            logger.warning(f"Failed to get RepoMap messages: {e}")
            if self.settings.repomap_debug_stdout:
                logger.debug(f"Failed to get RepoMap messages: {e}")
            return query, None


# Singleton instance
_context_enhancer = None


def get_context_enhancer(workspace: Optional[Path] = None, settings: Optional[Settings] = None) -> ContextEnhancer:
    """Get or create the global context enhancer."""
    global _context_enhancer
    if _context_enhancer is None:
        _context_enhancer = ContextEnhancer(workspace, settings)
    return _context_enhancer


def enhance_query(query: str, workspace: Optional[Path] = None) -> Tuple[str, Optional[List[dict]]]:
    """Convenience function to enhance a query with RepoMap context.

    Returns:
        Tuple of (original_query, repomap_messages)
        repomap_messages is None if no context, or a list of message dicts
    """
    enhancer = get_context_enhancer(workspace)
    return enhancer.get_repomap_messages(query)
