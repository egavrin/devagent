"""Dynamic context tracking for RepoMap that adapts across execution steps."""

import re
from pathlib import Path
from typing import Set, List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DynamicContextTracker:
    """Tracks evolving context across ReAct steps to enable adaptive RepoMap."""

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.mentioned_files: Set[str] = set()
        self.mentioned_symbols: Set[str] = set()
        self.step_count = 0
        self.last_refresh_context_hash: Optional[int] = None

    def update_from_step(self, step_record) -> None:
        """Extract context from a step's action and observation.

        Args:
            step_record: StepRecord from the ReAct loop
        """
        self.step_count += 1

        # Extract from action (what the LLM requested)
        if hasattr(step_record, 'action') and step_record.action:
            action = step_record.action

            # Check tool name and parameters
            if hasattr(action, 'tool'):
                tool_name = action.tool

                # Extract files from read/edit/write tools
                if tool_name in ['read', 'edit', 'write']:
                    if hasattr(action, 'parameters'):
                        params = action.parameters
                        if isinstance(params, dict):
                            file_path = params.get('file_path') or params.get('path')
                            if file_path:
                                self._add_file(file_path)

                # Extract patterns from grep/find tools
                elif tool_name in ['grep', 'find']:
                    if hasattr(action, 'parameters'):
                        params = action.parameters
                        if isinstance(params, dict):
                            pattern = params.get('pattern') or params.get('query')
                            if pattern:
                                self._extract_symbols_from_text(pattern)

        # Extract from observation (tool results)
        if hasattr(step_record, 'observation') and step_record.observation:
            obs = step_record.observation

            # Extract from outcome text
            if hasattr(obs, 'outcome') and obs.outcome:
                self._extract_from_text(obs.outcome)

            # Extract from display message
            if hasattr(obs, 'display_message') and obs.display_message:
                self._extract_from_text(obs.display_message)

            # Extract from artifacts (files created/modified)
            if hasattr(obs, 'artifacts') and obs.artifacts:
                for artifact in obs.artifacts:
                    self._add_file(str(artifact))

    def update_from_text(self, text: str) -> None:
        """Extract files and symbols from arbitrary text (e.g., LLM responses).

        Args:
            text: Text to analyze for file and symbol mentions
        """
        self._extract_from_text(text)

    def _extract_from_text(self, text: str) -> None:
        """Internal method to extract both files and symbols from text."""
        if not text:
            return

        # Extract file mentions
        # Pattern 1: File paths with extensions
        file_pattern = r'[\w\-./]+\.\w+'
        for match in re.findall(file_pattern, text):
            if not match.startswith('http') and '.' in match:
                self._add_file(match)

        # Pattern 2: Quoted paths
        quoted_pattern = r'["\']([^"\']+\.[a-zA-Z]+)["\']'
        for match in re.findall(quoted_pattern, text):
            self._add_file(match)

        # Extract symbols
        self._extract_symbols_from_text(text)

    def _extract_symbols_from_text(self, text: str) -> None:
        """Extract programming symbols from text."""
        # CamelCase identifiers
        camel_pattern = r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b'
        for match in re.findall(camel_pattern, text):
            self.mentioned_symbols.add(match)

        # snake_case identifiers
        snake_pattern = r'\b[a-z]+(?:_[a-z]+)+\b'
        for match in re.findall(snake_pattern, text):
            # Filter out common words
            if len(match) > 8 and match not in {'how_many', 'what_files'}:
                self.mentioned_symbols.add(match)

        # CONSTANT_CASE
        const_pattern = r'\b[A-Z]+(?:_[A-Z]+)+\b'
        for match in re.findall(const_pattern, text):
            self.mentioned_symbols.add(match)

    def _add_file(self, file_path: str) -> None:
        """Add a file to the mentioned set, normalizing the path."""
        try:
            # Try to resolve relative to workspace
            if not file_path.startswith('/'):
                full_path = self.workspace / file_path
                if full_path.exists():
                    # Store as relative path
                    rel_path = str(full_path.relative_to(self.workspace))
                    self.mentioned_files.add(rel_path)
                    return

            # Otherwise just store as-is
            self.mentioned_files.add(file_path)
        except Exception as e:
            logger.debug(f"Could not normalize file path {file_path}: {e}")
            # Store as-is if normalization fails
            self.mentioned_files.add(file_path)

    def get_context_summary(self) -> Dict[str, Any]:
        """Get a summary of the current context.

        Returns:
            Dict with files, symbols, and step count
        """
        return {
            'files': list(self.mentioned_files),
            'symbols': list(self.mentioned_symbols),
            'step_count': self.step_count,
            'total_mentions': len(self.mentioned_files) + len(self.mentioned_symbols)
        }

    def should_refresh_repomap(self) -> bool:
        """Determine if RepoMap should be refreshed.

        Returns:
            True if significant new context has been accumulated
        """
        # Refresh every 2-3 steps, or when significant context accumulated
        if not (self.step_count > 0 and
                (self.step_count % 2 == 0 or len(self.mentioned_files) > 3 or len(self.mentioned_symbols) > 5)):
            return False

        # Check if context has actually changed since last refresh
        current_hash = hash(frozenset(self.mentioned_files) | frozenset(self.mentioned_symbols))
        if self.last_refresh_context_hash == current_hash:
            # Context hasn't changed - skip refresh to avoid redundancy
            return False

        # Context has changed - update hash and refresh
        self.last_refresh_context_hash = current_hash
        return True

    def clear(self) -> None:
        """Clear all tracked context."""
        self.mentioned_files.clear()
        self.mentioned_symbols.clear()
        self.step_count = 0
