"""Context synthesis for agent-oriented system prompts."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import TYPE_CHECKING

from ai_dev_agent.tools import EDIT, READ, RUN


def _extract_modified_paths(diff_text: str) -> set[str]:
    """Extract file paths from SEARCH/REPLACE blocks, apply_patch, or unified diff strings."""
    paths: set[str] = set()
    if not diff_text:
        return paths

    lines = diff_text.splitlines()
    for i, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue

        # SEARCH/REPLACE format: file path on its own line before ```
        # Pattern: "path/to/file.ext" followed by "```" on next line
        if (
            not line.startswith(("```", "<<<", "===", ">>>", "***", "---", "+++"))
            and "." in line
            and "/" in line
            or line.endswith((".py", ".md", ".js", ".ts", ".json", ".txt", ".yaml", ".toml"))
        ):
            # Check if next line starts a code fence (SEARCH/REPLACE format)
            if i + 1 < len(lines) and lines[i + 1].strip().startswith("```"):
                paths.add(line)
                continue

        # Old apply_patch format (still supported for backwards compatibility)
        if (
            line.startswith("*** Update File:")
            or line.startswith("*** Add File:")
            or line.startswith("*** Delete File:")
        ):
            _, _, remainder = line.partition(":")
            candidate = remainder.strip()
            if candidate:
                paths.add(candidate)
            continue

        # Unified diff format
        if line.startswith("--- ") or line.startswith("+++ "):
            candidate = line[4:].strip()
            if candidate == "/dev/null":
                continue
            if candidate.startswith(("a/", "b/")):
                candidate = candidate[2:]
            if candidate:
                paths.add(candidate)

    return paths


if TYPE_CHECKING:
    from ai_dev_agent.providers.llm.base import Message


class ContextSynthesizer:
    """Synthesizes context from previous steps for agent continuity."""

    def __init__(self, max_context_chars: int = 2000):
        self.max_context_chars = max_context_chars

    def synthesize_previous_steps(self, history: list[Message], current_step: int) -> str:
        """Extract and summarize key findings from previous steps.

        Args:
            history: Session history messages
            current_step: Current iteration number
            include_tools: Whether to include tool usage summary

        Returns:
            Bulleted summary of previous findings
        """
        if current_step <= 1 or not history:
            return "This is your first step - no previous findings."

        findings = []
        files_examined = set()
        files_modified = set()
        searches_performed = []
        symbols_found = []
        key_discoveries = []
        errors_encountered = []

        # Analyze history for patterns
        for _i, msg in enumerate(history):
            if msg.role == "assistant" and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    if isinstance(tool_call, dict):
                        tool_name = tool_call.get("function", {}).get("name", "")
                        args = tool_call.get("function", {}).get("arguments", {})

                        # Track file operations
                        if tool_name == READ or "read" in tool_name.lower():
                            if isinstance(args, dict):
                                paths = args.get("paths") or [args.get("file_path")]
                                files_examined.update(p for p in paths if p)

                        # Track file modifications (EDIT tool invocations)
                        elif tool_name == EDIT or "edit" in tool_name.lower():
                            if isinstance(args, dict):
                                # Track path from edit operations
                                path = args.get("path")
                                if path:
                                    files_modified.add(path)
                                diff_or_patch = args.get("patch") or args.get("diff")
                                if isinstance(diff_or_patch, str):
                                    files_modified.update(_extract_modified_paths(diff_or_patch))

                        # Track searches
                        elif "search" in tool_name.lower() or "grep" in tool_name.lower():
                            if isinstance(args, dict):
                                query = args.get("query") or args.get("pattern", "")
                                if query:
                                    searches_performed.append(query[:50])

                        # Track symbol lookups
                        elif tool_name == "symbols":
                            if isinstance(args, dict):
                                name = args.get("name", "")
                                if name:
                                    symbols_found.append(name)

                        # Track code analysis
                        elif "ast" in tool_name.lower():
                            if isinstance(args, dict):
                                target = args.get("path", "")
                                if target:
                                    key_discoveries.append(f"Analyzed AST of {target}")

            # Extract insights from tool responses
            elif msg.role == "tool":
                content = str(msg.content) if msg.content else ""

                # Check for errors
                if "error" in content.lower()[:100] or "failed" in content.lower()[:100]:
                    error_snippet = content[:100].strip()
                    errors_encountered.append(error_snippet)

                # Look for key patterns in responses
                if "found" in content.lower() or "discovered" in content.lower():
                    # Extract first meaningful line
                    lines = content.split("\n")
                    for line in lines[:3]:
                        if len(line) > 20 and len(line) < 200:
                            key_discoveries.append(line.strip())
                            break

        # Build summary with priority order
        if files_modified:
            findings.append(f"Files modified: {', '.join(list(files_modified)[:5])}")

        if files_examined:
            findings.append(f"Files examined: {', '.join(list(files_examined)[:5])}")

        if symbols_found:
            findings.append(f"Symbols looked up: {', '.join(symbols_found[:5])}")

        if searches_performed:
            findings.append(f"Searches performed: {', '.join(searches_performed[:3])}")

        if errors_encountered:
            findings.append(f"Errors encountered: {len(errors_encountered)} (check tool outputs)")

        if key_discoveries:
            findings.append("Key discoveries:")
            for discovery in key_discoveries[:3]:
                findings.append(f"  • {discovery}")

        # Extract last assistant message for continuity
        last_assistant_msg = None
        for msg in reversed(history):
            if msg.role == "assistant" and msg.content:
                last_assistant_msg = str(msg.content)
                break

        if last_assistant_msg:
            # Extract key sentence from last response
            sentences = re.split(r"[.!?]+", last_assistant_msg)
            for sentence in sentences:
                if len(sentence) > 30 and any(
                    word in sentence.lower()
                    for word in ["found", "identified", "located", "discovered", "analyzed"]
                ):
                    findings.append(f"Previous step result: {sentence.strip()}")
                    break

        if not findings:
            findings.append(f"Completed {current_step - 1} exploratory steps")

        # Truncate if too long
        result = "\n".join(findings)
        if len(result) > self.max_context_chars:
            result = result[: self.max_context_chars - 20] + "\n... (context truncated)"

        return result

    def get_redundant_operations(self, history: list[Message]) -> dict[str, set[str]]:
        """Identify operations that should not be repeated.

        Returns:
            Dictionary mapping operation types to specific targets to avoid
        """
        redundant = defaultdict(set)

        for msg in history:
            if msg.role == "assistant" and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    if isinstance(tool_call, dict):
                        tool_name = tool_call.get("function", {}).get("name", "")
                        args = tool_call.get("function", {}).get("arguments", {})

                        if isinstance(args, dict):
                            # Track files already read
                            if tool_name == READ or "read" in tool_name.lower():
                                file_path = args.get("file_path")
                                if file_path:
                                    redundant["files_read"].add(file_path)

                            # Track searches already done
                            elif "search" in tool_name.lower():
                                query = args.get("query") or args.get("pattern")
                                if query:
                                    redundant["searches_done"].add(query)

                            # Track executed commands
                            elif (
                                tool_name == RUN
                                or "run" in tool_name.lower()
                                or "exec" in tool_name.lower()
                            ):
                                cmd = args.get("command") or args.get("cmd")
                                if cmd:
                                    redundant["commands_run"].add(cmd)

        return dict(redundant)

    def build_constraints_section(self, redundant_ops: dict[str, set[str]]) -> str:
        """Build constraints section based on redundant operations."""
        constraints = []

        if redundant_ops.get("files_read"):
            files = list(redundant_ops["files_read"])[:3]
            constraints.append(f"Already examined files: {', '.join(files)} (avoid re-reading)")

        if redundant_ops.get("searches_done"):
            searches = list(redundant_ops["searches_done"])[:2]
            constraints.append(
                f"Already searched for: {', '.join(searches)} (use different patterns)"
            )

        if not constraints:
            constraints.append("No redundant operations detected")

        return "\n".join(f"• {c}" for c in constraints)
