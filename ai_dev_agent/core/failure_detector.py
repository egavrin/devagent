"""Failure pattern detection to avoid wasted LLM calls on impossible queries."""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from collections import defaultdict


@dataclass
class FailurePatternDetector:
    """Detects repeated failures and suggests early exit to prevent wasted LLM calls.

    Based on analysis showing devagent burns 25 LLM calls retrying the same failed
    operations. This detector recognizes failure patterns and surfaces them to the LLM.
    """

    failure_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    failed_operations: List[Tuple[str, str, str]] = field(default_factory=list)

    MAX_SAME_FAILURE = 3  # After 3 identical failures, suggest giving up
    MAX_GREP_FAILURES = 3  # After 3 grep failures for related terms, suggest giving up
    MAX_READ_REJECTIONS = 5  # After 5 read rejections, suggest different approach

    def record_failure(self, operation: str, target: str, reason: str) -> None:
        """Record a tool failure for pattern detection.

        Args:
            operation: Tool name (e.g., "grep", "read", "find")
            target: Target of operation (pattern, file path, etc.)
            reason: Failure reason or error message
        """
        key = f"{operation}:{target}"
        self.failure_counts[key] += 1
        self.failed_operations.append((operation, target, reason))

    def should_give_up(self, operation: str, target: str) -> Tuple[bool, str]:
        """Check if we should stop trying this operation.

        Returns:
            Tuple of (should_stop, helpful_message)
        """
        key = f"{operation}:{target}"
        count = self.failure_counts[key]

        # Pattern 1: Exact same operation failed multiple times
        if count >= self.MAX_SAME_FAILURE:
            return True, (
                f"⚠️ **Repeated Failure Detected**\n\n"
                f"Attempted '{operation}' on '{target}' {count} times without success.\n"
                f"This artifact likely doesn't exist in the repository.\n\n"
                f"**Suggestions:**\n"
                f"1. Check if the path/symbol name is correct\n"
                f"2. Search with broader patterns (e.g., '*{target}*')\n"
                f"3. Try a different search approach\n"
                f"4. Conclude that this artifact is not present in this codebase"
            )

        # Pattern 2: Multiple grep failures for similar symbols
        if operation == "grep":
            target_lower = target.lower()
            grep_failures = [
                tgt for op, tgt, _ in self.failed_operations
                if op == "grep" and (target_lower in tgt.lower() or tgt.lower() in target_lower)
            ]
            if len(grep_failures) >= self.MAX_GREP_FAILURES:
                return True, (
                    f"⚠️ **Multiple Search Failures Detected**\n\n"
                    f"Multiple grep searches for '{target}' and related terms failed:\n"
                    f"  - {', '.join(grep_failures[:5])}\n\n"
                    f"The requested symbols may not exist in this codebase.\n\n"
                    f"**Suggestions:**\n"
                    f"1. Verify the symbol name is correct\n"
                    f"2. Try searching for a simpler/shorter pattern\n"
                    f"3. Check if this is the right repository for this code\n"
                    f"4. Conclude that these symbols are not present"
                )

        # Pattern 3: Multiple read rejections (files too large, binary, etc.)
        if operation == "read":
            read_rejections = [
                (tgt, reason) for op, tgt, reason in self.failed_operations
                if op == "read" and "reject" in reason.lower()
            ]
            if len(read_rejections) >= self.MAX_READ_REJECTIONS:
                rejected_paths = [tgt for tgt, _ in read_rejections]
                return True, (
                    f"⚠️ **Multiple Read Rejections Detected**\n\n"
                    f"Multiple read operations rejected ({len(read_rejections)} files).\n"
                    f"Files may be too large, binary, or inaccessible.\n\n"
                    f"**Rejected files:** {', '.join(rejected_paths[:3])}\n\n"
                    f"**Suggestions:**\n"
                    f"1. Use grep to search file contents instead of reading\n"
                    f"2. Use find with file type filters (-name '*.cpp' -type f)\n"
                    f"3. Check if files are actually text files\n"
                    f"4. Try reading smaller related files"
                )

        return False, ""

    def get_summary(self) -> str:
        """Get a summary of all failures for debugging."""
        if not self.failed_operations:
            return "No failures recorded."

        summary = f"Total failures: {len(self.failed_operations)}\n\n"

        # Group by operation type
        by_operation = defaultdict(list)
        for op, target, reason in self.failed_operations:
            by_operation[op].append((target, reason))

        for op, failures in by_operation.items():
            summary += f"{op}: {len(failures)} failures\n"
            for target, reason in failures[:3]:  # Show first 3
                summary += f"  - {target}: {reason[:50]}\n"
            if len(failures) > 3:
                summary += f"  ... and {len(failures) - 3} more\n"
            summary += "\n"

        return summary

    def reset(self) -> None:
        """Reset all failure tracking (e.g., for a new query)."""
        self.failure_counts.clear()
        self.failed_operations.clear()
