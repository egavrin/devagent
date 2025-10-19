"""Pattern Tracker for Dynamic Instruction System.

Tracks query patterns, success rates, tool sequences, and error patterns
to enable automatic instruction proposal generation.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class QueryRecord:
    """Record of a single query execution."""

    session_id: str
    success: bool
    tools_used: List[str]
    task_type: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    error_type: Optional[str] = None
    duration_seconds: Optional[float] = None


@dataclass
class PatternSignal:
    """A detected pattern with statistical significance."""

    pattern_type: str  # "tool_sequence", "error_recovery", "success_strategy"
    description: str
    query_count: int
    success_rate: float
    confidence: float  # 0.0-1.0 based on sample size and consistency
    examples: List[str] = field(default_factory=list)  # Session IDs

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PatternTracker:
    """Tracks query execution patterns for automatic instruction generation."""

    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize the pattern tracker.

        Args:
            storage_path: Path to store pattern data (default: ~/.devagent/dynamic_instructions/patterns.json)
        """
        self.storage_path = storage_path or (
            Path.home() / ".devagent" / "dynamic_instructions" / "patterns.json"
        )

        self._lock = threading.RLock()
        self._query_records: List[QueryRecord] = []
        self._query_count = 0

        # Pattern caches
        self._tool_sequences: defaultdict[Tuple[str, ...], List[bool]] = defaultdict(list)
        self._task_type_success: defaultdict[str, List[bool]] = defaultdict(list)

        # Load existing data
        self._load_data()

    def _load_data(self) -> None:
        """Load pattern data from storage."""
        if not self.storage_path.exists():
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created pattern tracker storage at {self.storage_path}")
            return

        try:
            with open(self.storage_path, "r") as f:
                data = json.load(f)

            # Load query records
            records_data = data.get("query_records", [])
            self._query_records = [
                QueryRecord(**record) for record in records_data
            ]
            self._query_count = data.get("query_count", len(self._query_records))

            # Rebuild pattern caches from records
            self._rebuild_caches()

            logger.debug(f"Loaded {len(self._query_records)} query records")

        except Exception as e:
            logger.warning(f"Failed to load pattern data: {e}")

    def _save_data(self) -> None:
        """Save pattern data to storage."""
        with self._lock:
            # Keep only last 500 records to prevent unbounded growth
            records_to_save = self._query_records[-500:]

            data = {
                "query_count": self._query_count,
                "query_records": [asdict(record) for record in records_to_save],
                "saved_at": datetime.now().isoformat()
            }

            try:
                with open(self.storage_path, "w") as f:
                    json.dump(data, f, indent=2)
                logger.debug(f"Saved pattern data ({len(records_to_save)} records)")
            except Exception as e:
                logger.warning(f"Failed to save pattern data: {e}")

    def _rebuild_caches(self) -> None:
        """Rebuild pattern caches from query records."""
        self._tool_sequences.clear()
        self._task_type_success.clear()

        for record in self._query_records:
            # Tool sequences
            if len(record.tools_used) >= 2:
                seq = tuple(record.tools_used[:3])  # Track up to 3-tool sequences
                self._tool_sequences[seq].append(record.success)

            # Task type success
            self._task_type_success[record.task_type].append(record.success)

    def record_query(
        self,
        session_id: str,
        success: bool,
        tools_used: List[str],
        task_type: str = "general",
        error_type: Optional[str] = None,
        duration_seconds: Optional[float] = None
    ) -> None:
        """Record a query execution.

        Args:
            session_id: Unique session identifier
            success: Whether query was successful
            tools_used: List of tools used in order
            task_type: Type of task (e.g., "debugging", "feature", "general")
            error_type: Type of error if failed
            duration_seconds: Query duration in seconds
        """
        with self._lock:
            record = QueryRecord(
                session_id=session_id,
                success=success,
                tools_used=tools_used,
                task_type=task_type,
                error_type=error_type,
                duration_seconds=duration_seconds
            )

            self._query_records.append(record)
            self._query_count += 1

            # Update caches
            if len(tools_used) >= 2:
                seq = tuple(tools_used[:3])
                self._tool_sequences[seq].append(success)

            self._task_type_success[task_type].append(success)

            # Save periodically (every 10 queries)
            if self._query_count % 10 == 0:
                self._save_data()

            logger.debug(f"Recorded query: {task_type}, success={success}, tools={len(tools_used)}")

    def get_query_count(self) -> int:
        """Get total number of queries recorded."""
        return self._query_count

    def has_significant_patterns(self, min_queries: int = 10) -> bool:
        """Check if enough data exists for meaningful pattern analysis.

        Args:
            min_queries: Minimum queries needed

        Returns:
            True if enough patterns exist
        """
        return self._query_count >= min_queries

    def detect_patterns(self, min_sample_size: int = 5, min_success_rate: float = 0.7) -> List[PatternSignal]:
        """Detect significant patterns from recorded queries.

        Args:
            min_sample_size: Minimum occurrences to consider a pattern
            min_success_rate: Minimum success rate to consider significant

        Returns:
            List of detected patterns with confidence scores
        """
        patterns: List[PatternSignal] = []

        with self._lock:
            # 1. Tool sequence patterns
            for seq, results in self._tool_sequences.items():
                if len(results) < min_sample_size:
                    continue

                success_rate = sum(results) / len(results)
                if success_rate < min_success_rate:
                    continue

                # Calculate confidence based on sample size and consistency
                confidence = self._calculate_confidence(results, success_rate)

                # Get example session IDs
                examples = [
                    record.session_id
                    for record in self._query_records
                    if tuple(record.tools_used[:3]) == seq
                ][:3]  # Keep up to 3 examples

                tool_names = " → ".join(seq)
                patterns.append(PatternSignal(
                    pattern_type="tool_sequence",
                    description=f"Tool sequence: {tool_names}",
                    query_count=len(results),
                    success_rate=success_rate,
                    confidence=confidence,
                    examples=examples
                ))

            # 2. Task-specific success patterns
            for task_type, results in self._task_type_success.items():
                if len(results) < min_sample_size:
                    continue

                success_rate = sum(results) / len(results)

                # Look for both high and low success rates
                if success_rate >= min_success_rate:
                    confidence = self._calculate_confidence(results, success_rate)

                    examples = [
                        record.session_id
                        for record in self._query_records
                        if record.task_type == task_type and record.success
                    ][:3]

                    patterns.append(PatternSignal(
                        pattern_type="success_strategy",
                        description=f"Task type '{task_type}' has high success rate",
                        query_count=len(results),
                        success_rate=success_rate,
                        confidence=confidence,
                        examples=examples
                    ))
                elif success_rate < 0.4:  # Low success - failure pattern
                    confidence = self._calculate_confidence(results, 1.0 - success_rate)

                    examples = [
                        record.session_id
                        for record in self._query_records
                        if record.task_type == task_type and not record.success
                    ][:3]

                    patterns.append(PatternSignal(
                        pattern_type="failure_pattern",
                        description=f"Task type '{task_type}' has low success rate",
                        query_count=len(results),
                        success_rate=success_rate,
                        confidence=confidence,
                        examples=examples
                    ))

            # 3. Error recovery patterns
            error_recoveries = defaultdict(lambda: {"success": 0, "total": 0, "examples": []})

            for i, record in enumerate(self._query_records[:-1]):
                if record.error_type:
                    next_record = self._query_records[i + 1]
                    if next_record.task_type == record.task_type:
                        error_recoveries[record.error_type]["total"] += 1
                        if next_record.success:
                            error_recoveries[record.error_type]["success"] += 1
                            error_recoveries[record.error_type]["examples"].append(next_record.session_id)

            for error_type, stats in error_recoveries.items():
                if stats["total"] < min_sample_size:
                    continue

                success_rate = stats["success"] / stats["total"]
                if success_rate < min_success_rate:
                    continue

                # Simplified confidence for recovery patterns
                confidence = min(0.9, success_rate * (stats["total"] / 20))

                patterns.append(PatternSignal(
                    pattern_type="error_recovery",
                    description=f"Recovery from '{error_type}' errors",
                    query_count=stats["total"],
                    success_rate=success_rate,
                    confidence=confidence,
                    examples=stats["examples"][:3]
                ))

        # Sort by confidence (highest first)
        patterns.sort(key=lambda p: p.confidence, reverse=True)

        logger.info(f"Detected {len(patterns)} significant patterns")
        return patterns

    def _calculate_confidence(self, results: List[bool], success_rate: float) -> float:
        """Calculate confidence score for a pattern.

        Confidence is based on:
        - Sample size (more samples = higher confidence)
        - Success rate consistency (less variance = higher confidence)
        - Minimum threshold (never below 0.5 for significant patterns)

        Args:
            results: List of success/failure booleans
            success_rate: Overall success rate

        Returns:
            Confidence score (0.0-1.0)
        """
        n = len(results)

        # Sample size component (asymptotically approaches 1.0)
        sample_confidence = min(1.0, n / 20)  # Max out at 20 samples

        # Consistency component (how stable is the success rate)
        # Perfect consistency (all same) = 1.0, highly variable = lower
        consistency = success_rate if success_rate > 0.5 else (1.0 - success_rate)

        # Combined confidence (weighted average)
        confidence = 0.6 * sample_confidence + 0.4 * consistency

        # Ensure minimum confidence for returned patterns
        return max(0.5, min(1.0, confidence))

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about recorded patterns.

        Returns:
            Dictionary of statistics
        """
        with self._lock:
            total_success = sum(1 for r in self._query_records if r.success)

            return {
                "total_queries": self._query_count,
                "success_rate": total_success / max(1, len(self._query_records)),
                "unique_tool_sequences": len(self._tool_sequences),
                "task_types": list(self._task_type_success.keys()),
                "records_in_memory": len(self._query_records),
                "storage_path": str(self.storage_path)
            }

    def clear(self) -> None:
        """Clear all recorded patterns (for testing)."""
        with self._lock:
            self._query_records.clear()
            self._query_count = 0
            self._tool_sequences.clear()
            self._task_type_success.clear()
            self._save_data()
            logger.info("Cleared all pattern data")
