"""Memory Distiller - Extracts reusable strategies and lessons from task completions.

Based on ReasoningBank paper: distills high-level lessons from both successes
and failures to create a compact, generalizable knowledge base.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from ai_dev_agent.providers.llm.base import Message

logger = logging.getLogger(__name__)


@dataclass
class Strategy:
    """A successful problem-solving strategy."""

    description: str
    context: str  # When to apply this strategy
    steps: list[str]  # Key steps taken
    tools_used: list[str]  # Tools that were effective
    effectiveness_score: float = 0.0  # Tracked over time

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Lesson:
    """A lesson learned from a failure or mistake."""

    mistake: str  # What went wrong
    context: str  # When this mistake occurs
    correction: str  # How to avoid/fix it
    severity: str  # "critical", "major", "minor"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Memory:
    """A distilled memory item for the reasoning bank."""

    memory_id: str = field(default_factory=lambda: str(uuid4()))
    task_type: str = ""  # debugging, feature, refactoring, etc.
    title: str = ""  # Brief description
    query: str = ""  # Original user query
    strategies: list[Strategy] = field(default_factory=list)
    lessons: list[Lesson] = field(default_factory=list)
    outcome: str = "unknown"  # success, partial, failure
    context_hash: str = ""  # Hash of original query for deduplication
    embedding: list[float] | None = None  # Vector for similarity search
    usage_count: int = 0
    effectiveness_score: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_used: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data["strategies"] = [
            s.to_dict() if isinstance(s, Strategy) else s for s in self.strategies
        ]
        data["lessons"] = [l.to_dict() if isinstance(l, Lesson) else l for l in self.lessons]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Memory:
        """Create from dictionary."""
        strategies = [
            Strategy(**s) if isinstance(s, dict) else s for s in data.get("strategies", [])
        ]
        lessons = [Lesson(**l) if isinstance(l, dict) else l for l in data.get("lessons", [])]

        # Remove strategies and lessons from data to avoid duplicate in kwargs
        memory_data = {k: v for k, v in data.items() if k not in ["strategies", "lessons"]}

        return cls(strategies=strategies, lessons=lessons, **memory_data)


class MemoryDistiller:
    """Distills reusable knowledge from completed tasks."""

    # Task type patterns
    TASK_PATTERNS = {
        "debugging": [
            r"fix\s+(?:the\s+)?(?:bug|error|issue)",
            r"debug(?:ging)?",
            r"(?:type|syntax|runtime)\s*error",
            r"exception|traceback|crash",
            r"not\s+working",
        ],
        "feature": [
            r"(?:add|implement|create)\s+(?:a\s+)?(?:new\s+)?feature",
            r"add\s+support\s+for",
            r"implement\s+\w+",
            r"create\s+(?:a\s+)?new",
        ],
        "refactoring": [
            r"refactor(?:ing)?",
            r"restructure",
            r"improve\s+(?:code\s+)?(?:quality|structure)",
            r"clean(?:\s+)?up",
            r"optimize",
        ],
        "testing": [
            r"(?:write|add|create)\s+(?:unit\s+)?tests?",
            r"test\s+coverage",
            r"testing",
            r"test\s+suite",
        ],
        "review": [
            r"review\s+(?:the\s+)?(?:code|changes|pr)",
            r"code\s+review",
            r"feedback",
            r"suggestions",
        ],
    }

    # Success/failure indicators
    SUCCESS_INDICATORS = [
        "successfully",
        "fixed",
        "resolved",
        "completed",
        "working",
        "implemented",
        "added",
        "created",
        "passed",
        "success",
    ]

    FAILURE_INDICATORS = [
        "failed",
        "error",
        "couldn't",
        "unable",
        "timeout",
        "incorrect",
        "wrong",
        "broken",
        "issue",
    ]

    def __init__(self, llm_client: Any | None = None):
        """Initialize the distiller.

        Args:
            llm_client: Optional LLM client for advanced analysis
        """
        self.llm_client = llm_client

    def distill_from_session(
        self, session_id: str, messages: list[Message], metadata: dict[str, Any] | None = None
    ) -> Memory:
        """Distill a memory from a completed session.

        Args:
            session_id: Session identifier
            messages: Conversation messages
            metadata: Optional metadata about the session

        Returns:
            Distilled memory item
        """
        # Extract basic information
        query = self._extract_user_query(messages)
        task_type = self._identify_task_type(query)
        outcome = self._determine_outcome(messages)

        # Extract strategies and lessons
        strategies = self._extract_strategies(messages, outcome)
        lessons = self._extract_lessons(messages, outcome)

        # Generate title
        title = self._generate_title(query, task_type, outcome)

        # Create context hash for deduplication
        context_hash = hashlib.md5(query.encode()).hexdigest()

        # Build memory
        memory = Memory(
            task_type=task_type,
            title=title,
            query=query,
            strategies=strategies,
            lessons=lessons,
            outcome=outcome,
            context_hash=context_hash,
            metadata=metadata or {},
        )

        logger.debug(f"Distilled memory from session {session_id}: {title}")
        return memory

    def _extract_user_query(self, messages: list[Message]) -> str:
        """Extract the original user query from messages."""
        for msg in messages:
            if msg.role == "user" and msg.content:
                # Skip RepoMap context messages
                if not msg.content.startswith("[") and not msg.content.startswith("Here are"):
                    return msg.content.strip()
        return ""

    def _identify_task_type(self, query: str) -> str:
        """Identify the type of task from the query."""
        query_lower = query.lower()

        for task_type, patterns in self.TASK_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return task_type

        return "general"

    def _determine_outcome(self, messages: list[Message]) -> str:
        """Determine if the task was successful, partial, or failed."""
        # Look at the last few assistant messages
        assistant_msgs = [m for m in messages if m.role == "assistant"][-3:]

        if not assistant_msgs:
            return "unknown"

        # Combine content for analysis
        content = " ".join(m.content or "" for m in assistant_msgs).lower()

        # Count indicators
        success_count = sum(1 for ind in self.SUCCESS_INDICATORS if ind in content)
        failure_count = sum(1 for ind in self.FAILURE_INDICATORS if ind in content)

        # Determine outcome
        if success_count > failure_count * 2:
            return "success"
        elif failure_count > success_count * 2:
            return "failure"
        elif success_count > 0 or failure_count > 0:
            return "partial"

        return "unknown"

    def _extract_strategies(self, messages: list[Message], outcome: str) -> list[Strategy]:
        """Extract successful strategies from the conversation."""
        strategies = []

        if outcome not in ["success", "partial"]:
            return strategies

        # Look for patterns indicating successful approaches
        for i, msg in enumerate(messages):
            if msg.role == "assistant" and msg.content:
                content = msg.content

                # Look for strategy indicators
                if any(
                    phrase in content.lower()
                    for phrase in [
                        "i'll",
                        "let me",
                        "first",
                        "next",
                        "then",
                        "approach",
                        "strategy",
                        "plan",
                    ]
                ):
                    # Extract tools used from this message
                    tools = self._extract_tools_from_message(msg)

                    # Look for step-by-step actions
                    steps = self._extract_steps_from_content(content)

                    if steps:
                        strategy = Strategy(
                            description=self._summarize_approach(content),
                            context=self._extract_context(messages[:i]),
                            steps=steps[:5],  # Keep top 5 steps
                            tools_used=tools,
                        )
                        strategies.append(strategy)

                        # Usually one main strategy per task
                        if len(strategies) >= 2:
                            break

        return strategies

    def _extract_lessons(self, messages: list[Message], outcome: str) -> list[Lesson]:
        """Extract lessons from failures and corrections."""
        lessons = []

        # Look for error patterns and corrections
        for i, msg in enumerate(messages):
            if msg.role == "assistant" and msg.content:
                content = msg.content

                # Check for error acknowledgment
                if any(
                    phrase in content.lower()
                    for phrase in [
                        "error",
                        "mistake",
                        "incorrect",
                        "failed",
                        "should have",
                        "instead",
                        "actually",
                        "the issue was",
                        "the problem is",
                    ]
                ):
                    # Extract the mistake and correction
                    mistake = self._extract_mistake(content)
                    correction = self._extract_correction(content, messages[i:])

                    if mistake and correction:
                        lesson = Lesson(
                            mistake=mistake,
                            context=self._extract_context(messages[:i]),
                            correction=correction,
                            severity=self._assess_severity(content),
                        )
                        lessons.append(lesson)

            # Also learn from user corrections
            elif msg.role == "user" and i > 0:
                prev_msg = messages[i - 1]
                if prev_msg.role == "assistant" and any(
                    phrase in msg.content.lower()
                    for phrase in [
                        "no",
                        "wrong",
                        "incorrect",
                        "actually",
                        "that's not",
                        "should be",
                        "try",
                    ]
                ):
                    mistake = self._summarize_approach(prev_msg.content or "")
                    correction = msg.content[:200]

                    lesson = Lesson(
                        mistake=mistake,
                        context=self._extract_context(messages[: i - 1]),
                        correction=f"User correction: {correction}",
                        severity="major",
                    )
                    lessons.append(lesson)

        return lessons[:3]  # Keep top 3 lessons

    def _extract_tools_from_message(self, msg: Message) -> list[str]:
        """Extract tool names from a message."""
        tools = []

        # Check tool_calls field
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for call in msg.tool_calls:
                if isinstance(call, dict) and "function" in call:
                    tool_name = call["function"].get("name", "")
                    if tool_name and tool_name not in tools:
                        tools.append(tool_name)

        # Also check content for tool indicators
        if msg.content:
            # Pattern for tool usage in content
            tool_patterns = [
                r"(?:read|reading)\s+(\S+)",
                r"(?:write|writing)\s+(?:to\s+)?(\S+)",
                r"(?:search|searching)\s+for",
                r"(?:grep|find)\s+",
                r"run(?:ning)?\s+(?:command\s+)?",
            ]

            for pattern in tool_patterns:
                if re.search(pattern, msg.content, re.IGNORECASE):
                    # Map to standard tool names
                    if "read" in pattern:
                        tools.append("read")
                    elif "write" in pattern:
                        tools.append("write")
                    elif "search" in pattern or "grep" in pattern or "find" in pattern:
                        tools.append("search")
                    elif "run" in pattern:
                        tools.append("run")

        return list(set(tools))  # Unique tools

    def _extract_steps_from_content(self, content: str) -> list[str]:
        """Extract actionable steps from content."""
        steps = []

        # Look for numbered or bulleted lists
        lines = content.split("\n")
        for line in lines:
            line = line.strip()

            # Numbered steps (1., 2., etc)
            if re.match(r"^\d+\.?\s+", line):
                step = re.sub(r"^\d+\.?\s+", "", line)
                if len(step) > 10 and len(step) < 200:
                    steps.append(step)

            # Bullet points
            elif re.match(r"^[-*•]\s+", line):
                step = re.sub(r"^[-*•]\s+", "", line)
                if len(step) > 10 and len(step) < 200:
                    steps.append(step)

            # Action phrases
            elif any(
                line.lower().startswith(word)
                for word in [
                    "check",
                    "verify",
                    "search",
                    "look for",
                    "find",
                    "read",
                    "examine",
                    "analyze",
                    "implement",
                    "add",
                    "fix",
                    "update",
                    "modify",
                    "create",
                    "write",
                ]
            ):
                if len(line) > 10 and len(line) < 200:
                    steps.append(line)

        return steps

    def _summarize_approach(self, content: str) -> str:
        """Create a brief summary of the approach."""
        # Take first meaningful sentence
        sentences = content.split(".")
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20 and len(sentence) < 200:
                # Clean up the sentence
                sentence = re.sub(r"\s+", " ", sentence)
                return sentence

        # Fallback: take first 150 chars
        summary = re.sub(r"\s+", " ", content[:150]).strip()
        if not summary.endswith("."):
            summary += "..."
        return summary

    def _extract_context(self, messages: list[Message]) -> str:
        """Extract context about when to apply strategy/lesson."""
        # Look for file mentions, error types, etc.
        context_parts = []

        for msg in messages[-3:]:  # Last 3 messages before
            if msg.content:
                # Look for file extensions
                extensions = re.findall(r"\.\w{2,4}\b", msg.content)
                if extensions:
                    context_parts.append(f"Working with {', '.join(set(extensions))} files")

                # Look for error types
                errors = re.findall(r"(?:Type|Syntax|Runtime|Import|Attribute)Error", msg.content)
                if errors:
                    context_parts.append(f"Dealing with {', '.join(set(errors))}")

                # Look for frameworks/libraries
                frameworks = re.findall(
                    r"\b(?:django|flask|react|vue|numpy|pandas|pytest)\b", msg.content.lower()
                )
                if frameworks:
                    context_parts.append(f"Using {', '.join(set(frameworks))}")

        return "; ".join(context_parts) if context_parts else "General development context"

    def _extract_mistake(self, content: str) -> str:
        """Extract what went wrong."""
        # Look for error descriptions
        patterns = [
            r"(?:the\s+)?(?:error|mistake|issue|problem)\s+(?:was|is)\s+(.+?)(?:\.|;|$)",
            r"(?:failed|couldn't|unable)\s+to\s+(.+?)(?:\.|;|$)",
            r"(?:incorrect|wrong)\s+(.+?)(?:\.|;|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, content.lower())
            if match:
                return match.group(1).strip()

        # Fallback: summarize the content
        return self._summarize_approach(content)

    def _extract_correction(self, content: str, following_msgs: list[Message]) -> str:
        """Extract how to fix the mistake."""
        # Look in current message
        patterns = [
            r"(?:should|need\s+to|must|have\s+to)\s+(.+?)(?:\.|;|$)",
            r"(?:instead|actually|correctly)\s+(.+?)(?:\.|;|$)",
            r"(?:the\s+)?(?:fix|solution|correction)\s+(?:is|was)\s+(.+?)(?:\.|;|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, content.lower())
            if match:
                return match.group(1).strip()

        # Look in following messages for the fix
        for msg in following_msgs[:2]:
            if msg.role == "assistant" and msg.content:
                if "fixed" in msg.content.lower() or "resolved" in msg.content.lower():
                    return self._summarize_approach(msg.content)

        return "Review approach and verify assumptions"

    def _assess_severity(self, content: str) -> str:
        """Assess the severity of a mistake."""
        content_lower = content.lower()

        if any(word in content_lower for word in ["critical", "severe", "crash", "data loss"]):
            return "critical"
        elif any(word in content_lower for word in ["error", "failed", "broken", "incorrect"]):
            return "major"
        else:
            return "minor"

    def _generate_title(self, query: str, task_type: str, outcome: str) -> str:
        """Generate a concise title for the memory."""
        # Clean and truncate query
        title = re.sub(r"\s+", " ", query[:100]).strip()

        # Add task type prefix if not already present
        if task_type != "general" and task_type not in title.lower():
            title = f"[{task_type.title()}] {title}"

        # Add outcome indicator
        if outcome == "success":
            title = f"✓ {title}"
        elif outcome == "failure":
            title = f"✗ {title}"
        elif outcome == "partial":
            title = f"⚠ {title}"

        return title

    def merge_similar_memories(self, memory1: Memory, memory2: Memory) -> Memory:
        """Merge two similar memories into one."""
        # Combine strategies, keeping unique ones
        combined_strategies = memory1.strategies + memory2.strategies
        unique_strategies = []
        seen_descriptions = set()

        for strategy in combined_strategies:
            if strategy.description not in seen_descriptions:
                unique_strategies.append(strategy)
                seen_descriptions.add(strategy.description)

        # Combine lessons, keeping unique ones
        combined_lessons = memory1.lessons + memory2.lessons
        unique_lessons = []
        seen_mistakes = set()

        for lesson in combined_lessons:
            if lesson.mistake not in seen_mistakes:
                unique_lessons.append(lesson)
                seen_mistakes.add(lesson.mistake)

        # Use the more successful outcome
        if memory1.outcome == "success" or memory2.outcome == "success":
            outcome = "success"
        elif memory1.outcome == "partial" or memory2.outcome == "partial":
            outcome = "partial"
        else:
            outcome = memory1.outcome

        # Merge metadata
        merged_metadata = {**memory1.metadata, **memory2.metadata}
        merged_metadata["merged_from"] = [memory1.memory_id, memory2.memory_id]
        merged_metadata["merge_date"] = datetime.now().isoformat()

        # Combine usage counts properly
        total_usage = memory1.usage_count + memory2.usage_count

        return Memory(
            task_type=memory1.task_type,
            title=memory1.title if len(memory1.title) <= len(memory2.title) else memory2.title,
            query=memory1.query,  # Keep first query
            strategies=unique_strategies[:5],  # Limit to top 5
            lessons=unique_lessons[:3],  # Limit to top 3
            outcome=outcome,
            context_hash=memory1.context_hash,
            usage_count=total_usage,
            effectiveness_score=max(memory1.effectiveness_score, memory2.effectiveness_score),
            created_at=min(memory1.created_at, memory2.created_at),
            metadata=merged_metadata,
        )
