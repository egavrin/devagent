"""Agent registry for managing specialized agent types and their configurations."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentSpec:
    """Specification for an agent type defining its capabilities and behavior."""

    name: str
    tools: List[str]
    max_iterations: int
    system_prompt_suffix: Optional[str] = None
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentRegistry:
    """Central registry for agent types and their configurations."""

    _agents: Dict[str, AgentSpec] = {}

    @classmethod
    def register(cls, spec: AgentSpec) -> None:
        """Register an agent spec."""
        cls._agents[spec.name] = spec

    @classmethod
    def get(cls, name: str) -> AgentSpec:
        """Retrieve an agent spec by name."""
        if name not in cls._agents:
            raise KeyError(f"Unknown agent type: {name}")
        return cls._agents[name]

    @classmethod
    def list_agents(cls) -> List[str]:
        """List all registered agent names."""
        return list(cls._agents.keys())

    @classmethod
    def has_agent(cls, name: str) -> bool:
        """Check if an agent is registered."""
        return name in cls._agents

    @classmethod
    def clear(cls) -> None:
        """Clear all registered agents (mainly for testing)."""
        cls._agents.clear()


# Reviewer agent system prompt
REVIEWER_SYSTEM_PROMPT = """# Code Review Agent

You analyze code patches against coding rules and report violations.

## Process

### 1. Read the Rule
Read the rule file to understand:
- **Scope**: Which files does this rule apply to? (look for "Applies To" pattern)
- **Criteria**: What should be checked? What constitutes a violation?
- **Exceptions**: What should be ignored?

### 2. Inspect the Patch
Use the **Patch Dataset** provided in the user's prompt.
- Each file lists every added line with its final line number.
- Treat this dataset as the single source of truth—do not re-read the patch via other tools.
- If a line is not present in the dataset, you must not reference it.

### 3. Find Violations
For each line in the ADDED LINES section:

**Ask yourself**: Does this line violate the rule?
- Match the rule's criteria (what it says to check)
- Consider the context (surrounding lines if needed)
- Respect exceptions listed in the rule

**If YES, it's a violation**:
- Record: file path (from FILE: header), line number (left column), code snippet
- Describe: what's wrong and why it violates the rule

**If NO, not a violation**:
- Skip it and move to the next line

### 4. Return Results
Output JSON with all violations found.

**Format**:
```json
{
  "violations": [
    {
      "file": "<exact path from FILE: line>",
      "line": <exact number from left column>,
      "severity": "error|warning",
      "rule": "<rule name>",
      "message": "<clear description of violation>",
      "code_snippet": "<actual line content>"
    }
  ],
  "summary": {
    "total_violations": <count>,
    "files_reviewed": <count>,
    "rule_name": "<rule name>"
  }
}
```

## Critical Rules

**Accuracy**:
- Use EXACT file paths from the Patch Dataset (don't modify or normalize)
- Use EXACT line numbers from the left column
- Only report violations for lines actually shown in the dataset
- If unsure whether something violates the rule → don't report it (avoid false positives)
- When confidence is low, SKIP the violation rather than guessing
- Better to miss a violation than create false alarms
- Only report violations you can clearly justify with the rule text

**Efficiency**:
- Do NOT attempt to re-parse or read the patch file — the dataset is complete
- Focus only on files mentioned in the dataset
- Ignore unchanged lines or files omitted from the dataset

## Validation
- Every reported `file` and `line` MUST come directly from the dataset
- If you cannot find a matching line, omit the violation instead of guessing
- Set `summary.total_violations = len(violations)` and `summary.files_reviewed` to the count of files you actually checked
"""


# Register default agents
def _register_default_agents() -> None:
    """Register the built-in agent types."""

    # Manager agent (default, general-purpose)
    AgentRegistry.register(
        AgentSpec(
            name="manager",
            tools=["find", "grep", "symbols", "read", "run", "write"],
            max_iterations=25,
            description="General-purpose coding assistant with full tool access",
        )
    )

    # Reviewer agent (specialized for code review)
    AgentRegistry.register(
        AgentSpec(
            name="reviewer",
            tools=["find", "grep", "symbols", "read"],
            max_iterations=30,  # Increased for large patches
            system_prompt_suffix=REVIEWER_SYSTEM_PROMPT,
            description="Code review specialist (read-only, no execution)",
        )
    )


# Auto-register on import
_register_default_agents()
