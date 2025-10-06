"""Provider-specific prompt templates."""
from __future__ import annotations

import os
import platform
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class PromptContext:
    """Context information for prompt generation."""

    working_directory: str
    is_git_repo: bool
    platform: str
    language: Optional[str] = None
    framework: Optional[str] = None
    project_structure: Optional[str] = None
    custom_instructions: Optional[str] = None
    phase: Optional[str] = None
    task_description: Optional[str] = None


class ProviderPrompts:
    """Provider-specific prompt templates."""

    # Base templates for different providers
    ANTHROPIC_BASE = """You are Claude, an AI assistant created by Anthropic.
You are an expert software engineer helping with development tasks.
You are direct, helpful, and focused on producing correct, working code."""

    OPENAI_BASE = """You are an expert software engineer assistant.
You provide clear, accurate technical guidance and working code solutions.
You follow best practices and write clean, maintainable code."""

    GEMINI_BASE = """You are an AI assistant specialized in software development.
You help users with coding tasks, debugging, and technical problem-solving.
You write efficient, well-structured code following industry standards."""

    DEEPSEEK_BASE = """You are a coding assistant focused on practical solutions.
You provide working code and clear technical explanations.
You emphasize correctness and efficiency in your implementations."""

    # Conciseness instructions
    CONCISENESS_PROMPT = """
# Communication Style
- Be concise, direct, and to the point
- Minimize output tokens while maintaining accuracy
- Avoid unnecessary preambles or explanations unless requested
- Provide one-word answers when appropriate
- Skip introductions and conclusions

Examples:
- Question: "What is 2+2?" Answer: "4"
- Question: "Is 11 prime?" Answer: "Yes"
- Question: "Command to list files?" Answer: "ls"
"""

    # Phase-specific guidance
    PHASE_GUIDANCE = {
        "exploration": """
# Current Phase: EXPLORATION
- Cast a wide net to understand the codebase
- Identify key components and architecture
- Map out file structures and dependencies
- Look for patterns and conventions
""",
        "investigation": """
# Current Phase: INVESTIGATION
- Focus on specific areas identified during exploration
- Deep dive into relevant code sections
- Validate hypotheses and assumptions
- Gather detailed implementation information
""",
        "consolidation": """
# Current Phase: CONSOLIDATION
- Connect findings from investigation
- Synthesize discoveries into actionable insights
- Validate conclusions against evidence
- Prepare comprehensive solution approach
""",
        "synthesis": """
# Current Phase: SYNTHESIS
- Provide complete, actionable response
- Include specific file locations and code references
- Highlight any remaining unknowns or risks
- Deliver self-contained solution
"""
    }

    # Language-specific hints
    LANGUAGE_HINTS = {
        "python": "Check requirements.txt/pyproject.toml for dependencies, follow PEP 8 conventions",
        "typescript": "Check package.json for dependencies, use proper TypeScript types",
        "javascript": "Check package.json, follow ES6+ patterns, consider async/await",
        "java": "Check pom.xml/gradle files, follow Java naming conventions",
        "go": "Check go.mod for dependencies, follow Go idioms and conventions",
        "rust": "Check Cargo.toml, follow Rust ownership patterns and idioms"
    }

    @classmethod
    def get_system_prompt(
        cls,
        provider: str,
        context: PromptContext
    ) -> str:
        """Generate provider-specific system prompt."""

        # Start with provider base
        base_prompt = cls._get_provider_base(provider)

        # Add environment context
        env_section = cls._build_environment_section(context)

        # Add phase guidance if applicable
        phase_section = ""
        if context.phase:
            phase_section = cls.PHASE_GUIDANCE.get(context.phase, "")

        # Add language hints
        language_section = ""
        if context.language:
            hint = cls.LANGUAGE_HINTS.get(context.language.lower(), "")
            if hint:
                language_section = f"\n# Language: {context.language}\n{hint}\n"

        # Add custom instructions
        custom_section = ""
        if context.custom_instructions:
            custom_section = f"\n# Project Instructions\n{context.custom_instructions}\n"

        # Add conciseness for appropriate providers
        conciseness = ""
        if provider in ["anthropic", "openai"]:
            conciseness = cls.CONCISENESS_PROMPT

        # Combine all sections
        prompt = f"""{base_prompt}

{env_section}
{phase_section}
{language_section}
{custom_section}
{conciseness}

# Task
{context.task_description or "Assist the user with their request."}
"""

        return prompt.strip()

    @classmethod
    def _get_provider_base(cls, provider: str) -> str:
        """Get base prompt for provider."""
        provider = provider.lower()

        if "anthropic" in provider or "claude" in provider:
            return cls.ANTHROPIC_BASE
        elif "openai" in provider or "gpt" in provider:
            return cls.OPENAI_BASE
        elif "gemini" in provider or "google" in provider:
            return cls.GEMINI_BASE
        elif "deepseek" in provider:
            return cls.DEEPSEEK_BASE
        else:
            return cls.OPENAI_BASE  # Default

    @classmethod
    def _build_environment_section(cls, context: PromptContext) -> str:
        """Build environment context section."""

        env_parts = [
            "# Environment",
            f"Working directory: {context.working_directory}",
            f"Platform: {context.platform}",
            f"Date: {datetime.now().strftime('%Y-%m-%d')}",
        ]

        if context.is_git_repo:
            env_parts.append("Git repository: Yes")

        if context.framework:
            env_parts.append(f"Framework: {context.framework}")

        if context.project_structure:
            env_parts.append(f"\n# Project Structure\n{context.project_structure}")

        return "\n".join(env_parts)

    @classmethod
    def get_reflection_prompt(cls, error_message: str, attempt: int) -> str:
        """Generate reflection prompt for error recovery."""
        return f"""# Reflection Required

Previous attempt {attempt} failed with error:
{error_message}

Please:
1. Analyze what went wrong
2. Adjust your approach
3. Try a different strategy

Avoid repeating the same failed operation."""

    @classmethod
    def get_summary_prompt(cls, preserve_details: bool = True) -> str:
        """Get conversation summary prompt."""

        if preserve_details:
            return """Create a detailed summary that preserves:
- All file paths and function names
- Error messages and stack traces
- Specific code changes made
- Technical decisions and rationale

Write from the user's perspective using "I asked you..."
Focus more on recent messages than older ones."""
        else:
            return """Create a concise summary of the conversation.
Focus on key outcomes and decisions.
Write from the user's perspective."""

    @classmethod
    def get_tool_reminder(cls, phase: str, available_tools: List[str]) -> str:
        """Generate phase-appropriate tool usage reminder."""

        if phase == "exploration":
            return f"""# Available Tools for Exploration
{chr(10).join(f'- {tool}' for tool in available_tools)}

Use search and discovery tools extensively to map the codebase."""

        elif phase == "investigation":
            return f"""# Available Tools for Investigation
{chr(10).join(f'- {tool}' for tool in available_tools)}

Focus on reading specific files and understanding implementations."""

        elif phase == "synthesis":
            return """# Synthesis Phase
No tools available - provide your complete analysis and recommendations."""

        else:
            return f"""# Available Tools
{chr(10).join(f'- {tool}' for tool in available_tools)}"""


class PromptLoader:
    """Load custom prompts from files."""

    # Standard instruction files to check
    INSTRUCTION_FILES = [
        "DEVAGENT.md",
        "AGENTS.md",
        "CLAUDE.md",
        "CONTEXT.md",  # deprecated but still checked
        ".devagent/instructions.md"
    ]

    @classmethod
    def load_custom_instructions(cls, project_root: Optional[Path] = None) -> Optional[str]:
        """Load custom instructions from project files."""

        if project_root is None:
            project_root = Path.cwd()

        instructions = []

        # Check each standard location
        for filename in cls.INSTRUCTION_FILES:
            file_path = project_root / filename

            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            instructions.append(f"# From {filename}\n{content}")
                except Exception:
                    continue

        # Also check home directory for global instructions
        home_instructions = Path.home() / ".devagent" / "instructions.md"
        if home_instructions.exists():
            try:
                with open(home_instructions, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        instructions.append(f"# Global Instructions\n{content}")
            except Exception:
                pass

        if instructions:
            return "\n\n".join(instructions)

        return None

    @classmethod
    def get_language_from_project(cls, project_root: Optional[Path] = None) -> Optional[str]:
        """Detect primary language from project files."""

        if project_root is None:
            project_root = Path.cwd()

        # Check for language-specific files
        checks = [
            ("package.json", "javascript"),
            ("tsconfig.json", "typescript"),
            ("requirements.txt", "python"),
            ("pyproject.toml", "python"),
            ("go.mod", "go"),
            ("Cargo.toml", "rust"),
            ("pom.xml", "java"),
            ("build.gradle", "java"),
            ("composer.json", "php"),
            ("Gemfile", "ruby")
        ]

        for filename, language in checks:
            if (project_root / filename).exists():
                return language

        # Fallback: count file extensions
        extension_counts = {}
        for pattern in ["*.py", "*.js", "*.ts", "*.java", "*.go", "*.rs"]:
            count = len(list(project_root.glob(f"**/{pattern}")))
            if count > 0:
                ext = pattern[2:]
                extension_counts[ext] = count

        if extension_counts:
            # Return language with most files
            most_common = max(extension_counts, key=extension_counts.get)
            return {
                "py": "python",
                "js": "javascript",
                "ts": "typescript",
                "java": "java",
                "go": "go",
                "rs": "rust"
            }.get(most_common, "unknown")

        return None
