"""Helpers for constructing consistent system prompts across DevAgent surfaces."""

from __future__ import annotations

import os
import platform
import subprocess
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from ai_dev_agent.core.utils.context_budget import summarize_text
from ai_dev_agent.prompts.loader import PromptLoader
from ai_dev_agent.providers.llm.base import Message
from ai_dev_agent.tool_names import EDIT, FIND, GREP, READ, RUN, SYMBOLS

if TYPE_CHECKING:
    from ai_dev_agent.core.utils.config import Settings

_LANGUAGE_HINTS: dict[str, str] = {
    "python": "\n- Use symbols for Python code structure\n- Check requirements.txt/setup.py for dependencies\n- Use import analysis for module relationships",
    "javascript": "\n- Consider package.json for dependencies\n- Use symbols for JS/TS structure\n- Check for .eslintrc for code standards",
    "typescript": "\n- Check tsconfig.json for compilation settings\n- Use symbols for TypeScript analysis\n- Consider type definitions in .d.ts files",
    "java": "\n- Check pom.xml or build.gradle for dependencies\n- Use symbols for class hierarchies\n- Consider package structure for organization",
    "c++": "\n- Check CMakeLists.txt or Makefile for build config\n- Look for .h/.hpp headers separately from .cpp/.cc files\n- Use compile_commands.json if available",
    "c": "\n- Check Makefile or CMakeLists.txt for build setup\n- Analyze header files (.h) for interfaces\n- Use grep for macro definitions",
    "go": "\n- Check go.mod for module dependencies\n- Use go tools for analysis\n- Consider internal vs external packages",
    "rust": "\n- Check Cargo.toml for dependencies\n- Use cargo commands for analysis\n- Consider module structure in lib.rs/main.rs",
    "ruby": "\n- Check Gemfile for dependencies\n- Look for rake tasks\n- Consider Rails structure if applicable",
    "php": "\n- Check composer.json for dependencies\n- Look for autoload configurations\n- Consider framework structure (Laravel, Symfony)",
    "c#": "\n- Check .csproj or .sln files\n- Use NuGet packages info\n- Consider namespace organization",
    "swift": "\n- Check Package.swift for dependencies\n- Look for .xcodeproj/xcworkspace\n- Consider iOS/macOS target differences",
    "kotlin": "\n- Check build.gradle.kts for configuration\n- Consider Android vs JVM targets\n- Use Gradle for dependency info",
    "scala": "\n- Check build.sbt for dependencies\n- Use sbt commands for analysis\n- Consider Play/Akka frameworks if present",
    "dart": "\n- Check pubspec.yaml for dependencies\n- Consider Flutter structure if applicable\n- Use dart analyze for code issues",
}

_DEFAULT_INSTRUCTION_GLOBS: Sequence[str] = (
    "AGENTS.md",
    "CLAUDE.md",
    "CONTEXT.md",
    ".devagent/instructions/*.md",
)

_GLOBAL_INSTRUCTION_CANDIDATES: Sequence[Path] = (
    Path.home() / ".devagent" / "AGENTS.md",
    Path.home() / ".config" / "devagent" / "instructions.md",
)

_PROMPT_LOADER: PromptLoader | None = None

_SYSTEM_FALLBACK_MESSAGE = "You are a helpful software development assistant."


def _get_prompt_loader() -> PromptLoader:
    global _PROMPT_LOADER
    if _PROMPT_LOADER is None:
        _PROMPT_LOADER = PromptLoader()
    return _PROMPT_LOADER


def build_system_messages(
    *,
    iteration_cap: int | None = None,
    repository_language: str | None = None,
    include_react_guidance: bool = True,
    extra_messages: list[str] | None = None,
    provider: str | None = None,
    model: str | None = None,
    workspace_root: Path | None = None,
    settings: Settings | None = None,
    instruction_paths: Sequence[str] | None = None,
) -> list[Message]:
    """Produce baseline system messages reused across DevAgent entry points."""

    root = _resolve_workspace_root(workspace_root, settings)

    guidance_sections: list[str] = []
    context_sections: list[str] = []

    provider_preamble = _provider_preamble(
        provider or (getattr(settings, "provider", None) or ""),
        model or getattr(settings, "model", None),
    )
    if provider_preamble:
        guidance_sections.append(provider_preamble)

    prompt_context = _system_prompt_context(
        iteration_cap=iteration_cap,
        repository_language=repository_language,
        settings=settings,
    )

    base_prompt = _base_system_prompt(prompt_context)
    if base_prompt:
        guidance_sections.append(base_prompt)

    if include_react_guidance:
        react_guidance = _react_guidance(prompt_context)
        if react_guidance:
            guidance_sections.append(react_guidance)

    environment_snapshot = _environment_snapshot(root)
    if environment_snapshot:
        context_sections.append(environment_snapshot)

    instruction_blocks = _instruction_overlays(root, instruction_paths, settings)
    if instruction_blocks:
        context_sections.append("Additional instructions:\n" + "\n\n".join(instruction_blocks))

    if extra_messages:
        context_sections.append(
            "\n".join(entry.strip() for entry in extra_messages if entry).strip()
        )

    primary_text = "\n\n".join(section.strip() for section in guidance_sections if section).strip()

    messages: list[Message] = []
    if primary_text:
        messages.append(Message(role="system", content=primary_text))

    for section in context_sections:
        text = section.strip()
        if text:
            messages.append(Message(role="system", content=text))

    if not messages:
        messages.append(Message(role="system", content=_SYSTEM_FALLBACK_MESSAGE))

    return messages


def _resolve_workspace_root(workspace_root: Path | None, settings: Settings | None) -> Path:
    candidate = workspace_root or getattr(settings, "workspace_root", None) or Path.cwd()
    try:
        return candidate.resolve()
    except OSError:
        return Path.cwd()


def _provider_preamble(provider: str, model: str | None) -> str:
    """Return an empty preamble to maintain provider-agnostic prompts."""
    _ = provider, model
    return ""


def _base_system_prompt(context: dict[str, str]) -> str:
    loader = _get_prompt_loader()
    prompt = loader.load_system_prompt(context=context)
    return prompt.strip()


def _react_guidance(context: dict[str, str]) -> str:
    loader = _get_prompt_loader()
    prompt = loader.render_prompt("system/react_loop.md", context)
    return prompt.strip()


def _system_prompt_context(
    *,
    iteration_cap: int | None,
    repository_language: str | None,
    settings: Settings | None,
) -> dict[str, str]:
    """Build context dict with UPPERCASE keys for {{PLACEHOLDER}} substitution."""
    language_hint = _language_hint_block(repository_language)
    iteration_note = _iteration_note(iteration_cap, settings)
    iteration_cap_value = str(iteration_cap) if iteration_cap is not None else ""

    # Keys are UPPERCASE to match {{PLACEHOLDER}} syntax in prompts
    return {
        "ITERATION_CAP": iteration_cap_value,
        "ITERATION_NOTE": iteration_note,
        "LANGUAGE_HINT": language_hint,
        "REPOSITORY_LANGUAGE": repository_language or "",
        "TOOL_EDIT": EDIT,
        "TOOL_FIND": FIND,
        "TOOL_GREP": GREP,
        "TOOL_SYMBOLS": SYMBOLS,
        "TOOL_READ": READ,
        "TOOL_RUN": RUN,
    }


def _iteration_note(iteration_cap: int | None, settings: Settings | None) -> str:
    if iteration_cap is None:
        fallback = getattr(settings, "max_iterations", None)
        if fallback:
            return (
                f"Iterations default to the configured maximum of {fallback}. "
                "Respect the budget and stop when the task is complete."
            )
        return ""
    return (
        f"Iterations are limited to {iteration_cap} steps. "
        "Plan tool usage so you finish within the cap."
    )


def _language_hint_block(repository_language: str | None) -> str:
    if not repository_language:
        return ""

    hint = _LANGUAGE_HINTS.get(str(repository_language).lower())
    if not hint:
        return ""

    return f"### Language-Specific Guidance ({repository_language}){hint}"


def _environment_snapshot(root: Path) -> str:
    lines = ["Environment snapshot:"]
    lines.append(f"  Workspace: {root}")
    lines.append(f"  Python: {platform.python_version()}")
    lines.append(f"  Platform: {platform.platform()}")
    lines.append(f"  Timestamp: {datetime.utcnow().isoformat()}Z")

    git_lines = _git_context(root)
    lines.extend(f"  {entry}" for entry in git_lines)

    return "\n".join(lines)


def _git_context(root: Path) -> list[str]:
    args = ["git", "rev-parse", "--is-inside-work-tree"]
    try:
        probe = subprocess.run(args, cwd=root, capture_output=True, text=True, check=False)
    except (OSError, ValueError):
        return ["Git: unavailable"]

    if probe.returncode != 0 or probe.stdout.strip().lower() != "true":
        return ["Git: not a repository"]

    context: list[str] = []
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], root)
    if branch:
        context.append(f"Git branch: {branch}")

    head = _run_git(["rev-parse", "--short", "HEAD"], root)
    if head:
        context.append(f"Git commit: {head}")

    status_raw = _run_git(["status", "--short"], root)
    if status_raw is not None:
        changes = [line for line in status_raw.splitlines() if line.strip()]
        if changes:
            sample = ", ".join(changes[:4])
            if len(changes) > 4:
                sample += ", …"
            context.append(f"Git status: {len(changes)} change(s) ({sample})")
        else:
            context.append("Git status: clean")
    return context


def _run_git(arguments: Sequence[str], root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments], cwd=root, capture_output=True, text=True, check=False
        )
    except (OSError, ValueError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _instruction_overlays(
    root: Path,
    instruction_paths: Sequence[str] | None,
    settings: Settings | None,
    *,
    max_chars: int = 4_000,
) -> list[str]:
    candidates: list[str] = []
    seen: set[Path] = set()

    for pattern in _DEFAULT_INSTRUCTION_GLOBS:
        candidates.extend(_expand_instruction_glob(root, pattern))

    for global_candidate in _GLOBAL_INSTRUCTION_CANDIDATES:
        if global_candidate.is_file():
            candidates.append(str(global_candidate))

    if settings:
        provider_cfg = getattr(settings, "provider_config", {})
        if isinstance(provider_cfg, dict):
            extra = provider_cfg.get("prompt_instructions")
            if isinstance(extra, str):
                candidates.extend([extra])
            elif isinstance(extra, Iterable):
                for item in extra:
                    if isinstance(item, str):
                        candidates.append(item)

    env_instructions = os.getenv("DEVAGENT_PROMPT_INSTRUCTIONS")
    if env_instructions:
        for token in env_instructions.split(os.pathsep):
            token = token.strip()
            if token:
                candidates.append(token)

    if instruction_paths:
        candidates.extend(list(instruction_paths))

    blocks: list[str] = []
    for candidate in candidates:
        path = Path(candidate)
        if not path.is_absolute():
            path = (root / path).resolve()
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        try:
            text = resolved.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        text = summarize_text(text.strip(), max_chars)
        if text:
            blocks.append(f"[{resolved.name}]\n{text}")
    return blocks


def _expand_instruction_glob(root: Path, pattern: str) -> list[str]:
    if "*" not in pattern:
        return [pattern]
    try:
        matches = list(root.glob(pattern))
    except OSError:
        return []
    return [str(match) for match in matches if match.is_file()]
