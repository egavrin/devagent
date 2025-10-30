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
from ai_dev_agent.providers.llm.base import Message
from ai_dev_agent.tool_names import FIND, GREP, READ, RUN, SYMBOLS, WRITE

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

_PROVIDER_PREAMBLES: dict[str, str] = {
    "anthropic": "Anthropic models favour concise tool arguments and defer to user confirmations when tooling is denied.",
    "deepseek": "DeepSeek models can emit multi-step tool calls; keep responses grounded in repository evidence and summarise final answers clearly.",
    "openai": "OpenAI GPT models support JSON function calls; prefer structured outputs when sharing lists or diagnostics.",
    "google": "Gemini models may rate-limit long outputs; batch tool calls and stream concise updates when possible.",
    "openrouter": "OpenRouter routes to various providers; follow model-specific best practices and provide clear, structured responses.",
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

    if include_react_guidance:
        guidance_sections.append(
            _react_guidance(iteration_cap, repository_language, settings=settings)
        )

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
        fallback = "You are a helpful assistant for the devagent CLI tool. Prioritise accurate reasoning and safe tool usage."
        messages.append(Message(role="system", content=fallback))

    return messages


def _resolve_workspace_root(workspace_root: Path | None, settings: Settings | None) -> Path:
    candidate = workspace_root or getattr(settings, "workspace_root", None) or Path.cwd()
    try:
        return candidate.resolve()
    except OSError:
        return Path.cwd()


def _provider_preamble(provider: str, model: str | None) -> str:
    key = provider.lower().strip()
    if not key:
        return ""

    # Detect model family for OpenRouter to apply appropriate optimizations
    model_lower = (model or "").lower()
    if key == "openrouter":
        if "claude" in model_lower or "anthropic" in model_lower:
            # Use Anthropic-specific guidance for Claude models via OpenRouter
            anthropic_guidance = _PROVIDER_PREAMBLES.get("anthropic", "")
            openrouter_base = _PROVIDER_PREAMBLES.get("openrouter", "")
            combined = f"{anthropic_guidance} {openrouter_base}".strip()
            if model:
                return f"Model: {model} (via OpenRouter). {combined}"
            return f"Provider: OpenRouter (Claude). {combined}"
        elif "gpt" in model_lower or "openai" in model_lower:
            # Use OpenAI-specific guidance for GPT models via OpenRouter
            openai_guidance = _PROVIDER_PREAMBLES.get("openai", "")
            openrouter_base = _PROVIDER_PREAMBLES.get("openrouter", "")
            combined = f"{openai_guidance} {openrouter_base}".strip()
            if model:
                return f"Model: {model} (via OpenRouter). {combined}"
            return f"Provider: OpenRouter (GPT). {combined}"
        elif "gemini" in model_lower or "google" in model_lower:
            # Use Google-specific guidance for Gemini models via OpenRouter
            google_guidance = _PROVIDER_PREAMBLES.get("google", "")
            openrouter_base = _PROVIDER_PREAMBLES.get("openrouter", "")
            combined = f"{google_guidance} {openrouter_base}".strip()
            if model:
                return f"Model: {model} (via OpenRouter). {combined}"
            return f"Provider: OpenRouter (Gemini). {combined}"

    base = _PROVIDER_PREAMBLES.get(key)
    if not base:
        return f"You are running on the {provider} provider{f' using {model}' if model else ''}. Optimise tool usage and produce grounded answers."
    if model:
        return f"Model: {model} ({provider}). {base}"
    return f"Provider: {provider}. {base}"


def _react_guidance(
    iteration_cap: int | None,
    repository_language: str | None,
    *,
    settings: Settings | None = None,
) -> str:
    core = (
        "You are a helpful assistant for the devagent CLI tool, specialised in efficient software development tasks.\n\n"
        "## MISSION\n"
        "Complete the user's task efficiently using available tools.\n"
        "Pay attention to budget status messages that guide your execution strategy.\n\n"
        "## CORE PRINCIPLES\n"
        "1. EFFICIENCY: Choose the most appropriate tool for each task\n"
        "2. AVOID REDUNDANCY: Never repeat identical tool calls\n"
        "3. BULK OPERATIONS: Prefer batch operations over individual file reads\n"
        "4. EARLY TERMINATION: Stop when you have sufficient information\n"
        "5. ADAPTIVE STRATEGY: Change approach if tools fail\n"
        "6. SCRIPT GENERATION: Create scripts for complex computations\n"
        "\n"
        "## COMMUNICATION STYLE\n"
        "Be concise and direct. Minimize output tokens while maintaining accuracy.\n"
        "- For simple questions, give direct answers without preamble\n"
        "- One-word or one-sentence answers are preferred when appropriate\n"
        "- Avoid unnecessary explanations unless asked\n"
        "- Don't add phrases like 'The answer is...' or 'Based on...'\n"
        "\nExamples of proper conciseness:\n"
        "User: 'What is 2+2?' -> You: '4'\n"
        "User: 'Is 11 prime?' -> You: 'Yes'\n"
        "User: 'Which file has the User class?' -> You: 'models/user.py'\n"
        "User: 'How many Python files in src/?' -> You: '23'\n"
        "\nFor complex tasks:\n"
        "- Provide complete information but be succinct\n"
        "- Match detail level to task complexity\n"
        "- Explain non-trivial commands before running them\n"
    )

    if repository_language:
        hint = _LANGUAGE_HINTS.get(str(repository_language).lower())
        if hint:
            core += f"\nLANGUAGE-SPECIFIC ({repository_language}):{hint}\n"

    tool_semantics = (
        "\nTOOL SEMANTICS:\n"
        f"- {RUN}: Runs commands in POSIX shell; pipes, globs, redirects work\n"
        "- Prefer machine-parsable output (e.g. find -print0) over formatted listings\n"
        "- Minimise tool calls – stop once you have the answer\n"
        "\nPARALLEL TOOL EXECUTION:\n"
        "- You have the capability to call multiple tools in a single response\n"
        "- When multiple independent pieces of information are requested, batch your tool calls for optimal performance\n"
        "- Independent tool calls will execute concurrently, providing 3-5x speedup\n"
        "- Examples:\n"
        "  • Reading 3 different files: batch into one response instead of 3 sequential calls\n"
        f"  • Multiple searches: run {FIND}/{GREP} queries in parallel\n"
        f"  • File read + search: execute simultaneously if independent\n"
        "- IMPORTANT: Only batch truly independent operations (don't batch if one depends on another's result)\n"
    )

    output_discipline = (
        "\nOUTPUT REQUIREMENTS:\n"
        "- State scope explicitly (depth, hidden files, symlinks)\n"
        "- Ensure counts match actual listed items\n"
        "- Stop executing once you have sufficient information\n"
    )

    code_edit_guidance = (
        "\nCODE EDITING BEST PRACTICES:\n"
        "\nWhen modifying code:\n"
        "1. Read the file first to understand context and conventions\n"
        "2. Follow existing code style (indentation, naming, imports)\n"
        f"3. Use {WRITE} for surgical changes to existing files\n"
        "4. Don't modify unrelated code - stay focused on the task\n"
        "5. Verify library imports exist in the project before using them\n"
        "\nFormat for file changes:\n"
        f"- Use unified diff format ({WRITE}) for existing files\n"
        "- Show context lines around changes for clarity\n"
        "- Make multiple small patches rather than one large rewrite\n"
        "\nExample patch structure:\n"
        "  --- a/file.py\n"
        "  +++ b/file.py\n"
        "  @@ -10,6 +10,7 @@\n"
        "   def existing_function():\n"
        "       existing_line\n"
        "  +    new_line_added\n"
        "       another_existing_line\n"
        "\nDon't:\n"
        "- Leave TODO/FIXME comments without implementing the code\n"
        "- Add unnecessary comments explaining obvious code\n"
        "- Modify code outside the scope of the user's request\n"
        "- Assume libraries are available without checking\n"
    )

    fs_recipes = (
        "\nCOMMON OPERATIONS:\n"
        "Count files: `find . -maxdepth 1 -type f | wc -l`\n"
        "List safely: `find . -maxdepth 1 -type f -print0 | xargs -0 -n1 basename`\n"
        "Verify location: `pwd` and `ls -la`\n"
    )

    failure_handling = (
        "\nFAILURE HANDLING:\n"
        "- First failure: Adjust parameters\n"
        "- Second failure: Switch tools/approach\n"
        "- System blocks identical calls after two failures\n"
        "- Three or more consecutive failures should trigger termination\n"
    )

    anti_patterns = (
        "\nANTI-PATTERNS TO AVOID:\n"
        "\nCode quality:\n"
        "- NEVER leave TODO or FIXME comments - implement the code completely\n"
        "- Don't add placeholder functions that need to be 'filled in later'\n"
        "- Don't write incomplete implementations with '# rest of code here' comments\n"
        "- Avoid over-commenting obvious code\n"
        "\nScope discipline:\n"
        "- Don't refactor or 'improve' code outside the user's request\n"
        "- Don't fix unrelated bugs or style issues unless asked\n"
        "- Don't add extra features beyond what was requested\n"
        "- Stay focused on the specific task at hand\n"
        "\nTool usage:\n"
        "- Don't search for the same pattern multiple times\n"
        "- Don't read files you've already read\n"
        "- Don't run tests without knowing the test command\n"
        "- Don't assume build tools or frameworks are available\n"
        "\nSecurity:\n"
        "- Never commit API keys, passwords, or secrets\n"
        "- Don't log sensitive information\n"
    )

    tool_guidance = (
        "\nUNIVERSAL TOOL STRATEGIES:\n"
        "- For counting/metrics: Use shell commands with find/grep/wc\n"
        "- For pattern search: Start with grep or find, then read specific files\n"
        "- For exploration: Use symbols for structure and definitions\n"
        "- For bulk operations: Generate and execute scripts\n"
        "- For file operations: Prefer find/xargs over individual reads\n"
        "- Verify unexpected results (especially counts <=1) with pwd and ls -la\n"
    )

    tool_priority = (
        "\nTOOL SELECTION GUIDE:\n"
        f"- Finding files -> {FIND} ('*.py' or '**/test_*.js')\n"
        f"- Searching content -> {GREP} ('TODO' or 'error.*', regex=true)\n"
        f"- Finding symbols -> {SYMBOLS} ('ClassName' or 'functionName')\n"
        f"- Reading specific files -> {READ}\n"
        f"- Running commands -> {RUN}\n"
        f"- Making changes -> {WRITE}\n"
    )

    tool_descriptions = (
        "\nDETAILED TOOL DESCRIPTIONS:\n"
        "\nfind:\n"
        "  Purpose: Find files by pattern using ripgrep glob syntax\n"
        "  When to use: Looking for files by name or extension\n"
        "  Examples:\n"
        "    find('*.py') - all Python files\n"
        "    find('**/test_*.js') - test files in any directory\n"
        "    find('src/**/*.ts') - TypeScript files under src\n"
        "  Parameters: query (glob pattern), path (optional), limit (default 100)\n"
        "  Results: Sorted by modification time (newest first)\n"
        "\ngrep:\n"
        "  Purpose: Search content in files using ripgrep\n"
        "  When to use: Finding specific text or patterns in code\n"
        "  Examples:\n"
        "    grep('TODO') - find all TODO comments\n"
        "    grep('func.*name', regex=true) - regex search\n"
        "    grep('error', path='src/') - search only in src\n"
        "  Parameters: pattern, path (optional), regex (default false), limit\n"
        "  Results: Grouped by file, sorted by modification time\n"
        "\nsymbols:\n"
        "  Purpose: Find symbol definitions using universal ctags\n"
        "  When to use: Looking for functions, classes, variables by name\n"
        "  Examples:\n"
        "    symbols('MyClass') - find class definition\n"
        "    symbols('process_data') - find function definition\n"
        "  Parameters: name (substring match), path (optional), limit\n"
        "  Note: Requires ctags/universal-ctags to be installed\n"
        f"\n{READ}:\n"
        "  Purpose: Read file contents from the repository\n"
        "  When to use: After finding files with find/grep/symbols\n"
        "  Parameters: paths (list of strings), optional context_lines or byte_range\n"
        f"\n{RUN}:\n"
        "  Purpose: Execute shell commands directly\n"
        "  When to use: For git operations, running tests, building\n"
        "  Parameters: cmd (string), optional args (list)\n"
        f"\n{WRITE}:\n"
        "  Purpose: Apply unified diff patches to files\n"
        "  When to use: Making precise changes to existing files\n"
        "  Parameters: patches (list of {path, patch_text})\n"
    )

    tool_workflows = (
        "\nCOMMON TOOL WORKFLOWS:\n"
        "\nFinding files:\n"
        "  1. find('*.py') -> Lists all Python files\n"
        f"  2. {READ} to examine specific files\n"
        "\nFinding and modifying a function:\n"
        "  1. symbols('function_name') -> Finds definition\n"
        f"  2. {READ} the file to see full context\n"
        f"  3. {WRITE} to make changes\n"
        "\nSearching for patterns:\n"
        "  1. grep('TODO') -> Finds all occurrences\n"
        "  2. grep('error.*handling', regex=true) for complex patterns\n"
        "\nRefactoring across files:\n"
        "  1. grep for usage patterns\n"
        f"  2. {READ} files that need changes\n"
        f"  3. {WRITE} each file\n"
        "\nSimple principles:\n"
        f"  - {FIND} for files, {GREP} for content, {SYMBOLS} for definitions\n"
        "  - Results are sorted by modification time (newest first)\n"
        "  - No hidden heuristics - what you search is what you get\n"
    )

    return (
        core
        + tool_semantics
        + output_discipline
        + code_edit_guidance
        + fs_recipes
        + failure_handling
        + anti_patterns
        + tool_guidance
        + tool_priority
        + tool_descriptions
        + tool_workflows
    )


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
