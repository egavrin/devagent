# DevAgent AI Agent Guide

This guide condenses the working rules and development process for DevAgent — a TypeScript monorepo providing an AI-powered CLI for development workflows. Every practice described here flows from the fail-fast philosophy: surface bugs loudly, avoid masking symptoms, and force root-cause fixes.

## Fail Fast Philosophy
- **Expose issues immediately**: let exceptions surface and prefer explicit failures over defensive fallbacks. Hidden errors compound into larger outages.
- **Fix causes, not symptoms**: trace failures to their root, add tests that reproduce them, and delete band-aid code once the underlying issue is resolved.
- **Learn from every failure**: capture the signal in tests, documentation, and status updates so the next agent starts with context instead of rediscovering the bug.

## Core Expectations
- **Lean into fail-fast coding**: no silent guards, no best-effort recoveries—raise loudly and repair quickly.
- **Never create new `.md` files unless the user explicitly asks**: capture notes in the conversation or update existing Markdown files instead.
- **Front-load situational awareness**: run `bun run test` and `bun run typecheck` before touching code.
- **Design before implementation**: draft the approach in the conversation or update existing docs.
- **Work in tight loops**: write tests before code, keep functionality changes tiny, leave git commits to the user.

## Implementation Loop
- **Tests-first TDD**: add or extend tests in `packages/*/src/*.test.ts` before touching production modules; ensure the new tests fail for the expected reason before editing functionality.
- **Run to failure**: execute the new tests and let them fail—capture stack traces and error messages so the root cause is obvious.
- **Implement after failing tests**: once the tests define the behavior and fail, update production code in minimal increments to make them pass without breaking prior capabilities.
- **Delete defensive shims**: remove temporary guards or fallback logic once the fix lands to keep the failure path visible if it ever regresses.
- **Commit handoff**: never run `git commit` or otherwise write to git history; prepare commit-ready diffs and message suggestions so the user can apply them when satisfied.

## Anti-Patterns To Avoid
- **Silent guards**: never swallow exceptions or return default values to keep the workflow limping along. If a dependency misbehaves, raise and investigate.
- **Maze-of-fallback logic**: When resolving files, configuration, or CLI entrypoints, avoid layered "best guess" heuristics. Resolve a single authoritative location (or require the caller to provide one) and fail fast if it is missing.
- **Implicit behavioural switches**: Configuration like settings, prompt paths, or tooling should be explicit. If logic needs to diverge, wire it through a well-documented parameter rather than hidden fallbacks.

## Project Structure

```
packages/
  cli/        # Terminal CLI entry point (bin: devagent)
  core/       # Types, config, events, approval gate, session management
  engine/     # TaskLoop, agents, orchestration, review pipeline, plugins
  tools/      # Tool registry, builtins (patch-parser, file ops), LSP, MCP
  providers/  # LLM provider abstraction (Anthropic, OpenAI, Ollama, ChatGPT)
  arkts/      # ArkTS linter support
  desktop/    # Tauri desktop app (SolidJS frontend)
models/       # LLM provider config files (TOML)
prompts/      # Shared prompt templates
```

## CLI Commands Reference

### Core Commands
- `devagent "<query>"` – Natural language query (direct execution, auto-detects intent)
- `devagent --plan "<query>"` – Query with LLM planning mode (read-only analysis)
- `devagent chat` – Interactive chat session with persistent context
- `devagent review <file> --rule <rule_file> [--json]` – Rule-based patch review with structured output

### Auth Commands
- `devagent auth login` – Store API key for a provider
- `devagent auth status` – Show configured credentials
- `devagent auth logout` – Remove stored credentials

### Options
- `-f, --file <path>` – Read the query from a file
- `--provider <name>` – LLM provider (anthropic, openai, ollama, chatgpt, github-copilot)
- `--model <id>` – Model ID
- `--max-iterations <n>` – Max tool-call iterations
- `--reasoning <level>` – Reasoning effort: low, medium, high
- `--resume <id>` – Resume a previous session
- `--continue` – Resume the most recent session
- `--suggest` / `--auto-edit` / `--full-auto` – Approval modes
- `-v, --verbose` / `-q, --quiet` – Verbosity levels
- `--json` – JSON output format

### Interactive Commands
- `/plan` – Switch to plan mode (read-only)
- `/act` – Switch to act mode
- `/review <file> [--diff]` – Quick code review
- `/clear` – Clear conversation history
- `/checkpoint list|restore|diff` – Checkpoint management
- `/skills` – List available skills
- `/commands` – List available plugin commands
- `/<skill-name> [args]` – Invoke a skill by name

## Skills System

DevAgent implements the [Agent Skills standard](https://agentskills.io) for cross-tool compatible skills.

### Skill Format
Skills are directories containing a `SKILL.md` file with YAML frontmatter:
```
my-skill/
  SKILL.md           # Required: frontmatter (name, description) + instructions
  scripts/           # Optional: executable scripts
  references/        # Optional: documentation
  assets/            # Optional: templates, resources
```

### Discovery Paths (priority order)
1. `.devagent/skills/` (project, highest priority)
2. `.claude/skills/` (project, Claude-compatible)
3. `.agents/skills/` (project, standard)
4. `~/.claude/skills/` (global)
5. `~/.agents/skills/` (global)
6. `~/.config/devagent/skills/` (global, lowest priority)

### Invocation
- **LLM auto-invocation**: Agent calls `invoke_skill` tool when task matches a skill
- **Slash command**: `/<skill-name> [arguments]` in interactive mode
- **Arguments**: `$ARGUMENTS`, `$0`/`$1`/`$N`, `${SKILL_DIR}`, `${SESSION_ID}`, `` !`command` ``

## Build & Test

```bash
# Install dependencies
bun install

# Build all packages
bun run build

# Run tests
bun run test

# Type check
bun run typecheck

# Install CLI globally
bun run install-cli
```

## Verification Checklist
- Run `bun run typecheck` — zero errors required.
- Run `bun run test` — all tests must pass.
- Run `bun run build` — successful build required.
- Scan recent history with `git log --oneline -5` for commit format compliance.
