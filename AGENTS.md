# DevAgent AI Agent Guide

This guide condenses the working rules, launch instructions, and helper tooling for Claude Code, Codex CLI, and any DevAgent-based automation. It combines the authoritative rules in `.devagent/ai_agent_instructions.md`, the user guide in `docs/USER_GUIDE.md`, the context builder in `scripts/ai_develop.sh`, and the copy-ready prompts in `CLAUDE_CODEX_PROMPT.txt`.

## Core Expectations
- **Obey the four critical rules**: do not break existing behavior, study reference repos before coding, keep coverage ≥90%, and follow the documented development process (see `docs/DEVELOPMENT.md`).
- **Never create new `.md` files unless the user explicitly asks**: capture notes in the conversation or update existing Markdown files instead.
- **Front-load situational awareness**: run the CLI smoke suite (`pytest tests/integration/test_end_to_end.py`), gather coverage with `pytest --cov=ai_dev_agent --cov-report=term`, and review `docs/CHANGELOG.md` for current status before touching code.
- **Inspect references first**: pull patterns from `/Users/eg/Documents/aider`, `cline-1`, `codex`, `claude-code`, and `opencode`; summarize findings inline unless the user explicitly asks for a new `.md` file.
- **Design before implementation**: draft the approach in the conversation or update existing docs, and never create new `.md` design files unless the user explicitly requests them.
- **Work in tight loops**: write tests before code, keep functionality changes tiny, and record component status through the state store helper (`ai_dev_agent.core.utils.state.StateStore`) while leaving git commits to the user.

## Implementation Loop
- **Tests-first TDD**: add or extend tests in `tests/` before touching production modules; ensure the new tests fail for the expected reason before editing functionality.
- **Implement after failing tests**: once the tests define the behavior and fail, update production code in minimal increments to make them pass without breaking prior capabilities.
- **Commit handoff**: never run `git commit` or otherwise write to git history; prepare commit-ready diffs and message suggestions so the user can apply them when satisfied.
- **Continuous status updates**: log each milestone by calling the shared state store (e.g., `StateStore.append_history(...)`) so parallel efforts stay synchronized.
- **Judge and gate checks**: after features stabilize, run `python -m ai_dev_agent.judges.verify --feature "<feature>"`, `pytest --cov=ai_dev_agent --cov-fail-under=90`, targeted compatibility suites under `tests/compatibility/`, performance benchmarks, and the full test battery before concluding the task.
- **Documentation upkeep**: refresh existing docs such as `docs/CHANGELOG.md` when needed, but do not add new `.md` files unless the user explicitly asks.

## Priority Roadmap
- **CLI Simplification** (Priority #1) – ✅ **COMPLETE** - Simplified to flat command structure with natural language fallback and auto-detection of intent. Removed nested namespaces (`agent`, `plan`).
- **Memory System** (Priority #2) – study Aider’s `RepoMap` and Cline’s persistence model.
- **Multi-Agent Coordination** (Priority #3) – extract orchestration ideas from Cline and OpenCode.
- **Testing Infrastructure** (Priority #4) – mirror Aider’s large-suite discipline and enforce the 90 % threshold.

## Daily Rhythm
- **Morning**: pull latest changes, run the test suite, review `docs/CHANGELOG.md`, confirm daily goals.
- **During development**: revisit references, drive via tests, keep changes minimal, refresh status tracking.
- **End of day**: rerun full tests, record progress, draft commit guidance for the user (do not commit), log the next plan.

## Emergency Playbooks
- **Breakage**: stop, inspect `git status`/`git diff`, undo the faulty change set, restart with smaller steps.
- **Failing tests**: repair the regression before proceeding; guard compatibility explicitly.
- **Coverage drop**: pause feature work, add tests (manual or AI-assisted), and restore ≥90 %.

## Anti-Patterns To Avoid
- **Maze-of-fallback logic**: When resolving files, configuration, or CLI entrypoints, avoid layered "best guess" heuristics. Resolve a single authoritative location (or require the caller to provide one) and fail fast if it is missing. Silent fallbacks mask packaging and deployment bugs.
- **Heuristic prompt substitutions**: Do not invent inline prompt text when a file is absent—ship the markdown file in `prompts/system/` (and ensure it is included in package data) or stop with a `FileNotFoundError`.
- **Implicit behavioural switches**: Configuration like settings, prompt paths, or tooling should be explicit. If logic needs to diverge, wire it through a well-documented parameter rather than hidden fallbacks.
## Operating Claude Code & Codex CLI
- **Claude Code quick start**: instruct the agent to read `.devagent/ai_agent_instructions.md` and `docs/DEVELOPMENT.md`, then to execute the feature or bug task with reference checks and TDD. Example session opener:
  ```
  Read and follow the instructions in .devagent/ai_agent_instructions.md and docs/DEVELOPMENT.md. Then implement <task> following the process exactly.
  ```
- **Codex CLI quick start**: invoke `codex` with the same context either inline or via `--system`/`--context`. Typical direct command:
  ```
  codex "Read /Users/eg/Documents/Coding Agent/.devagent/ai_agent_instructions.md and /Users/eg/Documents/Coding Agent/docs/DEVELOPMENT.md. Implement <task>. Check /Users/eg/Documents/aider and /Users/eg/Documents/cline-1 before coding. Use TDD and keep coverage ≥90%."
  ```
- **Prompt templates**: reuse the prewritten blocks in `CLAUDE_CODEX_PROMPT.txt` for starting new work, continuing sessions, or resuming with implementation status context. They already spell out priority #1 (Work Planning Agent) and the reference paths.
- **Persistent session context**: configure `.devagent/session_context.md` (see `docs/USER_GUIDE.md`) to remind agents of active rules, priorities, references, and quality gates across sessions.

## Automation Support
- **`scripts/ai_develop.sh`** streamlines setup. Run `./scripts/ai_develop.sh "<task description>"` to:
  - Print the current implementation status snippet.
  - Assemble a rich context prompt emphasizing the must-read files, reference repos, process checklist, and quality requirements.
  - Store a timestamped context file under `/tmp/devagent_context_<timestamp>.md` for `codex --context ...`.
  - Refresh `.devagent/current_session.md` with the task, mandatory instructions, and reference paths.
- Use the generated context file directly with Codex CLI: `codex --context /tmp/devagent_context_<timestamp>.md "<task>"`.
- Claude operators can copy the displayed instructions verbatim to seed a compliant session.

## CLI Commands Reference

### Core Commands
- `devagent "<query>"` – Natural language query (direct execution, auto-detects intent)
- `devagent --plan "<query>"` – Query with LLM planning mode
- `devagent query "<prompt>"` – Explicit query command
- `devagent chat` – Interactive chat session with persistent context
- `devagent review <file> [--rule <rule_file>] [--json]` – Code/patch review

### Specialized Commands
- `devagent create-design "<feature>" [--context] [--output]` – Create design documents
- `devagent generate-tests "<feature>" [--coverage N] [--type unit|integration|all]` – Generate test files
- `devagent write-code "<design_file>" [--test-file]` – Write implementation from design

### Global Options
- `-v, -vv, -vvv` – Verbosity levels (info, debug, trace)
- `-q, --quiet` – Minimal output
- `--json` – JSON output format
- `--plan` – Enable planning mode for complex queries

**Example Usage**:
```bash
# Natural language (auto-detects intent)
devagent "create a design for user authentication"
devagent "generate tests for the auth module"

# Explicit specialized commands
devagent create-design "REST API" --context "CRUD endpoints" --output design.md
devagent generate-tests "API endpoints" --coverage 95 --type integration
devagent write-code design.md --test-file tests/test_api.py

# Code review
devagent review src/auth.py
devagent review changes.patch --rule rules/security.md --json

# Interactive session
devagent chat
```

**Features**:
- ✅ Flat command structure (no nested namespaces)
- ✅ Auto-detection of intent for complex queries
- ✅ Unified review command (handles files and patches)
- ✅ Standard CLI patterns (-v/-vv/-vvv for verbosity)
- ✅ Natural language fallback for unrecognized commands

## Verification Checklist
- Run `pytest --cov=ai_dev_agent --cov-fail-under=90` and any required compatibility suites before declaring the task complete; surface key results in the handoff.
- Run `python -m ai_dev_agent.judges.verify` (feature-scoped or full) until GPT-4, Claude, and a third model approve before completion.
- Scan recent history with `git log --oneline -5` for commit format compliance.
- Validate priority tasks progress by reviewing `docs/CHANGELOG.md`.

Always start by reading `.devagent/ai_agent_instructions.md`; this guide is a quick companion, not a replacement.
