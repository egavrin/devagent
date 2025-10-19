# DevAgent AI Agent Guide

This guide condenses the working rules, launch instructions, and helper tooling for Claude Code, Codex CLI, and any DevAgent-based automation. It combines the authoritative rules in `.devagent/ai_agent_instructions.md`, the operator guide in `docs/HOW_TO_USE_AI_AGENTS.md`, the context builder in `scripts/ai_develop.sh`, and the copy-ready prompts in `CLAUDE_CODEX_PROMPT.txt`.

## Core Expectations
- **Obey the four critical rules**: do not break existing behavior, study reference repos before coding, keep coverage ≥90%, and follow the documented development process (`docs/DEVELOPMENT_PROCESS.md`, `docs/SAFE_IMPLEMENTATION_PLAN.md`).
- **Never create new `.md` files unless the user explicitly asks**: capture notes in the conversation or update existing Markdown files instead.
- **Front-load situational awareness**: run `pytest tests/test_backward_compatibility.py`, gather coverage with `pytest --cov=ai_dev_agent --cov-report=term`, and read `docs/IMPLEMENTATION_STATUS.md` before touching code.
- **Inspect references first**: pull patterns from `/Users/eg/Documents/aider`, `cline-1`, `codex`, `claude-code`, and `opencode`; summarize findings inline unless the user explicitly asks for a new `.md` file.
- **Design before implementation**: draft the approach in the conversation or update existing docs, and never create new `.md` design files unless the user explicitly requests them.
- **Work in tight loops**: write tests before code, keep functionality changes tiny, and update component status via `python -m ai_dev_agent.status.update` while leaving git commits to the user.

## Implementation Loop
- **Tests-first TDD**: add or extend tests in `tests/` before touching production modules; ensure the new tests fail for the expected reason before editing functionality.
- **Implement after failing tests**: once the tests define the behavior and fail, update production code in minimal increments to make them pass without breaking prior capabilities.
- **Commit handoff**: never run `git commit` or otherwise write to git history; prepare commit-ready diffs and message suggestions so the user can apply them when satisfied.
- **Continuous status updates**: log each milestone with the status updater module so parallel efforts stay synchronized.
- **Judge and gate checks**: after features stabilize, run `python -m ai_dev_agent.judges.verify --feature "<feature>"`, `pytest --cov=ai_dev_agent --cov-fail-under=90`, targeted compatibility suites under `tests/compatibility/`, performance benchmarks, and the full test battery before concluding the task.
- **Documentation upkeep**: refresh existing docs such as `docs/IMPLEMENTATION_STATUS.md` when needed, but do not add new `.md` files unless the user explicitly asks.

## Priority Roadmap
- **Work Planning Tools** (Priority #1) – ✅ **COMPLETE** - Fully integrated into CLI with `devagent plan` commands (create, list, show, next, start, complete, delete). See `docs/design/work_planning_design.md` for usage.
- **Memory System** (Priority #2) – study Aider’s `RepoMap` and Cline’s persistence model.
- **Multi-Agent Coordination** (Priority #3) – extract orchestration ideas from Cline and OpenCode.
- **Testing Infrastructure** (Priority #4) – mirror Aider’s large-suite discipline and enforce the 90 % threshold.

## Daily Rhythm
- **Morning**: pull latest changes, run the test suite, review `docs/IMPLEMENTATION_STATUS.md`, confirm daily goals.
- **During development**: revisit references, drive via tests, keep changes minimal, refresh status tracking.
- **End of day**: rerun full tests, record progress, draft commit guidance for the user (do not commit), log the next plan.

## Emergency Playbooks
- **Breakage**: stop, inspect `git status`/`git diff`, undo the faulty change set, restart with smaller steps.
- **Failing tests**: repair the regression before proceeding; guard compatibility explicitly.
- **Coverage drop**: pause feature work, add tests (manual or AI-assisted), and restore ≥90 %.

## Operating Claude Code & Codex CLI
- **Claude Code quick start**: instruct the agent to read `.devagent/ai_agent_instructions.md`, `docs/DEVELOPMENT_PROCESS.md`, and `docs/SAFE_IMPLEMENTATION_PLAN.md`, then to execute the feature or bug task with reference checks and TDD. Example session opener:
  ```
  Read and follow the instructions in .devagent/ai_agent_instructions.md, docs/DEVELOPMENT_PROCESS.md, and docs/SAFE_IMPLEMENTATION_PLAN.md. Then implement <task> following the process exactly.
  ```
- **Codex CLI quick start**: invoke `codex` with the same context either inline or via `--system`/`--context`. Typical direct command:
  ```
  codex "Read /Users/eg/Documents/Coding Agent/.devagent/ai_agent_instructions.md, /Users/eg/Documents/Coding Agent/docs/DEVELOPMENT_PROCESS.md, and /Users/eg/Documents/Coding Agent/docs/SAFE_IMPLEMENTATION_PLAN.md. Implement <task>. Check /Users/eg/Documents/aider and /Users/eg/Documents/cline-1 before coding. Use TDD and keep coverage ≥90%."
  ```
- **Prompt templates**: reuse the prewritten blocks in `CLAUDE_CODEX_PROMPT.txt` for starting new work, continuing sessions, or resuming with implementation status context. They already spell out priority #1 (Work Planning Agent) and the reference paths.
- **Persistent session context**: configure `.devagent/session_context.md` (see `docs/HOW_TO_USE_AI_AGENTS.md`) to remind agents of active rules, priorities, references, and quality gates across sessions.

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
- `devagent "<query>"` – Natural language query (direct execution)
- `devagent --plan "<query>"` – Query with LLM planning mode
- `devagent query "<prompt>"` – Explicit query command
- `devagent review [options]` – Code review functionality
- `devagent shell` – Interactive shell session
- `devagent diagnostics` – Show session diagnostics

### Work Planning Commands (Priority #1 - COMPLETE)
- `devagent plan create "<goal>" [--context]` – Create new work plan
- `devagent plan list` – List all work plans
- `devagent plan show <id>` – Display plan details (markdown)
- `devagent plan next <id>` – Get next task (respects dependencies)
- `devagent plan start <id> <task_id>` – Mark task as started
- `devagent plan complete <id> <task_id>` – Mark task as completed
- `devagent plan delete <id>` – Delete a work plan

**Example Workflow**:
```bash
# Create a plan
devagent plan create "Implement REST API" --context "CRUD endpoints for blog"

# View the plan
devagent plan show 296f3e

# Execute tasks in order
devagent plan next 296f3e           # Get: Task 1
devagent plan start 296f3e abc123   # Start Task 1
devagent plan complete 296f3e abc123 # Complete Task 1

devagent plan next 296f3e           # Get: Task 2 (dependency met)
# ... continue until 100%
```

**Features**:
- ✅ Dependency-aware task ordering
- ✅ Priority-based selection (Critical > High > Medium > Low)
- ✅ Progress tracking with completion percentage
- ✅ Markdown export for readable summaries
- ✅ Persistent storage (~/.devagent/plans/)
- ✅ Partial ID matching for convenience

**Documentation**: `docs/design/work_planning_design.md`

## Verification Checklist
- Run `pytest --cov=ai_dev_agent --cov-fail-under=90` and any required compatibility suites before declaring the task complete; surface key results in the handoff.
- Run `python -m ai_dev_agent.judges.verify` (feature-scoped or full) until GPT-4, Claude, and a third model approve before completion.
- Confirm reference analysis artifacts exist via `grep -r "reference" docs/reference_analysis/`.
- Scan recent history with `git log --oneline -5` for commit format compliance.
- Validate priority tasks progress by reviewing `docs/IMPLEMENTATION_STATUS.md`.

Always start by reading `.devagent/ai_agent_instructions.md`; this guide is a quick companion, not a replacement.
