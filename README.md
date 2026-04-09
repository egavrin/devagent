# DevAgent

Workflow-grade coding executor for staged software delivery. DevAgent turns a typed request into stage-specific artifacts through the fixed `devagent execute --request <file> --artifact-dir <dir>` contract, while still offering an interactive terminal UI for direct operator-driven work.

## Install

DevAgent requires Node.js 20+ or Bun 1.3+.

```bash
npm i -g @egavrin/devagent
```

Or run without installing:

```bash
npx @egavrin/devagent "fix failing tests"
```

On Ubuntu, do not rely on `apt install nodejs` for this project. Use Node 20 instead:

```bash
nvm install 20 && nvm use 20
```

If you prefer Bun, Bun 1.3+ is also supported:

```bash
bunx @egavrin/devagent "fix failing tests"
```

## Quick Start

```bash
# First-time setup for interactive use (provider, model, API key, subagents)
devagent setup

# Check your environment
devagent doctor

# Interactive coding
devagent "fix the bug in auth.ts"
devagent "add unit tests for the parser module"
devagent "review my last commit for issues"
```

## Staged Workflow

The primary DevAgent product surface is the staged executor contract:

```bash
devagent execute --request request.json --artifact-dir artifacts/
```

Lead with this canonical staged flow:

`design -> breakdown -> issue-generation -> implement -> review -> repair`

- `design` creates the design doc for the change.
- `breakdown` turns that design into small executable tasks.
- `issue-generation` converts the approved breakdown into executable issue specs.
- `implement` applies the requested code changes.
- `review` inspects the resulting workspace for defects.
- `repair` addresses review findings and closes the loop.

The workflow is a fixed supported stage set. Stages are chosen from the built-in `taskType` contract; users do not define new public stages or override stage semantics in config.

Stage prompts follow a fixed shape plus dynamic request context:

- Stage-specific behavior is code-defined by `taskType`.
- Each run still injects dynamic request context such as repo/work item metadata, summary, issue body, comments, changed file hints, requested skills, continuation context, and `extraInstructions`.
- This pass does not expose stage prompts as user-editable templates.

Canonical example:

```json
{
  "protocolVersion": "0.1",
  "taskId": "demo-design",
  "taskType": "design",
  "workspaceRef": {
    "id": "workspace-1",
    "name": "demo",
    "provider": "local",
    "primaryRepositoryId": "repo-1"
  },
  "repositories": [
    {
      "id": "repo-1",
      "workspaceId": "workspace-1",
      "alias": "primary",
      "name": "demo",
      "repoRoot": "/absolute/path/to/repo",
      "provider": "local"
    }
  ],
  "workItem": {
    "kind": "local-task",
    "externalId": "demo-design",
    "title": "Design the staged workflow docs refresh"
  },
  "execution": {
    "primaryRepositoryId": "repo-1",
    "repositories": [
      {
        "repositoryId": "repo-1",
        "alias": "primary",
        "sourceRepoPath": "/absolute/path/to/repo",
        "workBranch": "devagent/demo-design",
        "isolation": "git-worktree"
      }
    ]
  },
  "targetRepositoryIds": ["repo-1"],
  "executor": {
    "executorId": "devagent",
    "approvalMode": "full-auto"
  },
  "constraints": {
    "allowNetwork": true,
    "maxIterations": 12
  },
  "capabilities": {
    "canSyncTasks": true,
    "canCreateTask": true,
    "canComment": true,
    "canReview": true,
    "canMerge": true,
    "canOpenReviewable": true
  },
  "context": {
    "summary": "Design the docs-and-validation reframe for staged execute workflows."
  },
  "expectedArtifacts": ["design-doc"]
}
```

## Stage Matrix

| Stage | Behavior | Artifact | Role In Workflow |
|---------|-------------|-------------|-------------|
| `task-intake` | Readonly | `task-spec.md` | Capture goals, assumptions, and acceptance criteria before design work |
| `design` | Readonly | `design-doc.md` | Define architecture, interfaces, risks, and validation strategy |
| `breakdown` | Readonly, strict structured output | `breakdown-doc.json`, `breakdown-doc.md` | Turn the design into ordered, reviewable implementation slices |
| `issue-generation` | Readonly, strict structured output | `issue-spec.json`, `issue-spec.md` | Convert approved breakdown tasks into executable issue specs |
| `test-plan` | Readonly | `test-plan.md` | Define scenarios, regressions, and expected test outcomes |
| `triage` | Readonly | `triage-report.md` | Analyze impact area, risks, and immediate follow-up direction |
| `plan` | Readonly | `plan.md` | Produce an implementation plan for a narrower coding task |
| `implement` | Mutating | `implementation-summary.md` | Apply the requested change in the current workspace |
| `verify` | Readonly summary over external commands | `verification-report.md` | Summarize the outcome of caller-provided verification commands |
| `review` | Readonly | `review-report.md` | Inspect the current workspace for concrete defects |
| `repair` | Mutating | `final-summary.md` | Apply fixes for the current issue or review findings |
| `completion` | Readonly | `workflow-summary.md` | Summarize completed work, key decisions, and remaining risks |

The canonical public example flow is `design -> breakdown -> issue-generation -> implement -> review -> repair`, but the full fixed stage matrix above is supported by the executor contract.

Devagent API gateway:

```bash
DEVAGENT_API_KEY=ilg_your_gateway_key \
devagent --provider devagent-api --model cortex "fix failing tests"
```

## Features

- **Fixed staged workflow** — code-defined `taskType` stages with stage-specific artifact contracts
- **Machine orchestration** — `devagent execute` for CI/CD and runner integration
- **Multi-provider** — Anthropic, OpenAI, Devagent API, DeepSeek, OpenRouter, Ollama, ChatGPT, GitHub Copilot
- **Tool use** — reads/writes files, runs commands, searches code, git operations
- **Subagents** — spawns specialized agents (explore, review, architect) with configurable models
- **Session persistence** — resume previous sessions with `--resume` or `--continue`
- **Interactive TUI** — Ink-based terminal UI with streaming output, tool display, and plan tracking
- **Code review** — rule-based patch review with `devagent review`

## Usage

```bash
# Interactive TUI
devagent

# Single query
devagent "explain the config system"

# Query from file
devagent -f prompt.md

# Top-level help
devagent help

# Provider/model override
devagent --provider openai --model gpt-5.4 "optimize this function"

# Resume a session
devagent --resume <session-id-or-unique-prefix>
devagent --continue    # resume most recent

# Interactive safety modes
devagent --mode autopilot     # default: auto-approve everything
devagent --mode default       # opt into guarded prompts

# Code review
devagent review patch.diff --rule rules/security.md --json
```

## Commands

| Command | Description |
|---------|-------------|
| `devagent help` | Show top-level help |
| `devagent setup` | Guided global configuration wizard |
| `devagent doctor` | Check environment and dependencies |
| `devagent config get/set/path` | Inspect or edit global config directly |
| `devagent install-lsp` | Install LSP servers for code intelligence |
| `devagent update` | Update to latest version |
| `devagent completions <shell>` | Generate shell completions (bash/zsh/fish) |
| `devagent auth login/status/logout` | Manage provider credentials |
| `devagent sessions` | List recent sessions |
| `devagent execute --request <file> --artifact-dir <dir>` | Execute an SDK request and write artifacts |
| `devagent version` | Show version |

`devagent auth logout` also supports scriptable removal:

```bash
devagent auth logout chatgpt
devagent auth logout --all
```

Public machine contract:

```bash
devagent execute --request request.json --artifact-dir artifacts/
```

Interactive CLI and TUI usage remain supported public surfaces for operator-driven work, but the staged `execute` flow above is the primary machine-facing workflow story.

## Configuration

Global config: `~/.config/devagent/config.toml`

```toml
provider = "anthropic"
model = "claude-sonnet-4-20250514"

[safety]
mode = "autopilot"

[budget]
max_iterations = 0

[subagents.agent_model_overrides]
explore = "claude-haiku-4-20250414"

[subagents.agent_reasoning_overrides]
general = "medium"
explore = "low"
reviewer = "high"
architect = "high"
```

Project instructions are optional.
Create `AGENTS.md` manually in a repository when you want repo-specific guidance for DevAgent.

Devagent API gateway config:

The deployed gateway is OpenAI-compatible under the hood, but in DevAgent you should use the built-in `devagent-api` provider with model `cortex`, not a direct upstream provider configuration.

```toml
provider = "devagent-api"
model = "cortex"
```

```bash
export DEVAGENT_API_KEY=ilg_your_gateway_key
devagent doctor
```

## Shell Completions

```bash
# Bash
eval "$(devagent completions bash)"

# Zsh
eval "$(devagent completions zsh)"

# Fish
devagent completions fish > ~/.config/fish/completions/devagent.fish
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `OPENAI_API_KEY` | OpenAI API key |
| `DEVAGENT_API_KEY` | Devagent API gateway key (virtual key starting with `ilg_`) |
| `DEEPSEEK_API_KEY` | DeepSeek API key |
| `OPENROUTER_API_KEY` | OpenRouter API key |
| `DEVAGENT_PROVIDER` | Default provider |
| `DEVAGENT_MODEL` | Default model |

## Requirements

- Node.js >= 20 or Bun >= 1.3
- Git

## Development

```bash
git clone https://github.com/egavrin/devagent.git
cd devagent
bun install
bun run build
bun run test
bun run install-cli
```

## License

MIT
