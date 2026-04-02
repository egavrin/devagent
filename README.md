# DevAgent

AI-powered coding agent for the terminal. Reads your codebase, writes code, runs commands, and iterates — all from a single prompt.

## Install

```bash
npm i -g @egavrin/devagent
```

Or run without installing:

```bash
npx @egavrin/devagent "fix failing tests"
```

## Quick Start

```bash
# First-time setup (provider, model, API key, subagents)
devagent setup

# Check your environment
devagent doctor

# Start coding
devagent "fix the bug in auth.ts"
devagent "add unit tests for the parser module"
devagent "review my last commit for issues"
```

## Features

- **Multi-provider** — Anthropic, OpenAI, Devagent API, DeepSeek, OpenRouter, Ollama, ChatGPT, GitHub Copilot
- **Tool use** — reads/writes files, runs commands, searches code, git operations
- **Subagents** — spawns specialized agents (explore, review, architect) with configurable models
- **Session persistence** — resume previous sessions with `--resume` or `--continue`
- **Interactive TUI** — Ink-based terminal UI with streaming output, tool display, and plan tracking
- **Code review** — rule-based patch review with `devagent review`
- **Machine orchestration** — `devagent execute` for CI/CD integration

## Usage

```bash
# Interactive mode
devagent

# Single query
devagent "explain the config system"

# Query from file
devagent -f prompt.md

# Provider/model override
devagent --provider openai --model gpt-4.1 "optimize this function"

# Resume a session
devagent --resume <session-id>
devagent --continue    # resume most recent

# Approval modes
devagent --suggest     # ask before writing files (default)
devagent --auto-edit   # auto-approve file writes
devagent --full-auto   # auto-approve everything

# Code review
devagent review patch.diff --rule rules/security.md --json
```

## Commands

| Command | Description |
|---------|-------------|
| `devagent setup` | Interactive first-time setup |
| `devagent init` | Initialize project config (`.devagent/` + `AGENTS.md`) |
| `devagent doctor` | Check environment and dependencies |
| `devagent config get/set/path` | Manage configuration |
| `devagent update` | Update to latest version |
| `devagent completions <shell>` | Generate shell completions (bash/zsh/fish) |
| `devagent auth login/status/logout` | Manage provider credentials |
| `devagent sessions` | List recent sessions |
| `devagent version` | Show version |

Public machine contract:

```bash
devagent execute --request request.json --artifact-dir artifacts/
```

## Configuration

Global config: `~/.config/devagent/config.toml`

```toml
provider = "anthropic"
model = "claude-sonnet-4-20250514"

[approval]
mode = "suggest"

[budget]
max_iterations = 30

[subagents.agent_model_overrides]
explore = "claude-haiku-4-20250414"

[subagents.agent_reasoning_overrides]
general = "medium"
explore = "low"
reviewer = "high"
architect = "high"
```

Project config: `.devagent/instructions.md` and `AGENTS.md` in your repo root.

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
| `DEVAGENT_API_KEY` | Devagent API gateway key, and generic API key fallback for the default provider |
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
