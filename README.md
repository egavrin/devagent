# DevAgent

AI-powered coding agent for the terminal. Reads your codebase, writes code, runs commands, and iterates — all from a single prompt.

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
# First-time setup (provider, model, API key, subagents)
devagent configure

# Check your environment
devagent doctor

# Start coding
devagent "fix the bug in auth.ts"
devagent "add unit tests for the parser module"
devagent "review my last commit for issues"
```

Devagent API gateway:

```bash
DEVAGENT_API_KEY=ilg_your_gateway_key \
devagent --provider devagent-api --model cortex "fix failing tests"
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
# Interactive TUI
devagent

# Single query
devagent "explain the config system"

# Query from file
devagent -f prompt.md

# Top-level help
devagent help

# Provider/model override
devagent --provider openai --model gpt-4.1 "optimize this function"

# Resume a session
devagent --resume <session-id-or-unique-prefix>
devagent --continue    # resume most recent

# Interactive safety modes
devagent --mode default       # low-noise default for daily coding
devagent --mode autopilot     # auto-approve everything

# Code review
devagent review patch.diff --rule rules/security.md --json
```

## Commands

| Command | Description |
|---------|-------------|
| `devagent help` | Show top-level help |
| `devagent configure` | Guided global configuration wizard |
| `devagent doctor` | Check environment and dependencies |
| `devagent config get/set/path` | Inspect or edit global config directly |
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

[safety]
mode = "default"

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
