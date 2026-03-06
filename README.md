# DevAgent

> AI-powered CLI for development workflows. Interact with your codebase using natural language.

## Installation

```bash
git clone https://github.com/egavrin/devagent.git
cd devagent
bun install
bun run build
bun run install-cli
```

## Setup

Store your API key:
```bash
devagent auth login
```

Or set the environment variable:
```bash
export DEVAGENT_API_KEY="your-api-key"
```

## Usage

### One-shot queries

```bash
devagent "summarize this repository"
devagent "find all TODO comments"
devagent "explain how the config system works"
```

### Interactive chat session

```bash
devagent chat
```

### Plan mode (read-only analysis)

```bash
devagent --plan "how should we add user authentication?"
```

### Code review

```bash
# Rule-based patch review with structured JSON output
devagent review changes.patch --rule rules/security.md --json

# Free-form review via agent
devagent "review my last commit for issues"
```

### Provider and model selection

```bash
devagent --provider anthropic --model claude-sonnet-4-20250514 "explain this code"
devagent --provider ollama --model qwen2.5-coder "review main.ts"
```

## Project Structure

```
packages/
  cli/        # Terminal CLI entry point (bin: devagent)
  core/       # Types, config, events, approval gate, session management
  engine/     # TaskLoop, agents, orchestration, review pipeline, plugins
  tools/      # Tool registry, builtins, LSP, MCP support
  providers/  # LLM provider abstraction (Anthropic, OpenAI, Ollama, ChatGPT)
  arkts/      # ArkTS linter support
models/       # LLM provider config files (TOML)
prompts/      # Shared prompt templates
```

## Development

```bash
# Install dependencies
bun install

# Build all packages
bun run build

# Run tests
bun run test

# Type check
bun run typecheck

# Watch mode
bun run dev
```

See [AGENTS.md](AGENTS.md) for development philosophy and AI agent instructions.
