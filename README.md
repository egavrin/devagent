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

For the full local four-repo workflow stack, prefer the bootstrap flow documented in
[`../devagent-hub/README.md`](../devagent-hub/README.md) instead of wiring sibling repos by hand.

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

### Machine execution contract

```bash
devagent execute --request request.json --artifact-dir .devagent-runner/artifacts/task-123
```

`devagent execute` is the orchestration entrypoint used by `devagent-runner` and `devagent-hub`.
It consumes SDK `TaskExecutionRequest` payloads, emits normalized JSONL events on stdout, writes
the task artifact for the requested stage, and persists a machine-readable `result.json` in the
artifact directory.

During local multi-repo development the SDK packages are consumed through file dependencies from
`../devagent-sdk`, and `devagent-hub` reaches this entrypoint through `devagent-runner`.

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
  executor/   # SDK request execution mode for runner/hub orchestration
  engine/     # TaskLoop, agents, orchestration, review pipeline, plugins
  tools/      # Tool registry, builtins, LSP, MCP support
  providers/  # LLM provider abstraction (Anthropic, OpenAI, Ollama, ChatGPT)
  arkts/      # ArkTS linter support
  desktop/    # Desktop app work, separate from the validated runner/hub execution path
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

## Validated Flow

The current validated machine path is:

```text
devagent-hub -> devagent-runner -> devagent execute --request ... --artifact-dir ...
```

Live validation is currently exercised with `provider: chatgpt` and `model: gpt-5.4`.

See [AGENTS.md](AGENTS.md) for development philosophy and AI agent instructions.
