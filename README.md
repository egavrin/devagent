# DevAgent

A local coding agent centered on single-shot execution, review, and machine orchestration.

## Maturity

Public alpha. `devagent` is the first-party executor and the only production-grade executor path in
the current four-repo stack.

## Install

```bash
bun install
bun run install-cli
```

For the full local four-repo workflow stack, prefer the bootstrap flow documented in
[`../devagent-hub/README.md`](../devagent-hub/README.md) instead of wiring sibling repos by hand.

## Setup

Store your API key:

```bash
devagent auth login
```

## Examples

```bash
devagent "fix failing tests in the CLI"
devagent "review my last commit for issues"
devagent --resume <session-id> "finish the verification step"
devagent --continue "address the remaining review findings"
devagent review patch.diff --rule rules/security.md
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
devagent --provider chatgpt --model gpt-5.4 "review the last diff"
```

## Repository layout

```text
packages/
  cli/        # Terminal CLI entry point (bin: devagent)
  runtime/    # Consolidated runtime: core types/config, task loop, review, tools
  executor/   # SDK request execution mode for runner/hub orchestration
  providers/  # LLM provider abstraction (Anthropic, OpenAI, Ollama, ChatGPT)
  arkts/      # ArkTS linter support
models/       # LLM provider config files (TOML)
prompts/      # Shared prompt templates
```

## Supported CLI surface

- `devagent "<query>"`
- `devagent review <patch> --rule <rule_file> [--json]`
- `devagent execute --request <request.json> --artifact-dir <path>`
- `devagent auth login|status|logout`
- `--resume <session-id>`
- `--continue`

DevAgent no longer exposes chat/TUI mode, public plan mode, plugins, MCP, or internal checkpoints.
Sessions remain supported for continuation and debugging, and workspace rollback is delegated to Git.

## Development

```bash
bun install
bun run dev
bun run typecheck
bun run test
bun run check:oss
```

## Validated Flow

The current validated machine path is:

```text
devagent-hub -> devagent-runner -> devagent execute --request ... --artifact-dir ...
```

Live validation is currently exercised with `provider: chatgpt` and `model: gpt-5.4`.

Opt-in local validation commands:

```bash
bun run validate:live:smoke
bun run validate:live:full
bun run validate:live:scenario -- runtime-core-execute-triage
```

These runs expect sibling checkouts of `arkcompiler_ets_frontend` and
`arkcompiler_runtime_core`, ChatGPT auth configured via `devagent auth login`,
and a built ArkTS linter at `../arkcompiler_ets_frontend/ets2panda/linter/dist/tslinter.js`.

## Limitations

- the repo is public, but the workspace packages are not published to a registry yet
- the supported contributor path is the sibling checkout plus Hub bootstrap flow
- only the DevAgent executor path is production-grade; other executors remain experimental in Runner

See [AGENTS.md](AGENTS.md) for development philosophy and AI agent instructions.
