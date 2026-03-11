# DevAgent AI Agent Guide

This guide covers the current DevAgent development surface. The supported machine orchestration
entrypoint is `devagent execute --request ... --artifact-dir ...`.

## Fail Fast Philosophy

- Surface issues immediately. Prefer explicit failure over silent fallback.
- Fix causes, not symptoms. Add tests that reproduce the bug before changing behavior.
- Keep the failure path visible once the fix lands.

## Core Expectations

- Run `bun run test` and `bun run typecheck` before and after meaningful changes.
- Keep changes small and test-backed.
- Do not document or depend on deprecated Hub integration paths.
- Treat `devagent execute` as the only public machine interface for Hub/Runner orchestration.

## Project Structure

```text
packages/
  cli/        # Terminal CLI entry point (bin: devagent)
  core/       # Types, config, events, approval gate, session management
  executor/   # SDK request execution mode for runner/hub orchestration
  engine/     # Task loop, agents, orchestration, review pipeline, plugins
  tools/      # Tool registry, builtins, LSP, MCP
  providers/  # LLM provider abstraction
  arkts/      # ArkTS lint support
models/       # LLM provider config files
prompts/      # Shared prompt templates
```

## Supported CLI Surfaces

- `devagent "<query>"`
- `devagent --plan "<query>"`
- `devagent chat`
- `devagent review <file> --rule <rule_file> [--json]`
- `devagent execute --request <request.json> --artifact-dir <path>`
- `devagent auth login|status|logout`

Any older workflow-stage runner code is internal compatibility machinery for `execute`, not a
public integration contract.

## Build & Test

```bash
bun install
bun run build
bun run test
bun run typecheck
```
