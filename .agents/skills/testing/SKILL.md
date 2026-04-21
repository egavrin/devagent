---
name: testing
description: Keep DevAgent changes paired with focused verification and only the necessary test updates.
---

# Testing

When a task changes behavior, update or add the smallest relevant test coverage.

## Rules

- Use repository-native commands such as `bun run test`, `bun run typecheck`, and `bun run build`.
- Prefer updating an existing nearby test before creating a brand new suite.
- If the task is documentation-only, do not force code or test edits. State that no test changes were needed.
- Use `references/path-check-matrix.md` to choose the smallest acceptable verification set from the changed paths.

## Package hints

- `packages/runtime`: shared contracts, config, task loop, review, tools, skills
- `packages/executor`: machine execution mode
- `packages/cli`: human entrypoint and prompt assembly
- `packages/providers`: LLM provider adapters
