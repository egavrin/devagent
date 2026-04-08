---
name: surface-change-e2e
description: Handle user-visible DevAgent changes when a request says things like update CLI behavior, fix help text drift, adjust prompt output, or keep runtime, docs, and validation aligned.
triggers:
  - update CLI behavior
  - help text drift
  - prompt rendering
  - session UX
  - command help
paths:
  - packages/cli
  - packages/runtime
  - README.md
  - WORKFLOW.md
examples:
  - update CLI behavior and keep docs aligned
  - fix help text drift after a runtime change
---

# Surface Change End-to-End

Use this skill when the request sounds like “update CLI behavior”, “fix help text drift”, “adjust prompt output”, or “keep runtime, docs, and validation aligned” across `packages/cli`, `packages/runtime`, prompt assets, command help, transcript rendering, sessions, approval or config UX, or README and `WORKFLOW.md` claims.

Read `AGENTS.md`, then use `references/change-matrix.md` to scope the work.

## Workflow

1. Identify the changed surface before editing:
   - command parser or help text
   - prompt assets or transcript formatting
   - runtime config, approval, session, or task-loop behavior
   - docs or release-facing claims
2. Update the nearest tests first.
   - Prefer package-local tests in `packages/cli/src/*.test.ts` or `packages/runtime/src/**/*.test.ts`.
   - If the change is doc-only, state that no executable tests changed.
3. Reconcile surface drift after the code change.
   - Check README, `WORKFLOW.md`, command help, prompt assets, and documentation tests together.
   - Keep the public story aligned with the supported DevAgent surface.
4. Decide whether validation coverage changes.
   - If public behavior, command inventory, or release-critical flows changed, inspect `scripts/live-validation/` and `release-matrix.md`.
   - If no scenario or matrix update is needed, say why in the final summary.
5. Run verification in two tiers.
   - First run the minimum checks from `references/change-matrix.md`.
   - Then widen to `bun run check:oss`, bundle smoke, or live validation if the change is release-critical.

## Escalate

- Use `execute-contract` if the change touches `packages/executor` or the `devagent execute` contract.
- Use `live-validation-authoring` if a scenario needs to be added, removed, or rewritten.
- Use `release-train` for packaging, publish, install, or pre-release work.

## Red Flags

- Updating `README.md` without checking `packages/cli/src/documentation.test.ts`.
- Changing command behavior without checking help text or prompt assets.
- Treating live validation as optional when the public contract changed materially.
