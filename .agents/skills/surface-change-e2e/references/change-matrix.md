# Surface Change Matrix

Use this table to pick likely impact files and the minimum checks for a public-surface change.

| Signal | Likely files | Minimum checks | Additional follow-up |
| --- | --- | --- | --- |
| CLI command parsing, flags, help, onboarding | `packages/cli/src/main.ts`, `packages/cli/src/commands.ts`, `packages/cli/src/*test.ts`, `README.md` | `cd packages/cli && bun run test`, `bun run typecheck` | `bun run check:oss`; inspect live-validation scenarios if command surface changed |
| Prompt text, transcript rendering, status line, TUI copy | `packages/cli/src/prompts/*`, `packages/cli/src/format.ts`, `packages/cli/src/transcript-*.ts`, nearby tests | `cd packages/cli && bun run test`, `bun run typecheck` | Update docs or screenshots only if user-facing wording or workflow changed |
| Runtime config, approval, sessions, task loop | `packages/runtime/src/core/*`, `packages/runtime/src/engine/*`, nearby tests | `cd packages/runtime && bun run test`, `bun run typecheck` | Pair with CLI tests if behavior is surfaced through the user entrypoint |
| README or `WORKFLOW.md` claim changes | `README.md`, `WORKFLOW.md`, `packages/cli/src/documentation.test.ts` | `bun run check:oss` | Update release matrix if the supported release surface changed |
| Release-facing CLI flow changes | CLI files plus `scripts/live-validation/**`, `.agents/skills/validate-user-surface/references/release-matrix.md` | `bun run check:oss`, `bun test scripts/live-validation/live-validation.test.ts` | Escalate to `live-validation-authoring` or `release-train` |

Escalate to `validate-user-surface` when the change affects install, auth, packaging, `devagent review`, or the documented `devagent execute` flow.
