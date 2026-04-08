# Path Check Matrix

Use the changed files to pick the smallest acceptable verification set.

| Changed paths | Minimum local verification | Add these when needed |
| --- | --- | --- |
| `packages/cli/src/*.ts`, `packages/cli/src/prompts/*` | `cd packages/cli && bun run test`, `bun run typecheck` | `bun run check:oss` if help/docs/user-facing behavior changed |
| `packages/runtime/src/**/*.ts` | `cd packages/runtime && bun run test`, `bun run typecheck` | `bun run check:oss` if public behavior changed through CLI or docs |
| `packages/providers/src/*.ts`, `models/*.toml` | `cd packages/providers && bun run test`, `bun run typecheck` | Provider smoke or live validation if auth/proxy/default-provider behavior changed |
| `packages/executor/src/*.ts` | `cd packages/executor && bun run test`, `bun run typecheck` | `bun run check:oss` and `execute-contract` if machine behavior or docs changed |
| `scripts/live-validation/**` | `bun test scripts/live-validation/live-validation.test.ts`, `bun run typecheck` | `bun run validate:live:scenario -- <id>` or broader live validation when scenario semantics changed |
| `README.md`, `WORKFLOW.md`, package metadata, `scripts/check-oss.mjs` | `bun run check:oss` | `bun run test` and `bun run typecheck` if docs describe executable behavior |
| Packaging or publish files such as `scripts/bundle.ts`, `scripts/smoke-publish-bundle.ts`, root `package.json` | `bun run build`, `bun run test:bundle-smoke`, `bun run typecheck` | `release-train` and `validate-user-surface` for pre-release or install-flow changes |

When a change spans multiple rows, combine the minimum checks from each row and then prefer one root command if it is cheaper than many narrow commands.
