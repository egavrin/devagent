# packages/cli

This file provides guidance to AI agents working in this component.

## Purpose and Scope

- **component**: `packages/cli`
- **purpose**: Public `devagent` terminal entrypoint, argument parsing, prompt assembly, formatting, and CLI wiring into runtime, executor, providers, and ArkTS support.
- **primary languages/platforms**: TypeScript, Bun, Node.js

## Important Paths and Interfaces

```text
src/index.ts                 # Bun entrypoint
src/main.ts                  # CLI argument handling and command dispatch
src/auth.ts                  # Auth command surface
src/workflow-engine.ts       # CLI workflow execution wiring
src/prompts/                 # Prompt templates and prompt composition helpers
src/format.ts                # Human-facing output formatting
src/*test.ts                 # Component tests
package.json                 # Build/test scripts and bin mapping
```

## Build, Test, and Run

```bash
cd packages/cli
bun run build
bun run typecheck
bun run test
```

## Architecture and Workflows

- `src/index.ts` is a thin entrypoint that forwards to `main()` and reports failures through `extractErrorMessage`.
- `src/main.ts` owns the supported CLI surface. Keep it aligned with the repo root `README.md` and avoid reintroducing removed surfaces such as interactive chat or public plan mode.
- Prompt text lives in `src/prompts/*.md` and is copied into `dist/prompts/` during build. If prompt behavior changes, update the prompt assets and the nearby tests together.
- CLI behavior should delegate into workspace packages instead of reimplementing runtime or executor logic locally.
- The opt-in live validation harness lives under `scripts/live-validation.ts` and invokes the real CLI entrypoint from outside the package. When CLI behavior changes, keep that harness and the root README guidance aligned with the supported surface.

## Generated Files, Conventions, and Pitfalls

- Do not edit `dist/`; it is build output.
- Keep user-facing command docs synchronized with the actual command parser and tests.
- When changing review, execute, continuation, or formatting behavior, extend the nearest `src/*.test.ts` coverage instead of relying on manual validation alone.
- Preserve the current package boundary: `cli` may depend on `runtime`, `executor`, `providers`, and `arkts`, but it should not absorb their logic.

## Local Skills

- No component-local skill directory exists here. Use the shared repo skills from `.agents/skills`, especially `testing`, `review`, `tdd-workflow`, and `verification-checklist`.
