# DevAgent

This file provides guidance to AI agents working in this repository.

## Repository Identity

- **name**: `devagent`
- **purpose**: Bun/Turbo monorepo for the DevAgent CLI, shared runtime, machine executor, provider adapters, and ArkTS integration.
- **primary languages/platforms**: TypeScript, Bun, Node.js 20+, Turbo

## Repository Layout

```text
packages/
  cli/        # Public `devagent` terminal entrypoint and prompt assembly
  runtime/    # Shared core types/config, task loop, review pipeline, tools, skill loading
  executor/   # `devagent execute` SDK request execution and artifact/event contract
  providers/  # Provider adapters and API-specific request shaping
  arkts/      # ArkTS lint and diagnostic integration
models/       # Provider model registry TOML files
prompts/      # Shared prompt templates
scripts/      # Repo checks such as OSS surface validation
.agents/skills/  # Repo-local Codex-compatible skills
.devagent/workspaces/  # Runner-managed worktrees and generated mirrors; not canonical docs
```

## Build, Test, and Run

```bash
bun install
bun run build
bun run typecheck
bun run test
bun run check:oss
bun run test:live-validation

# package-focused work
cd packages/runtime && bun run test
cd packages/executor && bun run test
cd packages/cli && bun run test
cd packages/providers && bun run test
cd packages/arkts && bun run test

# install the local CLI
bun run install-cli
```

## Architecture and Workflows

- The supported machine orchestration entrypoint is `devagent execute --request <request.json> --artifact-dir <path>`. Treat this as the only public executor contract.
- The validated cross-repo flow is `devagent-hub -> devagent-runner -> devagent execute`. Keep docs and behavior aligned with that path, and avoid reintroducing deprecated Hub or workflow-stage surfaces.
- `packages/cli` assembles prompts, repository context, and human-facing command behavior, then delegates into `@devagent/runtime`, `@devagent/executor`, `@devagent/providers`, and `@devagent/arkts`.
- `packages/runtime` is the shared platform layer. Its `src/core`, `src/engine`, and `src/tools` directories are internal subdivisions of one package, not separate workspace packages.
- `packages/executor` validates SDK requests, builds task queries, resolves requested skills, runs the agent loop, and writes normalized artifacts and events. Contract drift here is high risk and should be test-backed in `packages/executor/src/index.test.ts`.
- Repo-local skills live under `.agents/skills/` and are discovered by the runtime skill loader. The similarly named `packages/runtime/src/core/skills/` directory is product source code for skill loading, not a repo-maintenance skill directory.
- `.devagent/workspaces/` contains runner-managed worktrees and stale documentation mirrors. Do not update guidance there unless the task is specifically about the worktree generation flow.
- Opt-in live runtime validation is driven by `scripts/live-validation.ts` and the `validate:live:*` root scripts. It exercises the real CLI and `devagent execute` surfaces against sibling `arkcompiler_*` repos and is not part of the default `bun run test` path.
- When adding or removing public commands or live-validation scenarios, update `.agents/skills/validate-user-surface/references/release-matrix.md` so the release checklist stays aligned with the supported user surface.

## Conventions and Pitfalls

- Follow the fail-fast philosophy: surface errors explicitly, fix root causes, and avoid silent fallbacks or defensive guards that hide breakage.
- Run `bun run typecheck` and `bun run test` before and after meaningful code changes. Run `bun run check:oss` when changing public docs, contributor workflow, or package metadata.
- Keep changes small and test-backed. If behavior changes, start with a failing or expanded test near the affected code.
- Prefer updating existing docs over creating new Markdown files, unless the task explicitly requires new documentation. Package-level `AGENTS.md` files in this repo are maintained documentation and may be updated when the component reality changes.
- Keep generated artifacts inside runner-managed artifact directories, not repo-tracked paths.
- Do not document unsupported executor parity, deprecated chat/TUI/plan surfaces, or any non-DevAgent executor story as production-ready.
- Treat `.agents/skills/` as the only maintained repo-local skill tree.

## Local Skills

- `.agents/skills/add-feature-e2e`: use for end-to-end feature work spanning types, implementation, wiring, and verification.
- `.agents/skills/tdd-workflow`: use when behavior changes should follow the repo’s strict tests-first loop.
- `.agents/skills/testing`: use to choose the smallest relevant test updates and verification scope.
- `.agents/skills/review`: use for code review passes focused on fail-fast behavior, module boundaries, tests, and contract drift.
- `.agents/skills/review-rule`: use when authoring or updating rule files for `devagent review`.
- `.agents/skills/debug-test-failure`: use when diagnosing failing test suites in this monorepo.
- `.agents/skills/security-checklist`: use when a change touches command execution, credentials, artifacts, or provider/auth flows.
- `.agents/skills/simplify`: use when removing unnecessary abstraction or dead code.
- `.agents/skills/execute-contract`: use when touching `packages/executor`, executor-facing docs, or any behavior that affects artifact/event/result expectations.
- `.agents/skills/oss-surface-guard`: use when editing public docs, contributor workflow, or package metadata that must stay within the supported OSS surface.
- `.agents/skills/verification-checklist`: use before finalizing substantial changes to run the repo’s verification gate.
- `.agents/skills/commit`: use only when the user explicitly asks for a commit or commit-message help.
