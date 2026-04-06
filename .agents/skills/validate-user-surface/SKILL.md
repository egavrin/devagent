---
name: validate-user-surface
description: Comprehensive pre-release validation for DevAgent user-facing surfaces. Use when Codex needs to run or expand live end-to-end checks across npm packaging, install and upgrade flows, README and help-text accuracy, interactive TUI behavior, single-shot CLI behavior, auth and config commands, `devagent review`, and the public `devagent execute --request --artifact-dir` contract across multiple real providers.
---

# Validate User Surface

Treat the repository like a release candidate. Prefer live execution against built artifacts, isolated HOME directories, temp workspaces, and publishable bundles over source inspection or mocked success.

## Start Here

- Read `README.md`, `package.json`, `scripts/bundle.ts`, `scripts/smoke-publish-bundle.ts`, `scripts/live-validation.ts`, and the scenario manifests under `scripts/live-validation/scenarios/`.
- Treat the root README, `devagent help`, and the generated `dist/package.json` as the public contract.
- Read `references/release-matrix.md` before planning coverage or writing the final report.
- Record the exact command, provider, model, exit code, and observed behavior for every user-facing check.

## Core Rules

- Run live checks for user-facing behavior. Do not count unit tests or code reading as release validation.
- Use isolated temp homes and temp repos. Do not reuse the operator's real `~/.config/devagent`.
- Prefer the publish bundle for install and packaging checks. Validate the developer CLI separately only when comparing dev-versus-publish behavior.
- Treat missing provider credentials or missing external dependencies as validation gaps, not silent skips.
- Do not publish to npm unless the user explicitly asks.
- Keep the public executor contract limited to `devagent execute --request <file> --artifact-dir <dir>`.

## Workflow

1. Build the release candidate and run the built-in gates first.

```bash
bun install
bun run build
bun run typecheck
bun run test
bun run check:oss
bun run build:publish
bun run test:bundle-smoke
bun run test:live-validation
bun run validate:live:full
```

2. Create isolated homes and disposable workspaces for each install, auth, TUI, and query-flow pass.
3. Use `cd dist && npm pack` to create a publishable tarball, then validate install and launch paths from that artifact.
4. Exercise documented install and launch paths live: tarball install, `npx`, `bunx`, bundled bootstrap, and linked local CLI when helpful.
5. Cover the provider matrix from `references/release-matrix.md`. Prefer every documented provider. If full coverage is impossible, call out each unvalidated provider explicitly.
6. Run both single-shot CLI and interactive TUI flows. Use a PTY for TUI checks and verify slash-command behavior in addition to a real task run.
7. Reconcile docs and help text with observed behavior. Fix the code or docs rather than normalizing drift.
8. End with a release-style report that separates passed, failed, and blocked surfaces.

## Mandatory Surfaces

- Packaging and install: `bun run build:publish`, `bun run test:bundle-smoke`, `npm pack` from `dist/`, tarball install, uninstall and reinstall, Node 20 bootstrap, and upgrade behavior.
- Docs and metadata: README install snippets, quick start, provider list, command list, environment variables, `WORKFLOW.md` claims, copied `dist/README.md`, and generated `dist/package.json`.
- CLI basics: `devagent help`, `version`, `doctor`, `configure`, `config get/set/path`, `completions`, `sessions`, `--resume`, `--continue`, `--provider`, `--model`, and `-f`.
- Auth: `devagent auth login/status/logout` for API-key providers and device-code providers in isolated homes.
- Query execution: interactive TUI, single-shot query execution, quiet and non-TTY behavior, `devagent review`, and `devagent execute`.
- Provider coverage: Anthropic, OpenAI, Devagent API, DeepSeek, OpenRouter, Ollama, ChatGPT, and GitHub Copilot when credentials or local services are available.

## Reporting

- Summarize by surface: packaging, install, docs, CLI, TUI, auth, review, execute, and provider matrix.
- For each failure, include the command, environment, observed behavior, expected behavior, and release impact.
- Mark any unvalidated surface as a release risk.
