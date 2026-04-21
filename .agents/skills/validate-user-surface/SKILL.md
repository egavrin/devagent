---
name: validate-user-surface
description: Comprehensive pre-release validation for DevAgent user-facing surfaces. Use when Codex needs to run or expand live end-to-end checks across npm packaging, install and upgrade flows, README and help-text accuracy, interactive TUI behavior, single-shot CLI behavior, auth and config commands, `devagent review`, and the public `devagent execute --request --artifact-dir` contract across multiple real providers.
---

# Validate User Surface

Treat the repository like a release candidate. Prefer live execution against built artifacts, isolated HOME directories, temp workspaces, and publishable bundles over source inspection or mocked success.

## Start Here

- Read `README.md`, `package.json`, `scripts/bundle.ts`, `scripts/smoke-publish-bundle.ts`, and the relevant validation helper under `scripts/live-validation/`.
- Treat the root README, `devagent help`, and the generated `dist/package.json` as the public contract.
- Read `references/release-matrix.md` before planning coverage or writing the final report.
- Record the exact command, provider, model, exit code, and observed behavior for every user-facing check.

## Core Rules

- Run live checks for user-facing behavior. Do not count unit tests or code reading as release validation.
- Use isolated temp homes and temp repos. Do not reuse the operator's real `~/.config/devagent`.
- Environment variables are not the only credential source. Run `bun run validate:live:provider-smoke` before marking providers blocked just because API-key env vars are unset.
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
bun run validate:live:provider-smoke
bun run validate:live:tui
```

2. Create isolated homes and disposable workspaces for each install, auth, TUI, and query-flow pass. When provider credentials exist in the local DevAgent credential store, copy only the required non-expired credentials into those isolated homes rather than running against the operator's real HOME.
3. Use `cd dist && npm pack` to create a publishable tarball, then validate install and launch paths from that artifact.
4. Exercise documented install and launch paths live: tarball install, `npx`, `bunx`, bundled bootstrap, and linked local CLI when helpful.
5. Cover the provider matrix from `references/release-matrix.md`. Prefer every documented provider. If full coverage is impossible, call out each unvalidated provider explicitly.
6. Run both single-shot CLI and interactive TUI flows. Use a PTY for TUI checks and verify slash-command behavior in addition to a real task run.
7. Reconcile docs and help text with observed behavior. Fix the code or docs rather than normalizing drift.
8. End with a release-style report that separates passed, failed, and blocked surfaces.

## Mandatory Surfaces

- Packaging and install: `bun run build:publish`, `bun run test:bundle-smoke`, `npm pack` from `dist/`, tarball install, uninstall and reinstall, Node 20 bootstrap help, installed-runtime session startup, and upgrade behavior.
- Docs and metadata: README install snippets, quick start, provider list, command list, environment variables, `WORKFLOW.md` claims, copied `dist/README.md`, and generated `dist/package.json`.
- CLI basics: `devagent help`, `version`, `doctor`, `configure`, `config get/set/path`, `completions`, `sessions`, `--resume`, `--continue`, `--provider`, `--model`, and `-f`.
- Auth: `devagent auth login/status/logout` for API-key providers and device-code providers in isolated homes.
- Query execution: interactive TUI, single-shot query execution, quiet and non-TTY behavior, `devagent review`, and `devagent execute`.
- Provider coverage: Anthropic, OpenAI, Devagent API, DeepSeek, OpenRouter, Ollama, ChatGPT, and GitHub Copilot when credentials or local services are available.

## Credential And Bootstrap Notes

- Use `bun run validate:live:provider-smoke` to discover locally stored credentials and local services; it reports per-provider pass/block status.
- For raw bundle checks, `node dist/bootstrap.js --help` is valid. Validate `sessions` from a staged or installed publish runtime, because raw `dist/` does not include installed native dependencies such as `better-sqlite3`.

## Reporting

- Summarize by surface: packaging, install, docs, CLI, TUI, auth, review, execute, and provider matrix.
- For each failure, include the command, environment, observed behavior, expected behavior, and release impact.
- Mark any unvalidated surface as a release risk.
