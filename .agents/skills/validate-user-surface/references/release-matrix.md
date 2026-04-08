# Release Matrix

Use this file when planning coverage or writing the final report.

## Minimum Release Bar

- Pass `bun run build`.
- Pass `bun run typecheck`.
- Pass `bun run test`.
- Pass `bun run check:oss`.
- Pass `bun run build:publish`.
- Pass `bun run test:bundle-smoke`.
- Pass `bun run test:live-validation`.
- Pass `bun run validate:live:full`.
- Pass `bun run validate:live:provider-smoke` when local provider credentials or services are available.
- Explain every remaining failure or blocked surface explicitly.

## Environment Matrix

- Validate on real Node 20+ because the publish bootstrap targets Node, not Bun's Node shim.
- Validate Bun-backed developer flows when the README or local contributor workflow depends on Bun.
- Use isolated `HOME`, `XDG_CONFIG_HOME`, and `XDG_CACHE_HOME` for every install and auth pass.
- Keep one clean temp repo for install and help checks and separate temp repos for query and mutation scenarios.

## Packaging And Install Matrix

- Run `bun run build:publish`.
- Run `bun run test:bundle-smoke`.
- Run `cd dist && npm pack`.
- Install the tarball into a temp prefix and verify `devagent help`, `devagent version`, and `devagent doctor`.
- Remove that install and repeat to catch stale-file issues.
- Validate `node dist/bootstrap.js --help` and `node dist/bootstrap.js sessions`.
- Validate `npx` and `bunx` invocation paths against a prerelease tag when one exists.
- If no prerelease tag exists, validate the closest local equivalent and mark registry-backed `npx` or `bunx` as still pending.
- Compare `dist/package.json` and copied `dist/README.md` against the root contract.

## Docs And Contract Matrix

- Verify every install snippet in `README.md`.
- Verify every quick-start command in `README.md`.
- Verify the supported provider list in `README.md`.
- Verify the command list in `README.md` against `devagent help`.
- Verify environment-variable documentation against actual runtime behavior.
- Verify `WORKFLOW.md` and any executor-facing docs still match `devagent execute --request --artifact-dir`.
- Treat doc drift as a release issue even when the code is correct.

## CLI Matrix

- `devagent help`
- `devagent version`
- `devagent doctor`
- `devagent setup`
- `devagent config path`
- `devagent config get provider`
- `devagent config set provider <value>` in an isolated home
- `devagent completions bash`
- `devagent completions zsh`
- `devagent completions fish`
- `devagent sessions`
- `devagent --resume <id>`
- `devagent --continue`
- `devagent -f prompt.md`
- `devagent --provider <provider> --model <model> "<query>"`
- `devagent --quiet "<query>"`
- Non-TTY interactive failure path for bare `devagent`

## TUI Matrix

- Start bare `devagent` in a PTY.
- Verify the welcome screen and model/version display.
- Submit at least one real prompt and confirm streamed output appears.
- Verify `/help`.
- Verify `/sessions`.
- Verify `/resume`.
- Verify `/continue`.
- Verify `/clear`.
- Verify the safety-mode toggle with `Shift+Tab`.
- Verify single-shot TUI rendering when stderr is a TTY and a query is provided.

## Auth Matrix

- API-key login, status, and logout for at least one API-key provider.
- Device-code login, status, and logout for ChatGPT when test credentials exist.
- Device-code login, status, and logout for GitHub Copilot when test credentials exist.
- Verify failure messaging for missing credentials.
- Verify provider-specific hints stay actionable.

## Query And Command Matrix

- Single-shot coding query that reaches the agent loop.
- Query from file with `-f`.
- `devagent review <patch> --rule <rule> --json`.
- `devagent execute --request request.json --artifact-dir artifacts/`.
- Resume flow after a completed or partial session.
- Continue flow after an iteration-limited session when practical.

## Provider Matrix

- `anthropic`
- `openai`
- `devagent-api`
- `deepseek`
- `openrouter`
- `ollama`
- `chatgpt`
- `github-copilot`

For each provider:

- Verify credential discovery or auth flow.
- Verify startup with a model that belongs to that provider.
- Verify provider-model mismatch guidance when practical.
- Run at least one real query or command that reaches provider setup.
- Record whether the surface was fully validated, partially validated, or blocked.

## Built-In Harness Coverage

- Use `bun run scripts/live-validation.ts --list-scenarios` to inventory current scenarios.
- Run the full suite before adding bespoke manual checks.
- Inspect `summary.json` and `summary.md` from the generated output directory.
- Add manual coverage for gaps the harness does not own yet, especially packaging, registry install paths, auth flows, TUI interactions, and full provider breadth.

## Final Report Shape

- `Passed`: surfaces exercised live with enough evidence to trust.
- `Failed`: surfaces that reproduced a defect with commands and logs.
- `Blocked`: surfaces that require credentials, services, or prerelease publishing that were not available.
- `Gaps`: user-facing paths not yet covered by automation or manual execution.
- `Release recommendation`: ship, fix-before-ship, or blocked pending validation.
