# Release Matrix

Use this file when planning coverage or writing the final report.

## Minimum Release Bar

- Pass `bun run build`.
- Pass `bun run typecheck`.
- Pass `bun run test`.
- Pass `bun run test:surface-smoke`.
- Pass `bun run check:oss`.
- Pass `bun run build:publish`.
- Pass `bun run test:bundle-smoke`.
- Pass `bun run test:live-validation`.
- Pass `bun run validate:live:full`.
- Pass `bun run validate:live:execute-deep` for a release-grade `devagent execute` packet when staged workflow quality matters.
- Pass `bun run validate:live:execute-chain` when you need proof that the canonical `design -> breakdown -> issue-generation -> implement -> review -> repair` flow works as one chained provider-backed run.
- Pass `bun run validate:live:provider-smoke` when local provider credentials or services are available.
- Explain every remaining failure or blocked surface explicitly.

## Environment Matrix

- Validate on real Node 20+ because the publish bootstrap targets Node, not Bun's Node shim.
- Validate Bun-backed developer flows when the README or local contributor workflow depends on Bun.
- Use isolated `HOME`, `XDG_CONFIG_HOME`, and `XDG_CACHE_HOME` for every install and auth pass.
- Do not infer provider credentials only from environment variables. The live validation harness can copy non-expired credentials from the local DevAgent `CredentialStore` into isolated homes; run provider smoke before marking credential-backed providers blocked.
- Keep one clean temp repo for install and help checks and separate temp repos for query and mutation scenarios.

## Packaging And Install Matrix

- Run `bun run build:publish`.
- Run `bun run test:bundle-smoke`.
- Run `cd dist && npm pack`.
- Install the tarball into a temp prefix and verify `devagent help`, `devagent version`, and `devagent doctor`.
- Remove that install and repeat to catch stale-file issues.
- Validate `node dist/bootstrap.js --help` directly from raw `dist/`.
- Validate `sessions` from the staged or installed publish runtime, because raw `dist/` intentionally lacks installed native dependencies such as `better-sqlite3`.
- Validate `npx` and `bunx` invocation paths against a prerelease tag when one exists.
- If no prerelease tag exists, validate the closest local equivalent and mark registry-backed `npx` or `bunx` as still pending.
- Compare `dist/package.json` and copied `dist/README.md` against the root contract.

## Docs And Contract Matrix

- Verify every install snippet in `README.md`.
- Verify every quick-start command in `README.md`.
- Verify the staged workflow section in `README.md`, including the lead example flow and the full stage matrix.
- Verify the supported provider list in `README.md`.
- Verify the command list in `README.md` against `devagent help`.
- Verify environment-variable documentation against actual runtime behavior.
- Verify `WORKFLOW.md` and any executor-facing docs still match `devagent execute --request --artifact-dir`.
- Verify the docs describe stage prompts as code-defined per `taskType` with dynamic request context, not user-configurable workflow stages.
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
- `devagent execute` readonly planning stages: `design`, `breakdown`, and `issue-generation`.
- Resume flow after a completed or partial session.
- Continue flow after an iteration-limited session when practical.

For execute validation:

- Verify the fixed supported stage set is documented and reflected in automation.
- Verify readonly stages stay non-mutating.
- Verify `breakdown` and `issue-generation` produce both structured JSON and rendered Markdown artifacts.
- Verify the canonical public flow `design -> breakdown -> issue-generation -> implement -> review -> repair` is covered by docs and scenario coverage, even if individual stages run as separate scenarios.
- When validating a true chained flow, verify later stages consume earlier-stage artifacts through request context such as comments, changed file hints, issue units, or context bundles rather than relying on a fresh ungrounded prompt.
- Preserve a reviewable packet with request JSON, stdout or stderr, emitted events, artifact inventory, workspace-effect review, and a human judgment note for each staged scenario.

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
- Run `bun run validate:live:provider-smoke` before declaring provider coverage blocked; it verifies stored local credentials and Ollama service availability from isolated homes.
- Use `bun run validate:live:execute-deep` when you need one ordered `execute` packet with prereqs, canonical staged flow, continuity checks, remainder coverage, and per-scenario review notes.
- Use `bun run validate:live:execute-deep --only canonical|continuity|remainder --skip-prereqs` for focused local reruns after a broader packet establishes the baseline.
- Use `bun run validate:live:execute-chain` when you need one disposable-worktree run that carries real stage artifacts forward into `implement`, `review`, and `repair`.
- Add manual coverage for gaps the harness does not own yet, especially packaging, registry install paths, auth flows, TUI interactions, and full provider breadth.

## Final Report Shape

- `Passed`: surfaces exercised live with enough evidence to trust.
- `Failed`: surfaces that reproduced a defect with commands and logs.
- `Blocked`: surfaces that require credentials, services, or prerelease publishing that were not available.
- `Gaps`: user-facing paths not yet covered by automation or manual execution.
- `Release recommendation`: ship, fix-before-ship, or blocked pending validation.
