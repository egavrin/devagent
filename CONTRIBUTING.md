# Contributing to DevAgent

## Who this repo is for

Contributors working on the first-party DevAgent executor, CLI, engine, tools, providers, and the
SDK-backed machine execution path.

## Prerequisites

- Bun `1.3.10+`
- Node `20+`
- sibling checkout of:
  - `devagent-sdk`
  - `devagent-runner`
  - `devagent`
  - `devagent-hub`

For the supported setup path, start from [`devagent-hub`](../devagent-hub/README.md):

```bash
cd ../devagent-hub
bun install
bun run bootstrap:local
```

## Local checks before opening a PR

```bash
bun install
bun run lint
bun run typecheck
bun run test
bun run check:oss
```

If your change affects the cross-repo machine path, also run the Hub baseline checks from
`../devagent-hub`.

## Contribution rules

- `devagent execute --request --artifact-dir` is the only supported machine orchestration contract.
- Keep changes small and explicit about provider/model assumptions.
- Do not document unsupported executor parity.
- Update tests and docs together when changing execution behavior.
