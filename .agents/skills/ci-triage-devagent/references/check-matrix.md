# CI Check Matrix

| Failing area | First local repro | Likely owner | Notes |
| --- | --- | --- | --- |
| TypeScript or build failure | `bun run typecheck` or `bun run build` | touched package plus exports | If one package is obvious, run its local tests next |
| Package test failure | `cd packages/<pkg> && bun run test` | package that owns the failing file | Pair with `debug-test-failure` |
| Docs or OSS drift | `bun run check:oss` | docs, README, package metadata, CLI docs | Check command-help tests when docs mention behavior |
| Bundle or publish smoke | `bun run test:bundle-smoke` | packaging, bundle scripts, CLI install flow | Escalate to `release-train` |
| Live-validation failure | `bun test scripts/live-validation/live-validation.test.ts` | scenario, runner, docs, or underlying feature | Distinguish provider/setup blockers from regressions |
| Provider auth or network failures | provider-local tests first, then targeted live validation if needed | `packages/providers`, CLI config, doctor/auth flow | Missing credentials or external service availability should be reported as blockers |

Prefer one narrow repro command with a clear expected signal over a broad “run everything” pass.
