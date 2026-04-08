# Scenario Matrix

Use this table to choose the right live-validation update path.

| Behavior change | Preferred surface | Typical files | Minimum checks |
| --- | --- | --- | --- |
| Help text, doctor, config, or other direct command output | `cli-command` invocation | `scripts/live-validation/scenarios/*.json`, CLI tests, `release-matrix.md` | `bun test scripts/live-validation/live-validation.test.ts` |
| Interactive CLI task flow, TUI behavior, review flow | `cli` invocation | scenario JSON, templates, `scripts/live-validation/tui-validator.ts` when relevant | `bun test scripts/live-validation/live-validation.test.ts`; narrow live run when needed |
| Machine execution flow or artifact expectations | `execute` invocation | scenario JSON, `scripts/live-validation/runner.ts`, `packages/executor` tests | live-validation test plus `execute-contract` checks |
| ArkTS validation or external tool setup | `cli` or `execute`, often with `requiresArktsLinter` | scenario JSON plus templates in `scripts/live-validation/templates/` | live-validation test; run the narrowest scenario or verification command |

No new scenario is usually needed when:
- the behavior is internal-only and already covered by unit tests
- docs changed without changing supported behavior
- an existing scenario already proves the contract after a small assertion update
