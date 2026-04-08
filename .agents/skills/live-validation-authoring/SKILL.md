---
name: live-validation-authoring
description: Add or update live-validation coverage when a request says add a scenario, cover a release-critical flow, prove a command works end to end, or keep release validation in sync.
triggers:
  - add a live-validation scenario
  - release validation drift
  - cover this flow end to end
  - update release matrix
  - prove this command works
paths:
  - scripts/live-validation
  - .agents/skills/validate-user-surface/references/release-matrix.md
examples:
  - add scenario coverage for a new CLI command
  - update the release matrix after changing validation flows
---

# Live Validation Authoring

Use this skill when the request sounds like “add a scenario”, “cover this flow end to end”, “prove this command works”, or “update release validation” and the change adds or removes commands, changes documented behavior, or modifies release-critical flows that should be covered by `scripts/live-validation*`.

Read `scripts/live-validation/manifest.ts`, `scripts/live-validation/types.ts`, and `references/scenario-matrix.md` before editing scenarios.

## Workflow

1. Classify the behavior change.
   - CLI command or help surface
   - interactive CLI task flow
   - `devagent execute` task flow
   - review, auth, provider, or ArkTS validation behavior
2. Choose the smallest scenario shape that proves the behavior.
   - Reuse an existing scenario when only assertions, setup, or invocation details need to change.
   - Add a new scenario only when the behavior is materially distinct.
3. Update the scenario set coherently.
   - Change the manifest JSON, any referenced templates, and `release-matrix.md` together.
   - Keep IDs, suites, target repo, and assertions deliberate.
4. Verify discoverability and coverage.
   - Run `bun test scripts/live-validation/live-validation.test.ts`.
   - If the scenario semantics changed materially, run the narrowest live-validation command that exercises it.
5. Document the decision.
   - If the public behavior changed but no scenario was needed, explain why in the final summary.

## Escalate

- Use `surface-change-e2e` for the broader runtime or CLI implementation work around the scenario.
- Use `execute-contract` when the scenario covers `devagent execute` contract behavior.
- Use `release-train` when the change is part of release hardening rather than feature development.

## Red Flags

- Editing `scripts/live-validation/scenarios/*.json` without checking the manifest schema in `types.ts`.
- Adding a scenario but forgetting its templates, verification commands, or release-matrix entry.
- Using live validation as a substitute for nearby package tests.
