---
name: release-train
description: Run release-hardening work when a request says bundle smoke, release checks, packaging changes, install-flow drift, or pre-release readiness without needing the full live-validation sweep.
triggers:
  - bundle smoke
  - release checks
  - packaging changes
  - install-flow drift
  - pre-release readiness
paths:
  - package.json
  - scripts/bundle.ts
  - scripts/smoke-publish-bundle.ts
examples:
  - run release checks after a packaging change
  - compare bundle behavior with README install guidance
---

# Release Train

Use this skill when the request sounds like “bundle smoke”, “run release checks”, “verify packaging”, or “check install-flow drift”, including version bumps, packaging changes, publish workflow edits, README install-flow edits, or pre-release checks that are lighter than the full `validate-user-surface` pass.

Read `package.json`, `scripts/bundle.ts`, `scripts/smoke-publish-bundle.ts`, and `references/release-checklist.md` first.

## Workflow

1. Confirm the release-facing change.
   - version bump
   - packaging or bundle script change
   - publish workflow or metadata change
   - README install, bootstrap, or upgrade guidance change
2. Run the release-hardening path in order.
   - `bun run build`
   - `bun run typecheck`
   - `bun run test`
   - `bun run check:oss`
   - `bun run test:bundle-smoke`
3. Compare the surfaced contract.
   - README install instructions
   - help text or documented commands
   - publish bundle metadata and install expectations
   - Node and Bun version assumptions
4. Produce a short release-readiness summary.
   - passed surfaces
   - failures or blockers
   - whether escalation to `validate-user-surface` is required

## Escalate

- Use `validate-user-surface` for full release-candidate validation, real provider coverage, or install/auth/TUI/executor end-to-end checks.
- Use `oss-surface-guard` for doc-only public-surface drift.
- Use `live-validation-authoring` when release-hardening uncovers missing scenario coverage.

## Red Flags

- Bumping versions without smoke-testing the publish bundle.
- Editing install docs without comparing them to the actual bundle behavior.
- Treating `validate-user-surface` as the default for every small packaging edit.
