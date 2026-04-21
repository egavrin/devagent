---
name: ci-triage-devagent
description: Triage DevAgent CI and check failures when a request says CI failure, why did this check fail, reproduce the failing job locally, or separate blockers from real regressions.
triggers:
  - CI failure
  - why did this check fail
  - reproduce the failing job locally
  - broken GitHub Actions check
  - failing bundle smoke
paths:
  - .github/workflows
  - scripts
  - packages
examples:
  - map a failing GitHub Actions job to a local repro command
  - separate provider credential blockers from a real regression
---

# CI Triage DevAgent

Use this skill when the request sounds like “CI failure”, “why did this check fail”, or “reproduce the failing job locally”, or when a risky change needs a fast local repro map before deeper debugging.

Read `references/check-matrix.md` first. Pair with the GitHub CI skill if the failure originates in GitHub Actions.

## Workflow

1. Identify the failing check and reduce it to a repo-native repro command.
2. Classify the failure before changing code:
   - compile or type issue
   - unit or integration test regression
   - OSS or docs drift
   - bundle or publish problem
   - validation-helper or provider-environment blocker
3. Reproduce the narrowest failure locally.
   - Prefer package-local tests when the failure points at one package.
   - Use root commands for cross-package or release checks.
4. Separate real regressions from blockers.
   - Missing credentials, missing sibling repos, or external service outages are blockers, not code fixes.
   - Report those explicitly before patching code.
5. Widen only after the narrow repro is understood.
   - Run broader root checks only after the likely root cause is identified.

## Escalate

- Use `debug-test-failure` once the failing test is isolated.
- Use `release-train` for publish or packaging failures.
- Use `validate-user-surface` when the failing CI path reflects stale release-validation coverage.

## Red Flags

- Starting with `bun run test` when the failing check is obviously narrower.
- Treating auth or provider-environment failures as code regressions.
- Fixing symptoms in docs or snapshots without reproducing the underlying failing command.
