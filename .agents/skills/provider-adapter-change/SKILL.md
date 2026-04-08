---
name: provider-adapter-change
description: Handle provider-facing DevAgent work when a request says things like provider proxy, auth mismatch, model registry drift, header changes, or streaming regressions.
triggers:
  - provider proxy
  - auth mismatch
  - model registry drift
  - provider headers
  - streaming regression
paths:
  - packages/providers
  - models
  - packages/cli/src/provider-config.ts
examples:
  - honor provider proxy environment variables
  - fix a provider auth or model mismatch
---

# Provider Adapter Change

Use this skill when the request sounds like “provider proxy”, “fix auth mismatch”, “update model registry”, or “why is this provider streaming weirdly” and the task touches `packages/providers`, `models/*.toml`, provider auth or config flow, proxy behavior, headers, streaming behavior, or provider-facing docs.

Read `packages/providers/AGENTS.md` and `references/verification-matrix.md` first.

## Workflow

1. Scope the provider change precisely.
   - Identify which adapters, shared helpers, registry entries, or model files are affected.
   - Separate transport changes from docs-only changes.
2. Inspect provider-specific tests before editing.
   - Prefer the narrowest existing test file such as `openai.test.ts`, `anthropic.test.ts`, `network.test.ts`, or `registry.test.ts`.
   - Add or expand failing tests before changing behavior.
3. Check the four common drift points.
   - env var and credential expectations
   - proxy or base URL behavior
   - headers, capability flags, and streaming semantics
   - model-registry and CLI/provider-config impact
4. Run verification from `references/verification-matrix.md`.
   - Start with `cd packages/providers && bun run test`.
   - Add root `bun run typecheck` when exports or shared config changed.
5. Reconcile docs only if supported behavior changed.
   - Update README, provider docs, or doctor guidance deliberately.
   - Never leak tokens or log secrets while adding diagnostics.

## Escalate

- Use `security-checklist` when credentials, auth flows, or headers change.
- Use `validate-user-surface` when user-facing auth, doctor, or default-provider behavior changed materially.
- Use `release-train` when packaging or publish validation is part of the provider change.

## Red Flags

- Reusing one broad test when a provider-specific regression already has a focused test file.
- Changing shared provider helpers without checking registry or CLI config impact.
- Adding richer diagnostics that expose secrets or raw credentials.
