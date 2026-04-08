---
name: oss-surface-guard
description: Keep DevAgent public docs and package metadata aligned with the supported OSS surface.
---

# OSS Surface Guard

Use this skill when a task edits public documentation, contributor workflow, release-facing metadata, or any text that describes supported DevAgent capabilities.

## Focus Areas

- Public docs describe only the supported DevAgent path and avoid deprecated or internal-only workflow stories.
- Contributor guidance stays aligned with the current repo layout and verification commands.
- Package metadata and top-level docs remain compatible with `scripts/check-oss.mjs`.
- Root docs, `devagent help`, and nearby command/documentation tests stay aligned when public behavior changes.
- When public commands, release-critical flows, or live-validation coverage changes, `.agents/skills/validate-user-surface/references/release-matrix.md` stays current.

## Primary Evidence

- `README.md`
- `CONTRIBUTING.md`
- `AGENTS.md`
- `WORKFLOW.md`
- `SECURITY.md`
- `REVIEW.md`
- `scripts/check-oss.mjs`
- `packages/cli/src/documentation.test.ts`
- `packages/cli/src/main.test.ts`
- `.agents/skills/validate-user-surface/references/release-matrix.md`

## Workflow

1. Read the docs and metadata you are changing plus `scripts/check-oss.mjs`.
2. Remove or avoid references to unsupported surfaces, deprecated workflows, local machine paths, or stale package layout claims.
3. If the change affects command help, CLI output, or other tested behavior, check the nearest command or documentation tests before making the claim.
4. If the change adds or removes public commands, live-validation scenarios, or release-critical flows, update `release-matrix.md` or explicitly confirm that no matrix change is needed.
5. Run `bun run check:oss` before finalizing, and pair it with `bun run typecheck` and `bun run test` when the change intersects executable behavior.

## Red Flags

- Mentioning internal-only Hub or workflow-stage commands in public docs.
- Documenting removed surfaces such as chat/TUI/public plan mode.
- Leaving stale references to the old split `core`/`engine`/`tools` workspace layout.
- Updating README or `WORKFLOW.md` without checking command-help coverage or release-matrix drift.
