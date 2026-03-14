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

## Primary Evidence

- `README.md`
- `CONTRIBUTING.md`
- `AGENTS.md`
- `WORKFLOW.md`
- `SECURITY.md`
- `REVIEW.md`
- `scripts/check-oss.mjs`

## Workflow

1. Read the docs and metadata you are changing plus `scripts/check-oss.mjs`.
2. Remove or avoid references to unsupported surfaces, deprecated workflows, local machine paths, or stale package layout claims.
3. If docs describe tested behavior, verify the underlying code or tests before making the claim.
4. Run `bun run check:oss` before finalizing, and pair it with `bun run typecheck` and `bun run test` when the change intersects executable behavior.

## Red Flags

- Mentioning internal-only Hub or workflow-stage commands in public docs.
- Documenting removed surfaces such as chat/TUI/public plan mode.
- Leaving stale references to the old split `core`/`engine`/`tools` workspace layout.
