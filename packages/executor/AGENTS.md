# packages/executor

This file provides guidance to AI agents working in this component.

## Purpose and Scope

- **component**: `packages/executor`
- **purpose**: SDK-backed execution path for `devagent execute`, including request validation, skill resolution, task-query assembly, artifact generation, and verification command handling.
- **primary languages/platforms**: TypeScript, Bun, Node.js

## Important Paths and Interfaces

```text
src/index.ts                 # Execute entrypoint, request parsing, query construction, verification logic
src/index.test.ts            # Contract and behavior coverage for `devagent execute`
package.json                 # Build/test scripts and package metadata
```

## Build, Test, and Run

```bash
cd packages/executor
bun run build
bun run typecheck
bun run test
```

## Architecture and Workflows

- This package owns the supported machine contract. Keep behavior aligned with `README.md`, `WORKFLOW.md`, and the SDK request/result types.
- `src/index.test.ts` is the primary contract guard for request validation, capability checks, skill resolution, task-query shaping, artifact extraction, and verification-command behavior. Extend it whenever `execute` behavior changes.
- Requested skills are resolved from repo-local skill directories and injected into the task query. Missing skills currently warn and continue; if you change that behavior, update both tests and docs deliberately.
- Verify-command rewriting is sensitive to Bun-vs-Node execution details. Treat changes in this area as contract work, not incidental refactoring.
- If executor artifact/event behavior changes, keep `README.md`, `WORKFLOW.md`, and `src/index.test.ts` aligned.

## Generated Files, Conventions, and Pitfalls

- Do not broaden the public executor surface beyond `devagent execute --request ... --artifact-dir ...` unless the repository docs and validation story change together.
- Artifact shape, event emission, and outcome classification are externally consumed behavior. Avoid “cleanup” edits here without explicit regression coverage.
- Keep imports pointed at workspace package entrypoints such as `@devagent/runtime`, not relative cross-package paths.

## Local Skills

- No component-local skill directory exists here. Use the shared repo skills from `.agents/skills`, especially `review`, `testing`, `security-checklist`, `execute-contract`, and `verification-checklist`.
