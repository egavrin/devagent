---
name: execute-contract
description: Guard the `devagent execute` machine contract when changing executor behavior or its public documentation.
---

# Execute Contract

Use this skill when a task touches `packages/executor`, executor-related runtime behavior, or docs that describe the supported machine path.

## Focus Areas

- `devagent execute --request <request.json> --artifact-dir <path>` remains the only supported machine orchestration surface.
- SDK request validation, capability checks, task-query construction, artifact extraction, result classification, and event emission stay aligned with the current contract.
- Requested skill behavior, verification-command behavior, and Bun-vs-Node execution details remain deliberate and test-backed.

## Primary Evidence

- `packages/executor/src/index.ts`
- `packages/executor/src/index.test.ts`
- `README.md`
- `WORKFLOW.md`
- `REVIEW.md`

## Workflow

1. Read the nearest executor implementation and its matching tests first.
2. If behavior changes, extend `packages/executor/src/index.test.ts` before or with the code change.
3. Re-check public docs that describe the executor path so they do not drift from the tested behavior.
4. Run `bun run typecheck`, `bun run test`, and `bun run check:oss` before finalizing.

## Red Flags

- Introducing a new public executor surface without coordinated docs and tests.
- Changing artifact, event, or result semantics as an incidental refactor.
- Reintroducing deprecated workflow-stage or unsupported executor stories into public docs.
