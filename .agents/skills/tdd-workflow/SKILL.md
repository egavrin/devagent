---
name: tdd-workflow
description: Fail-fast TDD loop for DevAgent — typecheck, write failing test, implement, verify.
---

# TDD Workflow

DevAgent follows a strict tests-first TDD loop rooted in the fail-fast philosophy. Every behavior change starts with a failing test.

## Before You Start

Run both checks to establish a clean baseline:

```bash
bun run typecheck   # Zero errors required
bun run test        # All tests must pass
```

If either fails, fix the existing issue first. Never start new work on a broken baseline.

## The Loop

### 1. Identify the Test File

Tests live next to production code:

```
packages/<pkg>/src/<module>.test.ts
```

If the test file doesn't exist yet, create it with the standard vitest imports:

```typescript
import { describe, it, expect, beforeEach, afterEach } from "vitest";
```

### 2. Write the Failing Test

Write a test that describes the desired behavior. Be specific:

```typescript
describe("myFunction", () => {
  it("returns parsed result for valid input", () => {
    const result = myFunction("valid-input");
    expect(result).toEqual({ parsed: true, value: "valid-input" });
  });
});
```

For tool tests, use the tmp directory pattern:

```typescript
import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import type { ToolContext } from "@devagent/core";

let tmpDir: string;
let ctx: ToolContext;

beforeEach(() => {
  tmpDir = mkdtempSync(join(tmpdir(), "devagent-test-"));
  ctx = {
    repoRoot: tmpDir,
    config: {} as ToolContext["config"],
    sessionId: "test-session",
  };
});

afterEach(() => {
  rmSync(tmpDir, { recursive: true, force: true });
});
```

### 3. Run to Failure

Run only the relevant package's tests:

```bash
bun run test -- --filter <package-name>
```

Or a specific test file:

```bash
cd packages/<pkg> && bunx vitest run src/<module>.test.ts
```

**Read the first error line.** That's the root cause. Later errors are usually cascading.

The test must fail for the expected reason (e.g., "function not defined" or "expected X but got Y"). If it fails for an unexpected reason, your test setup is wrong — fix that first.

### 4. Implement Minimal Code

Write the smallest change that makes the failing test pass. No extra features, no speculative generalization.

### 5. Verify

```bash
bun run typecheck   # Still zero errors
bun run test        # All tests pass (new + existing)
```

### 6. Repeat

Go back to step 2 for the next behavior.

## Anti-Patterns

- **Writing implementation before tests**: Always test first. The test defines the behavior.
- **Testing after the fact**: Tests written after implementation often test the implementation, not the behavior.
- **Silent guards in tests**: Don't catch exceptions to make tests pass. Let them surface.
- **Testing private internals**: Test public behavior through the public API.
- **Skipping typecheck**: Type errors caught early prevent runtime surprises.

## Package-Specific Test Patterns

| Package | Test focus | Mock pattern |
|---------|-----------|-------------|
| `core` | Pure logic, types, config parsing | Direct function calls |
| `tools` | File operations, git commands | `mkdtempSync` tmp dirs |
| `engine` | TaskLoop, judges, plugins | Mock `LLMProvider` + `EventBus` |
| `providers` | API integration | Mock HTTP responses |
| `cli` | Prompt assembly, argument parsing | Config fixtures |
