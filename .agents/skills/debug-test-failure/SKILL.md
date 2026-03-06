---
name: debug-test-failure
description: Systematic diagnosis of failing tests in the DevAgent monorepo.
---

# Debug Test Failure

Systematic approach to diagnosing and fixing test failures. Follow these steps in order — don't skip ahead.

## Step 1: Isolate the Failure

Run the specific failing test in isolation:

```bash
cd packages/<pkg> && bunx vitest run src/<module>.test.ts
```

To run a single test by name:

```bash
cd packages/<pkg> && bunx vitest run src/<module>.test.ts -t "test name pattern"
```

## Step 2: Read the First Error

The first error line is the root cause. Everything after it is usually cascading. Look for:

- **`TypeError: X is not a function`** → Wrong import, missing export, or renamed function
- **`ReferenceError: X is not defined`** → Missing import or undeclared variable
- **`Expected X to equal Y`** → Logic error in production code or outdated test expectation
- **`ENOENT: no such file`** → Missing fixture, wrong path, or tmp dir not created
- **`Cannot find module`** → Build not run after changes, or wrong import path

## Step 3: Classify the Error

### Type Error?

```bash
bun run typecheck
```

If typecheck fails, fix types first. Type errors often cause runtime failures with misleading messages.

### Runtime Error?

Read both the test file and the production code it exercises:

1. **Test file**: What behavior does this test expect?
2. **Production code**: What does the code actually do?
3. **Gap**: Where does the code diverge from the expectation?

### Fixture/Setup Error?

Check `beforeEach`/`afterEach` blocks:
- Is the tmp directory being created?
- Are test fixtures being written to the right paths?
- Is cleanup happening (stale state from a previous test)?

## Step 4: Check Staleness

Is the test outdated or is the code broken?

```bash
git log --oneline -5 -- <test-file>
git log --oneline -5 -- <production-file>
```

If the production code changed more recently than the test, the test likely needs updating.
If the test changed more recently, the production code likely has a regression.

## Step 5: Fix the Root Cause

**Never:**
- Wrap code in try/catch to silence the error
- Add a default return value to skip the failing path
- Delete or skip the test
- Add a `// @ts-ignore` or `as any` to make types pass

**Always:**
- Fix the actual logic error
- Update the test if the expected behavior changed intentionally
- Add a new test if you discovered an untested edge case

## Step 6: Verify the Fix

Run the fixed test:

```bash
cd packages/<pkg> && bunx vitest run src/<module>.test.ts
```

Then run the full suite to check for regressions:

```bash
bun run test
```

Then typecheck:

```bash
bun run typecheck
```

## Common Patterns in DevAgent

### Mock Provider Not Yielding Expected Chunks

Engine tests use a mock provider pattern. If the mock doesn't yield the right `StreamChunk` types, the TaskLoop will behave unexpectedly.

```typescript
// Check that mock responses include a "done" chunk
const responses = [
  [
    { type: "text" as const, text: "response" },
    { type: "done" as const },
  ],
];
```

### Tool Test Missing ToolContext

Tool handlers require a `ToolContext`. If the context is incomplete:

```typescript
const ctx: ToolContext = {
  repoRoot: tmpDir,
  config: {} as ToolContext["config"],
  sessionId: "test-session",
};
```

### Cross-Package Import Failures

After changing exports in a package's `index.ts`, dependent packages may fail. Run `bun run build` to regenerate `.d.ts` files, then re-run tests.

### Flaky Tests

If a test passes sometimes and fails other times:
1. Check for shared mutable state between tests
2. Check for filesystem race conditions (parallel test execution)
3. Check for time-dependent logic (use fixed dates in tests)
