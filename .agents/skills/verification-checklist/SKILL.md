---
name: verification-checklist
description: Pre-commit/PR verification gate for DevAgent changes.
---

# Verification Checklist

Run this checklist before considering a substantial change complete.

## The Checklist

### 1. Type Check

```bash
bun run typecheck
```

Required: zero errors across the workspace.

### 2. Test Suite

```bash
bun run test
```

Required: all tests pass.

For faster local iteration, run the affected package directly:

```bash
cd packages/runtime && bun run test
cd packages/executor && bun run test
cd packages/cli && bun run test
cd packages/providers && bun run test
```

### 3. Build

```bash
bun run build
```

Required: clean workspace build.

### 4. OSS Surface Check

If you changed public docs, contributor workflow, or package metadata, run:

```bash
bun run check:oss
```

Required: pass without forbidden public-surface references.

### 5. Change Scope

```bash
git diff --stat
git status --short
```

Review whether the diff is focused and free of accidental generated files or unrelated edits.

### 6. Contract-Sensitive Review

When applicable, inspect the nearest high-risk area:

- `packages/executor/src/index.test.ts` for `devagent execute` contract changes
- `packages/runtime/src/**/*.test.ts` for shared runtime or tool behavior
- `packages/providers/src/*.test.ts` for provider-specific API behavior
- `packages/cli/src/*.test.ts` for CLI surface and prompt changes

### 7. Security Scan

Check for secrets, unsafe command construction, and leftover debug output:

```bash
git diff --cached
```

Look specifically for:

- credentials, tokens, or API keys
- shell command concatenation with user-controlled input
- accidental artifact or log output committed to the repo

## When to Run

- before wrapping up substantial work
- before opening or updating a PR
- after rebasing or resolving conflicts
- after touching multiple packages
