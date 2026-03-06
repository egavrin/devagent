---
name: verification-checklist
description: Pre-commit/PR verification gate — typecheck, test, build, commit format, scope review.
---

# Verification Checklist

Run this checklist before considering any change complete. Every step must pass.

## The Checklist

### 1. Type Check

```bash
bun run typecheck
```

**Required: zero errors.** Type errors in one package can cascade to dependents. Fix all errors before proceeding.

### 2. Test Suite

```bash
bun run test
```

**Required: all tests pass.** If a test fails, it's either a regression from your change or a pre-existing issue. Either way, fix it before proceeding.

For faster feedback during development, run only the affected package:

```bash
bun run test -- --filter <package-name>
```

### 3. Build

```bash
bun run build
```

**Required: clean build.** The build is dependency-ordered (`core` → `tools`/`providers` → `engine` → `cli`). A build failure means either a type error leaked through or an import path is wrong.

### 4. Commit Format

```bash
git log --oneline -5
```

Verify recent commits follow the format: `type(scope): description`

Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`
Scopes: `core`, `engine`, `tools`, `providers`, `cli`, `deps`

### 5. Change Scope

```bash
git diff --stat
```

Review: are all changes intentional? Watch for:
- Unrelated file modifications (accidental saves, formatting changes)
- Missing files (forgot to stage a new file)
- Excessive scope (change should be focused on one concern)

### 6. Security Scan

Check the diff for:
- `.env` files or secrets in staged changes
- API keys or tokens in code
- Large binary files
- `console.log` or debug statements left behind

```bash
git diff --cached --name-only | grep -E '\.(env|key|pem|cert)$'
```

## Quick One-Liner

For a fast pass/fail check:

```bash
bun run typecheck && bun run test && bun run build && echo "All clear"
```

## When to Run

- Before preparing a commit
- Before creating a pull request
- After resolving merge conflicts
- After rebasing onto a new base branch
- After any change that touches multiple packages
