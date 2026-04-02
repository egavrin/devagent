---
name: simplify
description: Identify and remove over-engineering — dead code, premature abstractions, unnecessary complexity.
---

# Simplify

Find and remove unnecessary complexity from DevAgent code. Cover reuse, code quality, and efficiency. Every worthwhile simplification must be verified with focused checks.

## 1. Start From the Change Scope

- Inspect the selected local diff first: unstaged, staged, or last commit.
- If no scope is specified, start from the current local change set before expanding further.
- Keep the work grounded in the actual touched files unless the evidence clearly points to nearby shared code that should also change.

## 2. Review Lanes

Use an adaptive delegation policy:

- Single-file, low-churn scopes should stay local by default.
- Broader or higher-churn scopes should fan out into readonly delegates before editing.
- If the user explicitly says `no delegates`, stay local unless blocked.
- If the user explicitly asks for parallel review or delegates, honor that even on smaller scopes.

When the scope merits delegation, use readonly delegates to review the same change set in parallel across three lanes:

- **Reuse**: existing helpers or utilities that should replace new ad-hoc code
- **Quality**: dead code, over-abstraction, fallback mazes, parameter sprawl, leaky abstractions
- **Efficiency**: repeated work, unnecessary sync work, missed concurrency, no-op updates, hot-path bloat

Aggregate the findings, then apply the worthwhile fixes in the main agent.

## 3. What to Look For

### Dead Code

- Unused exports (functions, types, constants exported but never imported)
- Unreachable branches (`if` conditions that can never be true)
- Commented-out code (delete it — git has history)
- Unused variables or parameters (check for `_` prefix convention)

Search for unused exports:

```bash
# Find exports
rg "^export " packages/<pkg>/src/<file>.ts

# Check each export is imported somewhere
rg "import.*<name>" packages/
```

### Over-Abstraction

- One-time helpers: functions called from exactly one place that don't clarify intent
- Premature generalization: generic types or config options for scenarios that don't exist yet
- Unnecessary indirection: wrapper functions that just forward to another function
- Class where a function would do: stateless classes with a single method

### Fallback Mazes

Per DevAgent's fail-fast philosophy:

- Layered `try/catch` blocks with fallback behavior
- `?? defaultValue` chains that hide the real failure
- `if (!x) return []` patterns that return empty results instead of throwing
- Multiple resolution strategies (try path A, then B, then C)

Replace with: resolve a single authoritative source, throw if missing.

### Unnecessary Complexity

- Overly clever patterns (clever code is hard to debug)
- Premature optimization (profile first, optimize second)
- Feature flags for features that are always on
- Backwards-compatibility shims for internal-only APIs
- Re-exported types that have been renamed

### YAGNI Violations

- Features not required by current use cases
- Configuration options nobody uses
- Extra parameters "for future use"
- Abstract base classes with a single implementation

## 4. Process

### 1. Identify Target

Pick a module or file to simplify. Start with files that have recent churn or high complexity.

### 2. Analyze

Read the file and note:
- Which exports are used externally?
- Which functions are called from where?
- Are there patterns that violate fail-fast?
- Is there dead or unreachable code?

### 3. Simplify

Make changes one at a time:
- Remove dead code
- Inline one-time helpers
- Replace fallback chains with fail-fast throws
- Delete commented-out code
- Remove unused parameters
- Reuse existing utilities instead of cloning logic
- Tighten obviously wasteful work in the changed path

### 4. Verify After Each Change

```bash
bun run typecheck   # Types still valid
bun run test        # Behavior unchanged
```

Run `bun run build` when the touched package or change shape makes it relevant.

Never batch multiple risky simplifications without verifying between them. A single bad removal can cascade into confusing errors.

## 5. What NOT to Simplify

- **Public API contracts**: Don't remove exports that external consumers depend on
- **Error handling at system boundaries**: Validation of user input, API responses, file I/O is necessary
- **Test utilities**: Shared test helpers are worth the abstraction
- **Performance-critical code**: Don't simplify if profiling shows the complexity is justified

## Red Flags in DevAgent Code

| Pattern | Fix |
|---------|-----|
| `catch { return [] }` | Remove catch, let error propagate |
| `as any` | Fix the type properly |
| `// TODO: remove` | Remove it now |
| `if (false)` or `if (0)` | Delete the dead branch |
| Function with 5+ parameters | Consider an options object |
| `export` on internal helper | Remove export |
| Re-export with rename | Use the canonical name everywhere |
