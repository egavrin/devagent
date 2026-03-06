---
name: review
description: Code review checklist for DevAgent contributions — fail-fast compliance, types, tests, module boundaries.
---

# Code Review

Structured code review for changes to the DevAgent codebase. Review each area in order.

## 1. Scope and Focus

- Is the change minimal and focused on one concern?
- Are there unrelated modifications (formatting, refactoring, feature creep)?
- Could any part of this change be a separate commit/PR?

## 2. Fail-Fast Compliance

DevAgent's core principle. Flag any violations:

- **Silent catch blocks**: `catch {}` or `catch { return defaultValue }` — exceptions must propagate or be explicitly handled with logging/re-throw
- **Best-effort returns**: Functions returning a fallback value when they should throw
- **Defensive guards**: `if (!x) return` patterns that hide bugs instead of surfacing them
- **Swallowed errors**: `catch (e) { console.log(e) }` without re-throwing
- **Layered fallbacks**: Multiple try/catch or `?? defaultValue` chains that mask the real failure

## 3. Type Safety

- No `any` casts (`as any`, `: any`)
- No type assertions that bypass checks (`as SomeType` without validation)
- Readonly where appropriate (`readonly` properties, `ReadonlyArray`)
- New interfaces properly exported from package `index.ts`
- Generic types have meaningful constraints

## 4. Test Coverage

- Every new behavior has a corresponding test
- Tests verify behavior, not implementation details
- Test names describe what is being tested, not how
- Tests use the standard patterns:
  - Tool tests: `mkdtempSync` + `ToolContext` fixture
  - Engine tests: mock `LLMProvider` + `makeConfig` fixture
  - Plugin tests: `makeContext()` with `EventBus`
- Edge cases covered: empty input, missing data, error paths

## 5. Module Boundaries

Enforce the dependency DAG:

```
core ← tools, providers
core, tools, providers ← engine
core, engine ← cli
```

- `core` must NEVER import from `tools`, `engine`, `providers`, or `cli`
- `tools` must NEVER import from `engine` or `cli`
- `engine` must NEVER import from `cli`
- Cross-package imports use the package name (`@devagent/core`), not relative paths

## 6. Error Handling

- Uses `DevAgentError` hierarchy, not raw `new Error()`
- Error codes are meaningful (match the error class's `.code` property)
- `extractErrorMessage()` used for unknown caught values
- No `catch` blocks that lose the original error's stack trace (use `{ cause: err }`)

## 7. Configuration Changes

If the change adds or modifies config:

- Type defined in `packages/core/src/types.ts`
- Default value provided in `packages/core/src/config.ts`
- TOML parsing handles the new field
- Validation rejects invalid values with clear error messages
- Documented in AGENTS.md if user-facing

## 8. Export Hygiene

- New public types/classes exported from package `index.ts`
- No unnecessary exports (internal helpers stay private)
- Re-exports use `export type` for type-only exports
- Export order: types first, then values

## Review Output Format

For each issue found:

```
[severity] file:line — description
```

Severities:
- **error**: Must fix before merge (bugs, security issues, fail-fast violations)
- **warning**: Should fix (missing tests, poor naming, minor type issues)
- **info**: Consider fixing (style, potential simplification)
