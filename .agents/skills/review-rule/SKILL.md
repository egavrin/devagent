---
name: review-rule
description: How to author review rules for DevAgent's LLM-powered review pipeline.
---

# Writing Review Rules

Review rules are Markdown files that instruct the LLM reviewer what to look for in code patches. They're used with `devagent review <patch> --rule <rule_file>`.

## Rule File Structure

```markdown
# Rule Name

Brief description of what this rule detects.

## Applies To

**/*.ts

## Detect

Describe what patterns to flag. Be specific:
- What code patterns are violations?
- What severity level (error, warning, info)?
- What should be ignored (false positive guidance)?

## Examples

### Bad

```typescript
// This violates the rule because...
try {
  riskyOperation();
} catch {
  // silently swallowed
}
```

### Good

```typescript
// This is correct because...
try {
  riskyOperation();
} catch (err) {
  throw new DevAgentError("operation failed", { cause: err });
}
```
```

## Key Sections

### Applies To (Required)

File pattern that scopes which files the rule applies to. Supports:

- `**/*.ts` — all TypeScript files
- `src/**/*.test.ts` — only test files under src/
- `packages/runtime/**` — all files in the consolidated runtime package
- `packages/runtime/src/core/**` — runtime core/config/types only
- `packages/runtime/src/engine/**` — task loop, review, and subagent logic
- `packages/runtime/src/tools/**` — built-in tools and LSP integration
- `regex:.*\.(ts|js)$` — custom regex (prefix with `regex:`)
- Multiple patterns separated by commas: `**/*.ts, **/*.js`

### Detect

The core of the rule. Write clear, unambiguous detection criteria. The LLM will use this as its primary instruction for identifying violations.

Tips:
- Be specific about what constitutes a violation
- Mention edge cases that should NOT be flagged
- Include severity guidance (error for must-fix, warning for should-fix, info for style)

### Examples

Provide both bad (violation) and good (correct) examples. The LLM uses these for pattern matching. More examples = better accuracy.

## Testing Your Rule

### Create a test patch

```bash
git diff > /tmp/test.patch
```

Or for staged changes:

```bash
git diff --cached > /tmp/test.patch
```

### Run the review

```bash
devagent review /tmp/test.patch --rule my-rule.md --json
```

The `--json` flag outputs structured results for inspection:

```json
{
  "violations": [
    {
      "file": "src/example.ts",
      "line": 42,
      "severity": "error",
      "message": "Silent catch block swallows exception",
      "changeType": "added"
    }
  ],
  "summary": {
    "totalViolations": 1,
    "filesReviewed": 3,
    "ruleName": "no-silent-catch"
  }
}
```

### Iterate

1. Run on known-bad code — does it catch the violations?
2. Run on known-good code — does it produce false positives?
3. Adjust detection criteria and examples until accuracy is satisfactory

## DevAgent-Specific Rules to Consider

| Rule | Detects | Scope |
|------|---------|-------|
| no-silent-catch | Empty catch blocks, caught-and-ignored errors | `**/*.ts` |
| no-any-cast | `as any` type assertions | `**/*.ts` |
| fail-fast | Defensive fallbacks, best-effort returns | `packages/runtime/**/*.ts` |
| test-coverage | Functions without corresponding test | `packages/runtime/src/**/*.ts` |
| proper-errors | Raw `throw new Error()` instead of `DevAgentError` hierarchy | `packages/runtime/**/*.ts` |

## Severity Mapping

The pipeline normalizes severity aliases:

| Input | Normalized |
|-------|-----------|
| `critical`, `high`, `major` | `error` |
| `medium`, `moderate` | `warning` |
| `minor`, `low` | `info` |
