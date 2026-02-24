## Code Review Guidelines

When reviewing code (directly or via `delegate` reviewer agents), prioritize high-signal,
actionable issues over low-confidence speculation.

## What Counts as a Real Bug

A finding should be reported as a bug only if most of the following are true:
1. It materially impacts correctness, security, reliability, or performance.
2. It is specific and fixable at a concrete location.
3. It is introduced by the change under review.
4. The author would likely fix it if shown clear evidence.
5. It does not depend on unstated assumptions.
6. Impact can be explained with a concrete scenario.

If confidence is low, keep it as an open question instead of a hard finding.

## Severity

- **P0 (Blocking)**: release-blocking failures, data loss, clear security break.
- **P1 (Urgent)**: incorrect behavior, major edge-case failure, high-risk regression.
- **P2 (Normal)**: meaningful but non-blocking defects.
- **P3 (Low)**: minor concerns and suggestions.

## Finding Format

```
[P1] `src/handler.ts:42` — Missing null guard before `.toLowerCase()`.
Fix: `const email = user.email?.toLowerCase() ?? "";`
```

Guidelines:
- Explain why it is a problem.
- Keep tone factual and concise.
- Keep fix suggestions minimal and immediately actionable.

## Output Structure

1. **Findings** (grouped by severity, highest first).
2. **Open Questions / Assumptions** (only unresolved uncertainty).
3. **Summary** (count by severity, or explicit "No issues found.").
