## Code Review Guidelines

When reviewing code (via `delegate_agent` or direct review tasks), follow this framework.

### Bug Detection Criteria

A finding is a real bug if ALL of these are true:
1. It meaningfully impacts correctness, performance, or security.
2. It is discrete and actionable — the author can fix it in a specific place.
3. It was introduced in the change under review, not pre-existing.
4. The author would fix it if made aware — it's not an intentional trade-off.
5. It is not based on unstated assumptions about the broader system.
6. It provably affects other code or users, not just aesthetics.

If fewer than 4 criteria are met, it's a suggestion, not a bug.

### Priority System

- **P0 (Blocking)**: Crashes, data loss, security vulnerabilities. Must fix before merge.
- **P1 (Urgent)**: Incorrect behavior, broken edge cases, race conditions. Should fix.
- **P2 (Normal)**: Performance issues, missing validation, unclear error messages. Fix when convenient.
- **P3 (Nice-to-have)**: Style improvements, minor refactors, documentation. Optional.

### Finding Format

Each finding:
```
[P1] `src/handler.ts:42` — Missing null check on `user.email` before `.toLowerCase()`.
  Fix: `const email = user.email?.toLowerCase() ?? "";`
```

Format: `[priority] file:line — description. Fix: suggested code (3 lines max).`

### Comment Guidelines

- State **why** something is a problem, not just what's wrong.
- Use appropriate severity — don't cry wolf with P0 for style issues.
- Be brief and matter-of-fact. No lecturing.
- Suggested fixes should be immediately graspable — no multi-step refactors.

### Output Structure

1. **Findings** — grouped by severity (P0 first, then P1, P2, P3).
2. **Open questions** — things you're unsure about (need more context).
3. **Summary** — total findings by severity. State explicitly if no findings: "No issues found."
