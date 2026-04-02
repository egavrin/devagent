---
name: review
description: Code review checklist for DevAgent contributions — fail-fast compliance, types, tests, module boundaries.
---

# Code Review

Operational review workflow for local DevAgent changes. Use it when the user asks to review staged, unstaged, or recent local changes.

## 1. Establish Scope

- Start with the diff the user pointed at.
- If no scope is specified, inspect the current local change set first: unstaged, then staged, then the last commit.
- Stay local. Do not assume GitHub PR context unless the user explicitly asks for it.

## 2. Gather Evidence

- Read the changed files and the nearest call sites or dependent code paths.
- Use readonly `reviewer` delegates for independent bug-finding lanes when the scope is large or the evidence can be parallelized.
- Prefer concrete evidence over speculative style commentary.

## 3. Review Priorities

Check the change in this order:

- Correctness and regressions
- Fail-fast violations:
  `catch {}`, fallback mazes, swallowed errors, defensive early returns that hide bugs
- Type safety:
  `any`, unsafe assertions, missing readonly boundaries, export drift
- Test coverage:
  missing or weak tests for new behavior and error paths
- Module boundaries:
  invalid cross-package imports, executor contract drift
- Performance:
  repeated work, unnecessary sync work, N+1 patterns, broad scans on hot paths

## 4. What Counts as a Real Finding

Report a finding only when it is:

- Introduced by the change under review
- Concrete and fixable at a specific location
- Material to correctness, reliability, security, or performance
- Defensible without hidden assumptions

If confidence is low, move it to open questions instead of reporting it as a hard finding.

## 5. Output Format

Always present results in this order:

1. Findings
2. Open Questions / Assumptions
3. Short Summary

For each finding, use:

```text
[severity] path:line — description
```

Every finding must include:

- severity
- file (and line when known)
- rationale explaining why the issue matters

Severities:

- `error`: must-fix bugs, security issues, fail-fast violations
- `warning`: should-fix missing tests, reliability gaps, meaningful type issues
- `info`: worthwhile but non-blocking improvements
