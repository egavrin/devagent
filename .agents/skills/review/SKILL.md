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
- Unless the user explicitly forbids delegates, launch three readonly `reviewer` delegates before concluding.
- Use distinct lanes for correctness/regressions, tests/contracts, and performance/fail-fast risks.
- Aggregate the delegates' blocking issues into one final review instead of pasting their output verbatim.
- Prefer concrete evidence over speculative style commentary.

## 3. Review Priorities

Check the change in this order:

- Design and system fit:
  does the change fit the surrounding architecture and existing abstractions?
- Functionality and user impact:
  correctness, regressions, and broken user-facing behavior
- Concurrency and race-risk reasoning when applicable
- Fail-fast violations:
  `catch {}`, fallback mazes, swallowed errors, defensive early returns that hide bugs
- Unnecessary complexity or over-engineering:
  extra abstraction, duplication, or indirection without payoff
- Type safety:
  `any`, unsafe assertions, missing readonly boundaries, export drift
- Tests and documentation:
  missing or weak tests for new behavior and error paths, plus docs drift when behavior changed
- Module boundaries:
  invalid cross-package imports, executor contract drift
- Performance:
  repeated work, unnecessary sync work, N+1 patterns, broad scans on hot paths
- Style and polish only after the above, and only when it is worth mentioning

## 4. What Counts as a Real Finding

Report a finding only when it is:

- Introduced by the change under review
- Concrete and fixable at a specific location
- Material to correctness, reliability, security, or performance
- Defensible without hidden assumptions

If confidence is low, move it to open questions instead of reporting it as a hard finding.

## 5. Triage

Always present results in this order:

1. Blocking Findings
2. Non-blocking Suggestions
3. Open Questions / Assumptions
4. Short Summary

Use `Blocking Findings` only for issues that should stop approval or should be fixed before the change is considered healthy:

- correctness bugs or regressions
- security issues
- fail-fast violations
- contract drift
- serious missing coverage for risky new behavior
- material performance risks

Do not block on optional polish. Put stylistic, readability, naming, or minor maintainability feedback into `Non-blocking Suggestions`.

If a point is speculative or depends on hidden assumptions, move it to `Open Questions / Assumptions` instead of escalating it into a finding.

## 6. Output Format

Do not narrate your process. Do not add a preamble, status line, or duplicated headings. Start immediately with `Blocking Findings`.

For each blocking finding, use:

```text
[severity] path:line — description
```

Every blocking finding must include:

- severity
- file (and line when known)
- rationale explaining why the issue matters

Severities:

- `error`: must-fix bugs, security issues, fail-fast violations
- `warning`: should-fix missing tests, reliability gaps, meaningful type issues

For `Non-blocking Suggestions`, use either:

- `None.`
- bullet suggestions such as `- packages/cli/src/main.ts:42 — Consider renaming this helper to match the runtime terminology.`

In `Short Summary`, include a code-health verdict, for example:

- `Overall: improves code health`
- `Overall: does not improve code health`
