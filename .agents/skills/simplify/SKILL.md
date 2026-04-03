---
name: simplify
description: Aggressively simplify recent DevAgent changes with a diff-first, deletion-first pass that challenges newly added surface area and removes unjustified code.
---

# Simplify

Use this skill after implementation work to challenge whether newly added code should exist at all, remove unjustified surface area, and leave the smallest patch that still satisfies the user intent and DevAgent conventions.

The goal is not cosmetic cleanup. The goal is to remove feature-scope growth, unnecessary abstractions, and AI-generated bloat while preserving behavior at the user-intent level.

## 1. Honor Project Standards First

- Follow already-loaded project instructions before any generic cleanup heuristic.
- `AGENTS.md`, `CLAUDE.md`, and more specific subtree instruction files override this skill when they conflict.
- If local conventions disagree with a generic simplification instinct, follow the local convention.
- Keep DevAgent's fail-fast philosophy intact: remove silent fallbacks and defensive noise, do not add them.

## 2. Start From the Diff

- Inspect the selected local diff first: unstaged, staged, or last commit.
- If no scope is specified, start from the current local change set before expanding further.
- Stay grounded in the touched files unless the evidence clearly points to a nearby shared utility, type, or call site that should change too.
- Treat this as post-implementation cleanup, not broad opportunistic refactoring across the repo.
- Assume new code is guilty until it has a concrete justification.

## 3. Prefer Subagents for Independent Review Lanes

Use an adaptive delegation policy:

- Small, low-churn single-file scopes may stay local.
- For broader or higher-churn scopes, prefer parallel readonly reviewer subagents instead of doing every pass serially in one agent.
- If the user explicitly says `no delegates`, stay local unless blocked.
- If the user explicitly asks for parallel review or delegates, honor that even on smaller scopes.

Keep compatibility with the current `/simplify` orchestration:

- Use delegated subagents for the first three lanes whenever the scope supports it.
- Run the fourth deletion-biased synthesis lane in the main agent after aggregating delegate findings.
- If the environment cleanly supports an additional delegate for minimization, you may use it, but do not require it.

Delegate output is evidence, not the final answer. The main agent owns synthesis, deduplication, prioritization, removals, edits, and verification.

## 4. Review Lanes

Review the same diff across these lanes:

- **Reuse**: prefer existing helpers, utilities, types, constants, and patterns over new ad-hoc code
- **Quality**: remove dead code, over-abstraction, wrapper passthroughs, fallback mazes, parameter sprawl, leaky abstractions, and obvious comments
- **Efficiency**: remove repeated work, unnecessary sync work, duplicate I/O, no-op updates, missed concurrency, and hot-path bloat
- **Scope Challenge / Minimization**: challenge whether the newly added files, exports, config keys, tests, docs, abstractions, and extension points should exist at all

## 5. What to Look For

### Reuse

- New helper when an existing utility already fits
- New type alias, enum, constant, or wrapper for an existing concept
- Copy-paste with slight variation instead of reusing an existing path
- New public surface used from only one call site

### Quality

- Dead code, unreachable branches, commented-out code, and unused exports
- Redundant or derived state that should be computed instead
- Parameter sprawl and option bags added without real need
- Leaky abstractions and wrapper passthroughs
- Stringly-typed additions where an existing type or constant should be reused
- Unnecessary wrappers, nesting, or indirection
- Single-use abstractions that do not improve intent
- Speculative extensibility: flags, config branches, generic hooks, or extension points added "for later"
- Over-defensive branching that hides actual failures
- Obvious comments, task narration comments, and change-summary comments

Search helpers when needed:

```bash
rg "^export " packages/<pkg>/src
rg "<symbolName>" packages/
rg "TODO|FIXME|XXX|HACK" packages/
```

### Efficiency

- Repeated work in the same flow
- Duplicate API calls or duplicate I/O
- Sequential async work that can safely run in parallel
- Expensive work added to a hot path
- No-op updates or broad scans that can be narrowed
- Pre-checks that duplicate the failure semantics of the actual operation

### Scope Challenge / Minimization

Run this pass explicitly after the other lanes and make it skeptical by default:

- Does this change introduce more product surface than the request requires?
- Can the same intent be met with fewer files, fewer exports, fewer config keys, fewer tests, or fewer docs updates?
- Did the change add a reusable subsystem before reuse was proven?
- Did it add migration, compatibility, or extension behavior that the task did not require?
- Can a new helper be deleted and replaced with an existing call?
- Can two or three small new functions be collapsed into one clearer flow?
- Can a new abstraction layer be removed entirely?
- Can a new file, type, enum, or config branch be avoided?
- Can a one-call-site public API be demoted back to local scope?
- Can same-change tests, docs, or config be deleted because the supporting surface is removable?
- Can custom logic be replaced by a standard library or existing utility?
- For large diffs, can you remove 20-30% of the added lines without changing the user-visible outcome?

## 6. Aggressive Rules

- Assume every new file, export, config key, enum, type alias, helper, and test needs justification.
- If the new behavior can be preserved with less surface area, prefer that.
- If the change added supporting abstractions before proving reuse, remove them.
- If tests or docs exist only because optional surface was added, challenge the surface first.
- Do not stop at helper cleanup if the real simplification is removing a sub-feature, option, compatibility branch, or public-looking API.
- Prefer removing feature surface over refactoring feature surface more cleanly.
- Do not preserve internal scaffolding just because it already has tests.

## 7. Process

### 1. Establish Scope

- Start from the requested diff.
- Read only the touched code and the nearest call sites or helpers needed to judge reuse and risk.

### 2. Identify Added Surface Area

- List the newly added files, exports, config keys, flags, entry points, tests, docs updates, wrappers, and extension points.
- Treat each one as removable until justified.

### 3. Run the Review Lanes

- On trivial scopes, run all four lanes locally.
- On broader scopes, launch parallel readonly reviewer subagents for **Reuse**, **Quality**, and **Efficiency**.
- Then run **Scope Challenge / Minimization** locally as the deletion-biased synthesis pass unless a clean fourth delegate is available.

### 4. Remove First, Refactor Later

- Collect findings from all lanes.
- Deduplicate overlapping findings before changing code.
- Remove unjustified additions first.
- Only then refactor what remains.
- Prefer deletions, collapses, demotions, and reuse over introducing a "cleaner" abstraction that still grows the patch.

### 5. Apply Aggressively but Safely

Usually safe:

- Delete dead code and unreachable branches
- Inline trivial one-use wrappers
- Remove obvious comments and commented-out code
- Reuse an existing helper instead of keeping a duplicate
- Remove redundant state or obviously duplicated work
- Delete same-change tests, docs, or config that only support removable internal surface
- Demote one-call-site exports that do not need external visibility

Use caution:

- Changing async structure or concurrency
- Removing or narrowing public contracts
- Changing control flow around side effects, retries, cleanup, or resource lifetimes
- Replacing types or APIs that may have external consumers
- Removing validation or error handling at real system boundaries

### 6. Verify After Meaningful Removals

Run focused checks after worthwhile simplifications:

```bash
bun run typecheck
bun run test
```

Run `bun run build` when the touched package or change shape makes it relevant.

Never batch multiple risky simplifications without verification between them.

## 8. What NOT to Simplify

- **Public API contracts**: do not remove or narrow exports that external consumers depend on
- **System-boundary validation**: input validation, API response validation, and file I/O checks remain necessary
- **Proven performance-motivated complexity**: do not simplify complexity that exists for measured hot-path reasons
- **Required error handling**: keep boundary error handling that surfaces real failures clearly

Do not treat speculative internal structure as protected just because it exists. Protect only real boundaries, real contracts, and real behavior requirements.

## 9. Red Flags in DevAgent Code

| Pattern | Preferred simplification |
|---------|---------------------------|
| New helper duplicates existing utility | Reuse the existing utility and delete the helper |
| New file exists only to host a tiny wrapper | Collapse it back into the existing module |
| New export used once | Demote to local scope if no external contract needs it |
| New config key or flag supports optional behavior | Delete the option unless the task required it |
| Same-change tests/docs only support removable surface | Remove them with the surface |
| `catch { return [] }` | Remove the silent fallback and let the failure surface |
| Derived value stored as state | Compute it instead of persisting redundant state |
| Wrapper that only forwards args | Inline or delete the wrapper |
| Added comment explains obvious code | Delete it unless it captures non-obvious why |
| Sequential independent async calls | Parallelize if behavior and ordering stay safe |
| New flag/config branch "for future use" | Delete speculative extensibility |
