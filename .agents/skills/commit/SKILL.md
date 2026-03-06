---
name: commit
description: Use when creating git commits, generating commit messages, or reviewing commit message quality — enforces Conventional Commits format with strict formatting rules
---

# Commit Message Rules

Conventional Commits format with strict plain-text formatting. Every commit message produced by devagent must follow these rules exactly.

## Format

```
type(scope): imperative summary

- bullet point explaining what changed and why
- another detail (only if title isn't self-documenting)
```

## Subject Line

- **Format**: `type(scope): description` — scope is optional
- **Imperative mood**: "add feature" not "added feature" or "adds feature"
- **Length**: aim for <50 chars, hard limit 72 chars
- **No period** at the end
- **No markdown** — plain text only
- Reference section titles, not numbers: "refine migration protocol" not "update section 9"

## Types

| Type | Use for |
|------|---------|
| `feat` | New feature or capability |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Formatting, whitespace, semicolons — no logic change |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `test` | Adding or updating tests |
| `chore` | Build, CI, tooling, dependencies |
| `perf` | Performance improvement |

## Scope

Derive from the package or module changed:

- `feat(cli):` — change in `packages/cli`
- `fix(tools):` — change in `packages/tools`
- `refactor(engine):` — change in `packages/engine`
- `test(core):` — change in `packages/core`
- Omit scope for cross-cutting changes

## Body

- **Blank line** between subject and body
- **Bullet points only** — use dash prefix, no paragraph text
- **Wrap at 72 characters**
- Do not repeat the subject line — add supplementary details only
- Summarize impact rather than listing every file
- **Omit body entirely** if the subject is self-documenting

## Breaking Changes

Use `!` after type/scope and explain in body:

```
feat(core)!: remove legacy config format

- Drop support for v1 TOML config
- Migration: run `devagent config migrate` first
```

Or add a `BREAKING CHANGE:` footer.

## Co-Authored-By Trailer

Every commit made by devagent must include a co-authorship trailer:

```
Co-Authored-By: devagent <devagent@noreply>
```

Place it as the last line after a blank line following the body (or subject if no body).

## Binary and Dependency Changes

- Name critical files explicitly, state encoding changes
- Summarize key package updates — never use generic "update dependencies"

## Anti-Patterns

- Generic messages: "update code", "fix stuff", "misc changes"
- Past tense: "added", "fixed", "updated"
- Markdown formatting: `**bold**`, `` `code` ``, `# headers`
- Exhaustive file lists — summarize at a higher level
- Period at end of subject line
- Subject exceeding 72 characters

## Quick Reference

```
feat(cli): add interactive mode flag          # new feature
fix(engine): handle empty tool response       # bug fix
refactor(core)!: simplify config schema       # breaking refactor
test(tools): cover patch-parser edge cases    # test addition
chore: bump vitest to 3.x                     # dependency update
docs: update CLI commands reference           # docs only
```
