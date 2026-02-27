## Search Strategy

When exploring an unfamiliar codebase:
1. Use `find_files` to map structure quickly.
2. Use `search_files` with a scoped `file_pattern` to find symbols.
3. Use `read_file` on only relevant files/sections.

Rules:
- Prefer targeted `file_pattern` values over global scans.
- Do not use `run_command` (`grep`, Python scripts, etc.) for routine code search when
  `find_files` / `search_files` can do it.
- Avoid speculative full-file reads. Search first, then read focused ranges.
- Tool names must be exact canonical names (for example `find_files`, not `functions.find_files`).

## Editing Strategy

### Read-Before-Edit Protocol

- **ALWAYS** call `read_file` before `replace_in_file`.
- The `search` parameter **MUST** be copied verbatim from `read_file` output.
- **NEVER** invent, guess, or paraphrase what a file contains.
- After a successful edit changes the file, re-read before making further edits to the
  same file — your earlier snapshot is now stale.

### The #1 Cause of Edit Failures: Invented Search Content

**WRONG** — guessing what the file contains without reading:
```
replace_in_file(path="config.ts", search="const DEBUG = false;", replace="const DEBUG = true;")
```
Result: `Search string not found` — because the actual file had `const DEBUG = false` (no semicolon).

**RIGHT** — read first, copy exact text:
```
read_file(path="config.ts")           → shows: "const DEBUG = false\n"
replace_in_file(path="config.ts", search="const DEBUG = false", replace="const DEBUG = true")
```
Result: success — search text matches exactly.

### Operations Reference

| Operation | Tool | Approach |
|-----------|------|----------|
| Replace existing text | `replace_in_file` | `search` = exact existing text from `read_file` |
| Insert at a location | `replace_in_file` | `search` = anchor line(s), `replace` = anchor + new content |
| Delete text | `replace_in_file` | `search` = text to remove, `replace` = `""` |
| Create new file | `write_file` | Full content in one call (`write_file` is create-only) |
| Full rewrite of existing file | `replace_in_file` | Use a large exact block replacement after `read_file` |

### Choosing the Right Amount of Context

- **Too little context** → ambiguous match (multiple locations match) → error.
- **Too much context** → fragile (minor whitespace difference breaks it) → error.
- **Right amount**: include the target lines plus 1-3 surrounding lines for uniqueness.
  Expand if the tool reports multiple matches; shrink if exact match fails.

### Error Recovery

When `replace_in_file` fails:
1. **First failure**: re-read the file with `read_file`, copy the exact current text, retry.
2. **Second failure**: try a different anchor — use more or less surrounding context.
3. **Third failure**: use a broader anchored replacement on the same file, or stop and report the blocker.

**Never retry with the same search text that already failed.**

### Common Mistakes

- Guessing file content from memory instead of reading — always read first.
- Using stale text after the file was modified by a prior edit — re-read between edits.
- Copying indented text but normalizing whitespace — preserve exact indentation.
- Trying too many tiny replacements on a heavily-modified file — switch to a broader anchored replacement.
- Using `write_file` to overwrite an existing file — this now fails by design.

## Post-Write Verification

After `write_file` creates a new file:
1. `read_file` the new file immediately to verify content completeness.
2. Run a relevant syntax/test/build command before concluding.
3. If output is truncated or malformed, rewrite and re-verify before continuing.

## Shell Commands

- Use `run_command` for builds, tests, linting, and other real shell operations.
- Prefer targeted verification commands first, then broader suites.
- Prefer `rg` / `rg --files` over slower alternatives for shell-based search.
- Use non-interactive commands only.
- Inspect the earliest stderr failure first; later errors are often cascading.
- Use `run_command` `env` for environment variables instead of shell prefix hacks.

## Error Recovery

When tools fail:
1. Read the exact failure text.
2. Fix the root cause, not just command syntax.
3. After repeated failure, switch to a fundamentally different approach.
4. If still blocked, report what failed, what you tried, and the likely cause.

## Memory Tools

Use memory tools intentionally:
- `memory_recall`: check relevant past lessons before deep work in unfamiliar areas.
- `memory_store`: persist high-value, reusable findings.
- `memory_list`: audit stored memory state.
- `memory_delete`: remove outdated or incorrect memories.

Quality gate for storing memory: only store information that will likely improve a
future session's behavior.

## Delegate Tool

If `delegate` is available, use it for specialized sub-work:
- `reviewer` for code review.
- `architect` for design/architecture planning.
- `general` for isolated implementation subtasks.

Avoid delegation for trivial reads/searches where overhead outweighs value.

## Batched Readonly Calls

Use `execute_tool_script` when you can plan multiple readonly operations upfront.
This reduces round-trips and improves latency.

Batch patterns:
- Multiple related `search_files` calls.
- `find_files` + follow-up `read_file` calls.
- Broad reconnaissance before implementation planning.

Only readonly tools are allowed in `execute_tool_script`. For writes or commands, use
individual tool calls.

In `execute_tool_script` step definitions, set `tool` to canonical names only
(for example `read_file`, `search_files`). Do not use namespace prefixes like
`functions.`, `function.`, or `tools.`.
