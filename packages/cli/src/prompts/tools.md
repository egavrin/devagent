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
- **Use dedicated tools instead of `run_command` for git operations**: `git_diff` for diffs, `git_status` for status. These tools are optimized for context efficiency — their outputs receive higher compaction priority and deduplication.
- Prefer targeted verification commands first, then broader suites.
- Prefer `rg` / `rg --files` over slower alternatives for shell-based search.
- Use non-interactive commands only.
- Inspect the earliest stderr failure first; later errors are often cascading.
- Use `run_command` `env` for environment variables instead of shell prefix hacks.

## Error Recovery

When tools fail:
1. Read the exact failure text — the first error line is usually the root cause.
2. Fix the root cause, not just command syntax.
3. **Never retry a tool with the same arguments that already failed.** If the same error
   recurs, the approach is wrong, not the timing.
4. After 3 failures of the same tool, you **must** switch to a different tool or strategy.
   Continuing to retry is never productive.
5. If still blocked, report what failed, what you tried, and the likely cause.

### `run_command` Error Recovery

- **Test/build failures**: fix the code that caused the failure, then re-run. Do not re-run
  the same command hoping for a different result.
- **Infrastructure errors** (timeout, killed, ENOMEM): try a more targeted command
  (e.g., run a single test file instead of the full suite).
- **Command not found**: check the project's package manager and scripts before inventing
  commands.
- Inspect stderr before stdout — the earliest stderr line is usually the root cause;
  later errors are often cascading.

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

### `execute_tool_script` Format and Error Recovery

**Format requirements:**
- `tool` must be a canonical name: `read_file`, `search_files`, `find_files`, `git_status`,
  `git_diff`. Never use prefixes like `functions.read_file` or `tools.search_files`.
- `args` must be a valid JSON string (not an object) encoding the tool parameters.
- `id` must be unique within the script and cannot reference itself.
- Forward references (`$step2` used in step1 before step2 runs) are rejected.

**When a script fails:**
1. Check which specific steps failed — the output shows `[FAILED]` per step.
2. **Do not retry the entire script.** Break the failed steps into individual tool calls.
3. If the error is about tool names or argument format, fix the format and use a single
   direct tool call instead of re-wrapping in a script.
4. If multiple steps fail with the same error, the common cause is usually a format issue
   (namespace prefix, malformed JSON args, or wrong tool name).

**Common mistakes:**
- Using `functions.read_file` instead of `read_file` — tool names must be bare canonical names.
- Passing args as an object `{path: "..."}` instead of a JSON string `"{\"path\": \"...\"}"`.
- Referencing a step ID that doesn't exist or hasn't executed yet.
- Retrying a failed script with the same steps — always simplify or break apart.

## Post-Compaction Context

After context compaction, your earlier tool outputs (file reads, search results) are
pruned to save space. **Do not re-read files just because you can't see the full output
in context.** Instead:

- Trust `save_finding` artifacts — they survive compaction by design.
- Check the plan status and session state summaries — they reflect your actual progress.
- Only re-read a file if you need a specific detail you did not already capture in a
  finding or plan step.
- If you find yourself re-reading 3+ files immediately after compaction, stop and
  synthesize from what you already know.

## Skills

Skills are reusable instruction sets that guide how to approach specific tasks.
Available skills are listed in the "Available Skills" section of the system prompt.

When you see a matching skill:
1. Call `invoke_skill` with the skill name before starting work.
2. Follow the skill's instructions as guidance for the task.
3. Skills may reference supporting files available through `skill://<skill-name>/...`
   in the readonly file tools after you invoke the skill.

Arguments: pass arguments to `invoke_skill` using the `arguments` parameter. The
skill can reference them as `$ARGUMENTS` (full string), `$0`/`$1`/`$N` (positional),
or `${SKILL_DIR}` (skill directory path).
