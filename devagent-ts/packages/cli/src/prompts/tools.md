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

## Editing Strategy

- Read before edit: use `read_file` first so edits match exact current text.
- Use `replace_in_file` for surgical edits and `write_file` for new files/full rewrites.
- If `replace_in_file` fails with "not found", re-read and retry against fresh content.
- For large mechanical multi-file edits, use `run_command` only when it is clearly more
  efficient and still safe.

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
