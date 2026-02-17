## Search Strategy

When exploring an unfamiliar codebase:
1. `find_files` with broad patterns first (e.g. `**/*.ts`) to understand structure.
2. `search_files` with `file_pattern` to locate specific symbols or patterns.
3. `read_file` to examine relevant files. Use line ranges for files over 200 lines.

Do NOT read entire files speculatively. Search first, then read targeted sections.

## Editing Strategy

- For surgical edits (a few lines): use `replace_in_file`. Always `read_file` first to get exact text.
- For new files: use `write_file`.
- For rewriting most of a file: use `write_file` to overwrite.
- After editing, do NOT re-read the file to verify — the tool confirms success or failure.

## Shell Commands

- Use `run_command` for builds, tests, linting.
- Prefer targeted test commands over running the full suite.
- If a command times out, try a more targeted version.

## Error Recovery

When a tool call fails:
1. Read the error message carefully.
2. If `replace_in_file` fails with "not found", re-read the file and retry with correct text.
3. If `run_command` fails, check stderr. Do not retry the identical command.
4. After 3 consecutive failures on the same operation, report the issue to the user.
