# Apply Patch Format

All edits must use the canonical `apply_patch` structure. Every payload starts with `*** Begin Patch`, contains one or more actions, and ends with `*** End Patch`.

```
*** Begin Patch
*** Update File: relative/path.py
*** Move to: optional/new/path.py
@@
-old line copied verbatim
+replacement line
*** End Patch
```

## Action Types

1. **Update File**
   ```
   *** Update File: src/app.py
   @@
   -def ping():
   -    return "pong"
   +def ping() -> str:
   +    return "pong!"
   ```
   - Optional `*** Move to: ...` immediately after the header renames the file.
   - Each `@@` chunk must include the exact `-` lines that currently exist plus the new `+` lines.

2. **Add File**
   ```
   *** Add File: src/new_helper.py
   +def helper():
   +    return "ok"
   ```
   - Prefix every line with `+`. Include blank `+` lines for spacing.

3. **Delete File**
   ```
   *** Delete File: scripts/old_task.sh
   ```

## Critical Rules

- **Headers always use colons**: `*** Update File: path`, `*** Add File: path`, `*** Delete File: path`, and `*** Move to: target`. Missing the colon makes the entire patch invalid.
- **Exact matches only**: Copy every removed line (`-`) exactly, including whitespace, punctuation, and comments.
- **Preflight validation**: DevAgent runs a dry-run before applying patches; if your `-` lines don’t match current files, the entire edit is rejected so re-read the file and rebuild the patch.
- **Read before editing**: Use the READ tool immediately before creating the patch to avoid stale context.
- **One chunk per region**: Use multiple `@@` sections if you edit distant parts of the same file.
- **Maintain order**: Actions execute sequentially; list updates before deletes if they depend on earlier context.
- **EOF handling**: Append `*** End of File` within a chunk if you modify the file footer or final newline.

## Before Sending a Patch
1. Re-run `{tool_read}` on every file you plan to edit so the patch uses fresh context.
2. Build a **single** `patch` string that wraps all actions between `*** Begin Patch` and `*** End Patch`.
3. Confirm every header includes the colon (`*** Update File: path`, etc.) and optional `*** Move to: target`.
4. Double-check each `@@` chunk: the `-` lines must match the file exactly and the `+` lines contain the intended changes.
5. If validation fails (missing colon, context mismatch, move conflict), re-read the file, rebuild the chunk, and resend.

## Examples

### Update + Move
```
*** Begin Patch
*** Update File: pkg/old_name.py
*** Move to: pkg/new_name.py
@@
-IDENTITY = "old"
+IDENTITY = "new"
*** End Patch
```

### Multiple Updates
```
*** Begin Patch
*** Update File: config.py
@@
-DEBUG = False
+DEBUG = True
@@
-TIMEOUT = 30
+TIMEOUT = 60
*** End Patch
```

### Add + Delete Together
```
*** Begin Patch
*** Add File: scripts/hello.sh
+#!/usr/bin/env bash
+echo "Hello"
*** Delete File: scripts/legacy.sh
*** End Patch
```

## Error Recovery

If the tool reports context or file errors:
1. **Re-read the file** and rebuild the chunk from current content.
2. **Verify paths** are workspace-relative and case-sensitive.
3. **Check move targets** – the destination must not already exist.
4. **Ensure every action is enclosed** between `*** Begin Patch` and `*** End Patch`.

## Common Errors and Fixes

### Error: "Missing colon in directive"
**Cause**: Directive header is missing the required colon.

**Wrong**:
```
*** Update File README.md
```

**Correct**:
```
*** Update File: README.md
```

Note: DevAgent will auto-correct minor colon omissions, but always include them for reliability.

### Error: "context not found in 'file.py'"
**Cause**: The `-` lines in your patch don't match the actual file content.

**Fix steps**:
1. Use `{tool_read}` to view current file content
2. Copy the exact lines (including whitespace) into your `-` lines
3. Never guess or paraphrase – whitespace and punctuation must be exact

**Wrong** (guessed content):
```
*** Update File: config.py
@@
-debug = false
+debug = true
```

**Correct** (after reading file shows `DEBUG = False`):
```
*** Update File: config.py
@@
-DEBUG = False
+DEBUG = True
```

Note: DevAgent has whitespace tolerance for trailing spaces and uniform indentation differences, but exact matches are most reliable.

## Insertion Operations

For pure insertions (adding new content without replacing existing lines):

### Append to End of File
Use `@@ EOF` or the `*** End of File` marker:
```
*** Begin Patch
*** Update File: README.md
@@
+
+## New Section
+Content appended at end.
*** End of File
*** End Patch
```

### Insert Near Existing Content
Include a small anchor of existing lines:
```
*** Begin Patch
*** Update File: README.md
@@
-## Existing Section
+## Existing Section
+
+## New Section (inserted after)
+New content here.
*** End Patch
```
