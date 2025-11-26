# SEARCH/REPLACE Edit Format

All file edits MUST use the **SEARCH/REPLACE block** format (git merge conflict style markers).

## Format

```
path/to/file.py
```python
<<<<<<< SEARCH
exact content to find
=======
replacement content
>>>>>>> REPLACE
```

## Format Rules (8 Steps)

Every SEARCH/REPLACE block **MUST** follow this exact format:

1. **File path** alone on a line (no decorations, no backticks)
2. **Opening fence** with language: ` ```python ` or ` ```markdown ` etc.
3. **SEARCH marker**: `<<<<<<< SEARCH`
4. **Content to find** (EXACT match from READ output, or **empty for insertions**)
5. **Divider**: `=======`
6. **Replacement content** (what to put in place of SEARCH content)
7. **REPLACE marker**: `>>>>>>> REPLACE`
8. **Closing fence**: ` ``` `

## Critical Rules

- **ALWAYS READ the file first** before editing (use `{{TOOL_READ}}`)
- **SEARCH content MUST be copied EXACTLY** from READ output
- **NEVER invent or guess** what the file contains
- **For insertions, use EMPTY SEARCH** (appends to file — always works)
- **For new files, use EMPTY SEARCH** with non-existent path
- **For deletions, use EMPTY REPLACE** to remove matched content

## Operations

### 1. Replace Existing Content

```
config.py
```python
<<<<<<< SEARCH
DEBUG = False
=======
DEBUG = True
>>>>>>> REPLACE
```

### 2. Insert New Content (append at end)

Use **empty SEARCH** to append content:

```
README.md
```markdown
<<<<<<< SEARCH
=======

## New Section
Content appended at end.
>>>>>>> REPLACE
```

**Empty SEARCH = append at end of file. ALWAYS works. No context matching needed.**

### 3. Delete Content

Use **empty REPLACE** to delete content:

```
src/debug.py
```python
<<<<<<< SEARCH
print("DEBUG:", value)
=======
>>>>>>> REPLACE
```

### 4. Create New File

Use **empty SEARCH** with non-existent path:

```
src/new_helper.py
```python
<<<<<<< SEARCH
=======
def helper():
    return "ok"
>>>>>>> REPLACE
```

### 5. Multiple Changes to Same File

Use multiple SEARCH/REPLACE blocks:

```
config.py
```python
<<<<<<< SEARCH
DEBUG = False
=======
DEBUG = True
>>>>>>> REPLACE
```

config.py
```python
<<<<<<< SEARCH
TIMEOUT = 30
=======
TIMEOUT = 60
>>>>>>> REPLACE
```

## Common Errors and Fixes

### Error: "SEARCH content not found"

**Cause**: The SEARCH content doesn't match the actual file content.

**Fix**:
1. Use `{{TOOL_READ}}` to view current file content
2. Copy the **exact lines** (including whitespace) into SEARCH
3. Never guess or paraphrase

### Error: "No SEARCH/REPLACE blocks found"

**Cause**: Missing fence markers or wrong format.

**Fix**: Ensure format includes:
- File path on its own line
- Opening fence with language (` ```python `)
- `<<<<<<< SEARCH` marker
- `=======` divider
- `>>>>>>> REPLACE` marker
- Closing fence (` ``` `)

## The #1 Cause of Failures: Invented SEARCH Content

**WRONG** — Inventing anchor content:
```
README.md
```markdown
<<<<<<< SEARCH
## License
MIT License
=======
## License
MIT License

## New Section
>>>>>>> REPLACE
```

**Why it fails**: The SEARCH content was guessed, not copied from READ output.

**RIGHT** — Empty SEARCH for insertions:
```
README.md
```markdown
<<<<<<< SEARCH
=======

## New Section
Content here.
>>>>>>> REPLACE
```

**Why it works**: Empty SEARCH = append. Always succeeds.

## Error Recovery Protocol

When EDIT fails:
1. **First failure**: RE-READ the file, COPY exact content, retry
2. **Second failure**: Try different approach — use **empty SEARCH** for insertions
3. **Third failure**: STOP and explain what's blocking you

**NEVER retry with the same SEARCH content that already failed.**
