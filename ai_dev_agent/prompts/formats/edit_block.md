# Edit Block Format

When editing code, use this SEARCH/REPLACE format for precise, reliable edits:

```
<<<<<<< SEARCH
[exact code to be replaced - copy verbatim from file]
=======
[new code to replace it with]
>>>>>>> REPLACE
```

## Critical Rules

**MOST IMPORTANT:** The SEARCH block must match the file content **EXACTLY**:
- ✅ Copy the exact text from the file using READ tool first
- ✅ Preserve all whitespace: spaces, tabs, blank lines
- ✅ Include the exact indentation (don't convert tabs↔spaces)
- ✅ Copy complete lines - don't truncate or paraphrase
- ❌ Do NOT modify, clean up, or "fix" the search text
- ❌ Do NOT add comments that aren't in the original

**Why this matters:** The tool validates ALL blocks before applying ANY changes. If even one SEARCH block doesn't match exactly, NO changes are applied. This prevents partial file modifications.

## Guidelines

1. **Always READ first**: Use the READ tool to see current file content before editing
2. **Copy exact text**: Select and copy the exact text you want to replace
3. **Include context**: Add 1-2 surrounding lines for unique matching
4. **One logical change per block**: Each block should make one cohesive change
5. **Multiple blocks OK**: You can include multiple SEARCH/REPLACE blocks in one edit

## Creating New Files

To create a new file with multiple sections, use empty SEARCH blocks:

```python
<<<<<<< SEARCH
=======
def first_function():
    pass
>>>>>>> REPLACE

<<<<<<< SEARCH
=======

def second_function():
    pass
>>>>>>> REPLACE
```

## Examples

### ✅ CORRECT - Exact match
```python
# File contains:
#   def calculate(x):
#       return x * 2

<<<<<<< SEARCH
def calculate(x):
    return x * 2
=======
def calculate(x):
    """Double the input value."""
    return x * 2
>>>>>>> REPLACE
```

### ❌ WRONG - Modified whitespace
```python
# File has 4 spaces, but you wrote 2 spaces:
<<<<<<< SEARCH
def calculate(x):
  return x * 2  # ❌ Wrong indentation!
=======
def calculate(x):
    return x * 2
>>>>>>> REPLACE
```

### ❌ WRONG - Added comment not in file
```python
<<<<<<< SEARCH
def calculate(x):  # ❌ This comment doesn't exist in file!
    return x * 2
=======
def calculate(x):
    return x * 2
>>>>>>> REPLACE
```

### ✅ CORRECT - Multiple blocks
```python
<<<<<<< SEARCH
def func_one():
    pass
=======
def func_one():
    return 1
>>>>>>> REPLACE

<<<<<<< SEARCH
def func_two():
    pass
=======
def func_two():
    return 2
>>>>>>> REPLACE
```

## Error Recovery

If you get "SEARCH text not found" error:
1. **Check the error message** - it shows what you searched for with visible whitespace (·=space, →=tab, ⏎=newline)
2. **Look at "Closest match found"** - compare it with your search to spot differences
3. **READ the file again** - ensure you have current content
4. **Copy exact text** - don't type it from memory or paraphrase

## Common Mistakes

1. **Paraphrasing code**: Don't clean up or rephrase - copy exactly as-is
2. **Wrong whitespace**: Tabs vs spaces, wrong indentation level
3. **Missing lines**: Include all lines in the section you want to match
4. **Stale content**: File changed since you last read it - READ again
5. **Multiple similar blocks**: Add more context lines to make match unique
