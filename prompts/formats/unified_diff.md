# Unified Diff Format

## Purpose

Use unified diff format for reviewing changes, creating patches, or showing precise modifications.

## Format Specification

```diff
--- a/{original_file_path}
+++ b/{modified_file_path}
@@ -{start_line},{num_lines} +{start_line},{num_lines} @@ {optional_context}
 {context_line}
-{removed_line}
+{added_line}
 {context_line}
```

## Symbol Meanings

- `---`: Original file path
- `+++`: Modified file path
- `@@`: Hunk header (location in file)
- ` ` (space): Context line (unchanged)
- `-`: Line removed
- `+`: Line added

## Guidelines

1. **Context Lines**: Include 3 lines of context before/after changes
2. **Hunk Headers**: Must accurately reflect line numbers
3. **Whitespace**: Preserve exact whitespace (critical for Python)
4. **Multiple Changes**: Can include multiple hunks in one diff

## Example

```diff
--- a/src/utils/calculator.py
+++ b/src/utils/calculator.py
@@ -10,7 +10,7 @@ class Calculator:
     def __init__(self):
         self.result = 0

-    def add(self, x, y):
+    def add(self, x: float, y: float) -> float:
         """Add two numbers."""
         return x + y

@@ -25,6 +25,11 @@ class Calculator:
     def divide(self, x, y):
         """Divide x by y."""
         if y == 0:
-            raise ValueError("Cannot divide by zero")
+            raise ZeroDivisionError("Cannot divide by zero")
         return x / y
+
+    def power(self, x: float, y: float) -> float:
+        """Raise x to the power of y."""
+        return x ** y
```

## Creating Diffs

### From Git
```bash
git diff HEAD~1 HEAD -- src/file.py
git diff --staged
git show HEAD
```

### Programmatically
```python
import difflib

def create_unified_diff(original, modified, filename):
    original_lines = original.splitlines(keepends=True)
    modified_lines = modified.splitlines(keepends=True)

    diff = difflib.unified_diff(
        original_lines,
        modified_lines,
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
        lineterm=""
    )

    return "".join(diff)
```

## Best Practices

- Always include file paths in headers
- Ensure line numbers are accurate
- Preserve exact indentation
- Include enough context for clarity
- Group related changes in single hunks when possible
- Split unrelated changes into separate hunks
