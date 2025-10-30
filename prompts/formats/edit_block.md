# Edit Block Format

## Purpose

Use this format for making specific, targeted changes to existing files.

## Format Specification

```
FILE: {file_path}
OPERATION: {operation_type}
LOCATION: {location_identifier}
---
<<<ORIGINAL>>>
{original_code_block}
<<<MODIFIED>>>
{modified_code_block}
```

## Operation Types

- **REPLACE**: Replace existing code block
- **INSERT_AFTER**: Insert new code after specified location
- **INSERT_BEFORE**: Insert new code before specified location
- **DELETE**: Remove code block

## Location Identifiers

- Line numbers: `lines 45-67`
- Function/class names: `function: calculate_total`
- Unique code patterns: `pattern: "if user.is_authenticated"`
- Markers: `after: # TODO: Add validation`

## Guidelines

1. **Context**: Include enough surrounding code for unambiguous location
2. **Precision**: Make changes as focused as possible
3. **Validation**: Ensure indentation matches surrounding code
4. **Multiple Edits**: Use separate blocks for each logical change

## Examples

### Replace Example
```
FILE: src/calculator.py
OPERATION: REPLACE
LOCATION: function: add
---
<<<ORIGINAL>>>
def add(a, b):
    return a + b
<<<MODIFIED>>>
def add(a: float, b: float) -> float:
    """Add two numbers."""
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("Arguments must be numeric")
    return a + b
```

### Insert Example
```
FILE: src/main.py
OPERATION: INSERT_AFTER
LOCATION: line 10
---
<<<ORIGINAL>>>
import sys
import os
<<<MODIFIED>>>
import sys
import os
import logging
```

### Delete Example
```
FILE: src/old_module.py
OPERATION: DELETE
LOCATION: lines 50-75
---
<<<ORIGINAL>>>
def deprecated_function():
    # Old implementation
    pass
<<<MODIFIED>>>
# Code removed
```

## Best Practices

- Keep edit blocks small and focused
- Preserve surrounding code structure
- Maintain consistent indentation
- Include comments explaining complex changes
- Test after each edit block
