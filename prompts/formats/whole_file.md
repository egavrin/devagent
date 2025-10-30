# Whole File Format

## Purpose

Use this format when creating new files or completely replacing file contents.

## Format Specification

```
FILE: {file_path}
LANGUAGE: {language}
PURPOSE: {brief_description}
---
{full_file_content}
```

## Guidelines

1. **When to Use**:
   - Creating new files from scratch
   - Complete file rewrites
   - Files under 200 lines
   - When most of the file needs changes

2. **When NOT to Use**:
   - Making small edits to large files
   - Changing a few lines in existing code
   - Files over 500 lines (use edit_block instead)

## Example

```
FILE: src/utils/validator.py
LANGUAGE: python
PURPOSE: Input validation utilities
---
"""Input validation utilities for the application."""

from typing import Any, Dict, List, Optional


class Validator:
    """Validates input data against defined rules."""

    def __init__(self, rules: Optional[Dict[str, Any]] = None):
        """Initialize validator with optional rules."""
        self.rules = rules or {}

    def validate(self, data: Any) -> bool:
        """Validate data against configured rules."""
        if not self.rules:
            return True

        # Implementation here
        return self._apply_rules(data)

    def _apply_rules(self, data: Any) -> bool:
        """Apply validation rules to data."""
        # Rule application logic
        pass
```

## Best Practices

- Include file header comments/docstrings
- Follow language-specific conventions
- Ensure proper imports are included
- Add type hints for Python
- Include error handling
- Keep files focused on a single responsibility
