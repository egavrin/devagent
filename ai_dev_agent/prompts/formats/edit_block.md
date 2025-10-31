# Edit Block Format

When editing code, use this format:

```
<<<<<<< SEARCH
[code to be replaced]
=======
[new code]
>>>>>>> REPLACE
```

## Guidelines
- Show complete context around changes
- Use exact indentation and whitespace
- Include enough context for unique matching
- Make one logical change per block

## Example
```python
<<<<<<< SEARCH
def old_function():
    return 42
=======
def new_function():
    return 43
>>>>>>> REPLACE
```
