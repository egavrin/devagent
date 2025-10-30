# Error Handling Guidelines

## Error Detection

When analyzing code or during implementation, actively look for:
- Missing error handling for external calls
- Unhandled exceptions
- Resource leaks (files, connections, locks)
- Race conditions
- Null/undefined reference errors
- Type mismatches
- Security vulnerabilities

## Error Response Strategies

### 1. Graceful Degradation
- Provide fallback behavior when possible
- Log errors for debugging
- Continue operation with reduced functionality

### 2. Fail Fast
- For critical errors, fail immediately with clear error messages
- Prevent data corruption or security breaches
- Ensure cleanup of resources

### 3. Retry Logic
- Implement exponential backoff for transient failures
- Set maximum retry limits
- Log retry attempts

## Error Reporting Format

When reporting errors, include:
```
ERROR: <brief description>
File: <file_path>:<line_number>
Context: <what was being attempted>
Cause: <root cause if known>
Impact: <what functionality is affected>
Suggested Fix: <recommended solution>
```

## Common Python Error Patterns

```python
# Good: Specific exception handling
try:
    result = risky_operation()
except ValueError as e:
    logger.error(f"Invalid value: {e}")
    return default_value
except IOError as e:
    logger.error(f"IO operation failed: {e}")
    raise
finally:
    cleanup_resources()

# Good: Context manager for resources
with open(file_path) as f:
    content = f.read()

# Good: Guard clauses
if not input_data:
    raise ValueError("Input data cannot be empty")
```

## Testing Error Conditions

Always test:
- Happy path
- Edge cases
- Error conditions
- Resource exhaustion
- Concurrent access issues
