---
title: JSDoc Required for Public API Methods
severity: error
applies_to: "stdlib/.*\\.ets$"
category: documentation
---

# JSDoc Required for Public API Methods

All public methods in the standard library must have complete JSDoc documentation.

## Applies To

- **Scope**: `stdlib/**/*.ets`
- **Target**: All `public` methods in exported classes
- **Exemptions**: None - this is mandatory for stdlib

## Requirements

Public methods must have JSDoc with:

1. **Description** - What the method does
2. **@param** - For each parameter (name, type, description)
3. **@returns** - Return type and description (if not void)
4. **@throws** - Any exceptions thrown
5. **@example** - Usage example (recommended)

## Rationale

The standard library is used by all eTS developers. Complete documentation:
- Helps IDE autocomplete
- Enables better code review
- Reduces support burden
- Improves developer experience

## Examples

### ❌ Violation: Missing JSDoc

```typescript
export class ArrayList<T> {
  public insert(index: number, element: T): void {
    if (index < 0 || index > this.size()) {
      throw new Error("Index out of bounds");
    }
    this.data.splice(index, 0, element);
  }
}
```

### ✅ Compliant: Complete JSDoc

```typescript
export class ArrayList<T> {
  /**
   * Inserts an element at the specified position in this list.
   * Shifts the element currently at that position (if any) and any
   * subsequent elements to the right.
   *
   * @param index - Position where the element should be inserted (0-based)
   * @param element - Element to be inserted
   * @throws Error if index is out of bounds (index < 0 || index > size())
   * @example
   * ```
   * const list = new ArrayList<number>();
   * list.add(1);
   * list.add(3);
   * list.insert(1, 2); // list is now [1, 2, 3]
   * ```
   */
  public insert(index: number, element: T): void {
    if (index < 0 || index > this.size()) {
      throw new Error("Index out of bounds");
    }
    this.data.splice(index, 0, element);
  }
}
```

## Detection

Look for lines matching the pattern:
- `public <methodName>(...): <returnType> {`
- Check if there is a JSDoc comment block (`/**`) immediately above
- Verify the JSDoc contains required tags

## Common Mistakes

1. **Missing @throws** - Method throws but doesn't document it
2. **Incomplete @param** - Parameter listed but no description
3. **Generic descriptions** - "Does stuff" instead of explaining behavior
4. **No @returns** - Missing return value documentation

## Enforcement

This rule is enforced during:
- Code review (automated via this check)
- CI/CD pipeline
- Pre-merge validation

All violations must be fixed before merge to main.
