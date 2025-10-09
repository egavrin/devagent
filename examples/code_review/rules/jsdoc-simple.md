---
title: JSDoc Required for Exported Functions
severity: error
applies_to: "src/**/*.js"
---

# JSDoc Required for Exported Functions

All exported functions must have JSDoc comments documenting:
- Function purpose
- Parameters (with types)
- Return value (with type)

## Rationale

Documentation helps maintainers understand function contracts without reading implementation.

## Examples

### ❌ Violation

```javascript
export function calculateDiscount(price, percentage) {
  return price * (percentage / 100);
}
```

### ✅ Compliant

```javascript
/**
 * Calculate discount amount based on percentage
 * @param {number} price - Original price
 * @param {number} percentage - Discount percentage (0-100)
 * @returns {number} Discount amount
 */
export function calculateDiscount(price, percentage) {
  return price * (percentage / 100);
}
```

## Enforcement

- Check all lines starting with `export function`
- Verify JSDoc comment block immediately preceding
- JSDoc must include `@param` for each parameter and `@returns`
