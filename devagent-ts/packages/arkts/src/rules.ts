/**
 * ArkTS lint rules — built-in rule implementations.
 * Based on the official ets2panda/linter rule set.
 * Each rule analyzes TypeScript source text and reports violations.
 *
 * ArkTS-compatible: no `any`, explicit types.
 */

// ─── Rule Types ──────────────────────────────────────────────

export interface ArkTSViolation {
  readonly rule: string;
  readonly message: string;
  readonly line: number;
  readonly column: number;
  readonly severity: "error" | "warning";
  readonly fix?: string;
}

export interface ArkTSRule {
  readonly name: string;
  readonly description: string;
  readonly category: "type-system" | "operators" | "functions" | "modules" | "other";
  check(source: string, filePath: string): ReadonlyArray<ArkTSViolation>;
}

// ─── Rule Implementations ────────────────────────────────────

/**
 * arkts-no-any — Disallow `any` type.
 * ArkTS requires explicit types everywhere.
 */
export const noAnyRule: ArkTSRule = {
  name: "arkts-no-any",
  description: "No `any` type allowed (use explicit types)",
  category: "type-system",
  check(source: string): ReadonlyArray<ArkTSViolation> {
    const violations: ArkTSViolation[] = [];
    const lines = source.split("\n");

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i]!;
      // Skip comments
      const trimmed = line.trim();
      if (trimmed.startsWith("//") || trimmed.startsWith("*")) continue;

      // Match `: any` or `as any` or `<any>` patterns
      const patterns = [
        { regex: /:\s*any\b/, msg: "Type annotation uses `any`" },
        { regex: /\bas\s+any\b/, msg: "Type assertion uses `any`" },
        { regex: /<any>/, msg: "Generic parameter uses `any`" },
      ];

      for (const pattern of patterns) {
        const match = line.match(pattern.regex);
        if (match) {
          violations.push({
            rule: "arkts-no-any",
            message: pattern.msg,
            line: i + 1,
            column: (match.index ?? 0) + 1,
            severity: "error",
            fix: "Replace `any` with a specific type",
          });
        }
      }
    }

    return violations;
  },
};

/**
 * arkts-no-var — Disallow `var` declarations.
 */
export const noVarRule: ArkTSRule = {
  name: "arkts-no-var",
  description: "No `var` declarations (use `let` or `const`)",
  category: "other",
  check(source: string): ReadonlyArray<ArkTSViolation> {
    const violations: ArkTSViolation[] = [];
    const lines = source.split("\n");

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i]!;
      const trimmed = line.trim();
      if (trimmed.startsWith("//") || trimmed.startsWith("*")) continue;

      const match = line.match(/\bvar\s+/);
      if (match) {
        violations.push({
          rule: "arkts-no-var",
          message: "Use `let` or `const` instead of `var`",
          line: i + 1,
          column: (match.index ?? 0) + 1,
          severity: "error",
          fix: "Replace `var` with `let` or `const`",
        });
      }
    }

    return violations;
  },
};

/**
 * arkts-no-delete — Disallow `delete` operator.
 */
export const noDeleteRule: ArkTSRule = {
  name: "arkts-no-delete",
  description: "No `delete` operator",
  category: "operators",
  check(source: string): ReadonlyArray<ArkTSViolation> {
    const violations: ArkTSViolation[] = [];
    const lines = source.split("\n");

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i]!;
      const trimmed = line.trim();
      if (trimmed.startsWith("//") || trimmed.startsWith("*")) continue;

      const match = line.match(/\bdelete\s+/);
      if (match) {
        violations.push({
          rule: "arkts-no-delete",
          message: "`delete` operator is not supported in ArkTS",
          line: i + 1,
          column: (match.index ?? 0) + 1,
          severity: "error",
          fix: "Use Map.delete() or set property to undefined",
        });
      }
    }

    return violations;
  },
};

/**
 * arkts-no-in-operator — Disallow `in` operator.
 */
export const noInOperatorRule: ArkTSRule = {
  name: "arkts-no-in-operator",
  description: 'No `in` operator (use explicit property checks)',
  category: "operators",
  check(source: string): ReadonlyArray<ArkTSViolation> {
    const violations: ArkTSViolation[] = [];
    const lines = source.split("\n");

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i]!;
      const trimmed = line.trim();
      if (trimmed.startsWith("//") || trimmed.startsWith("*")) continue;
      // Skip for...in loops (different pattern)
      if (/\bfor\s*\(/.test(line)) continue;
      // Skip import statements
      if (/\bimport\b/.test(line)) continue;

      const match = line.match(/["']\w+["']\s+in\s+\w+/);
      if (match) {
        violations.push({
          rule: "arkts-no-in-operator",
          message: '`in` operator is not supported in ArkTS',
          line: i + 1,
          column: (match.index ?? 0) + 1,
          severity: "error",
          fix: "Use hasOwnProperty() or explicit property access",
        });
      }
    }

    return violations;
  },
};

/**
 * arkts-no-function-expressions — Disallow function expressions.
 */
export const noFunctionExpressionsRule: ArkTSRule = {
  name: "arkts-no-function-expressions",
  description: "No function expressions (use arrow functions or declarations)",
  category: "functions",
  check(source: string): ReadonlyArray<ArkTSViolation> {
    const violations: ArkTSViolation[] = [];
    const lines = source.split("\n");

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i]!;
      const trimmed = line.trim();
      if (trimmed.startsWith("//") || trimmed.startsWith("*")) continue;

      // Match `= function` or `: function` (function expressions)
      // But not `function name(` (function declarations) or `export function`
      const match = line.match(/[=:,]\s*function\s*[\(<]/);
      if (match) {
        violations.push({
          rule: "arkts-no-function-expressions",
          message: "Function expressions are not allowed in ArkTS",
          line: i + 1,
          column: (match.index ?? 0) + 1,
          severity: "error",
          fix: "Use arrow function syntax instead: `() => { ... }`",
        });
      }
    }

    return violations;
  },
};

/**
 * arkts-no-with-statement — Disallow `with` statement.
 */
export const noWithStatementRule: ArkTSRule = {
  name: "arkts-no-with-statement",
  description: "No `with` statement",
  category: "other",
  check(source: string): ReadonlyArray<ArkTSViolation> {
    const violations: ArkTSViolation[] = [];
    const lines = source.split("\n");

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i]!;
      const trimmed = line.trim();
      if (trimmed.startsWith("//") || trimmed.startsWith("*")) continue;

      const match = line.match(/\bwith\s*\(/);
      if (match) {
        violations.push({
          rule: "arkts-no-with-statement",
          message: "`with` statement is not supported in ArkTS",
          line: i + 1,
          column: (match.index ?? 0) + 1,
          severity: "error",
        });
      }
    }

    return violations;
  },
};

/**
 * arkts-no-globalthis — Disallow `globalThis`.
 */
export const noGlobalThisRule: ArkTSRule = {
  name: "arkts-no-globalthis",
  description: "No `globalThis` access",
  category: "other",
  check(source: string): ReadonlyArray<ArkTSViolation> {
    const violations: ArkTSViolation[] = [];
    const lines = source.split("\n");

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i]!;
      const trimmed = line.trim();
      if (trimmed.startsWith("//") || trimmed.startsWith("*")) continue;

      const match = line.match(/\bglobalThis\b/);
      if (match) {
        violations.push({
          rule: "arkts-no-globalthis",
          message: "`globalThis` is not available in ArkTS",
          line: i + 1,
          column: (match.index ?? 0) + 1,
          severity: "error",
        });
      }
    }

    return violations;
  },
};

/**
 * arkts-no-require — Disallow require() imports.
 */
export const noRequireRule: ArkTSRule = {
  name: "arkts-no-require",
  description: "No require() imports (use ES6 imports)",
  category: "modules",
  check(source: string): ReadonlyArray<ArkTSViolation> {
    const violations: ArkTSViolation[] = [];
    const lines = source.split("\n");

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i]!;
      const trimmed = line.trim();
      if (trimmed.startsWith("//") || trimmed.startsWith("*")) continue;

      const match = line.match(/\brequire\s*\(/);
      if (match) {
        violations.push({
          rule: "arkts-no-require",
          message: "`require()` is not supported in ArkTS — use ES6 imports",
          line: i + 1,
          column: (match.index ?? 0) + 1,
          severity: "error",
          fix: 'Use `import { ... } from "module"`',
        });
      }
    }

    return violations;
  },
};

/**
 * arkts-no-prototype-assignment — Disallow prototype mutations.
 */
export const noPrototypeAssignmentRule: ArkTSRule = {
  name: "arkts-no-prototype-assignment",
  description: "No prototype mutations",
  category: "functions",
  check(source: string): ReadonlyArray<ArkTSViolation> {
    const violations: ArkTSViolation[] = [];
    const lines = source.split("\n");

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i]!;
      const trimmed = line.trim();
      if (trimmed.startsWith("//") || trimmed.startsWith("*")) continue;

      const match = line.match(/\.prototype\s*[.=]/);
      if (match) {
        violations.push({
          rule: "arkts-no-prototype-assignment",
          message: "Prototype mutations are not supported in ArkTS",
          line: i + 1,
          column: (match.index ?? 0) + 1,
          severity: "error",
          fix: "Use class declarations with proper inheritance",
        });
      }
    }

    return violations;
  },
};

// ─── Rule Registry ───────────────────────────────────────────

/**
 * All built-in ArkTS rules.
 */
export const ALL_RULES: ReadonlyArray<ArkTSRule> = [
  noAnyRule,
  noVarRule,
  noDeleteRule,
  noInOperatorRule,
  noFunctionExpressionsRule,
  noWithStatementRule,
  noGlobalThisRule,
  noRequireRule,
  noPrototypeAssignmentRule,
];
