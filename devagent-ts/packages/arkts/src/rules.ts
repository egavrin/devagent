/**
 * ArkTS linter types — mirrors the ets2panda/linter ProblemInfo structure.
 *
 * The real linter (arkcompiler_ets_frontend/ets2panda/linter) is a 140+ rule
 * AST-based tool invoked as a subprocess. These types represent its JSON output.
 */

// ─── Linter Output Types ────────────────────────────────────

/**
 * ProblemSeverity enum — matches ets2panda/linter's ProblemSeverity.
 * In IDE-interactive JSON output, severity is a number: 1=WARNING, 2=ERROR.
 */
export const ProblemSeverity = {
  WARNING: 1,
  ERROR: 2,
} as const;

/** A single problem reported by ets2panda/linter (matches its ProblemInfo). */
export interface TsLinterProblem {
  readonly line: number;
  readonly column: number;
  readonly endLine: number;
  readonly endColumn: number;
  /** Start offset in the source text. */
  readonly start?: number;
  /** End offset in the source text. */
  readonly end?: number;
  /** AST node type (e.g. "Identifier", "VariableDeclaration"). */
  readonly type?: string;
  /** Numeric severity: 1=WARNING, 2=ERROR (see ProblemSeverity). */
  readonly severity: number;
  /** Internal fault ID for the rule. */
  readonly faultId?: number;
  readonly problem: string;
  readonly suggest: string;
  readonly rule: string;
  /** Internal rule tag number. */
  readonly ruleTag?: number;
  readonly autofixable?: boolean;
  readonly autofix?: ReadonlyArray<TsLinterAutofix>;
}

/** An autofix suggestion from the linter. */
export interface TsLinterAutofix {
  readonly replacementText: string;
  readonly line: number;
  readonly column: number;
  readonly endLine: number;
  readonly endColumn: number;
}

/** Parsed output from the IDE-interactive mode (one JSON line per file). */
export interface TsLinterFileResult {
  readonly filePath: string;
  readonly problems: ReadonlyArray<TsLinterProblem>;
}

// ─── Severity Mapping ───────────────────────────────────────

/**
 * Map tslinter numeric severity to DevAgent severity format.
 * IDE-interactive mode uses numbers: 2=ERROR, 1=WARNING.
 * Also handles string values for backwards compatibility.
 */
export function mapSeverity(severity: number | string): "error" | "warning" {
  if (typeof severity === "number") {
    return severity >= ProblemSeverity.ERROR ? "error" : "warning";
  }
  return severity.toUpperCase() === "ERROR" ? "error" : "warning";
}

/**
 * Parse a single JSON line from the tslinter IDE-interactive output.
 * Returns null if the line is not valid JSON or doesn't have the expected shape.
 */
export function parseTsLinterLine(line: string): TsLinterFileResult | null {
  const trimmed = line.trim();
  if (!trimmed || trimmed.startsWith("{\"content\":")) {
    // Skip control messages like {"content":"report finish","messageType":1,...}
    return null;
  }

  try {
    const parsed = JSON.parse(trimmed) as Record<string, unknown>;
    if (typeof parsed["filePath"] !== "string" || !Array.isArray(parsed["problems"])) {
      return null;
    }
    return {
      filePath: parsed["filePath"] as string,
      problems: parsed["problems"] as TsLinterProblem[],
    };
  } catch {
    return null;
  }
}
