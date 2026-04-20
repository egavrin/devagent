/**
 * ArkTS Diagnostic Provider — bridges the tslinter subprocess to
 * the DevAgent DoubleCheck diagnostic pipeline.
 *
 * Uses structural typing to match the DiagnosticProvider signature
 * from @devagent/runtime without importing it (avoids circular deps).
 */

import { ArkTSLinter, isTsLinterAvailable } from "./linter.js";
import { mapSeverity } from "./rules.js";
import type { ArkTSConfig } from "@devagent/runtime";

// ─── DiagnosticProvider-compatible type (structural match) ──

interface Diagnostic {
  readonly message: string;
  readonly severity: string;
}

type DiagnosticProviderFn = (filePath: string) => Promise<ReadonlyArray<Diagnostic>>;

// ─── Factory ────────────────────────────────────────────────

/**
 * Create a DiagnosticProvider that runs the ets2panda/linter (tslinter)
 * on .ets files and returns structured diagnostics.
 *
 * Returns null if the linter is not available (not built or path missing).
 */
export function createArkTSDiagnosticProvider(
  config: ArkTSConfig,
): DiagnosticProviderFn | null {
  if (!config.linterPath) {
    return null;
  }

  if (!isTsLinterAvailable(config.linterPath)) {
    return null;
  }

  const linter = new ArkTSLinter({
    linterPath: config.linterPath,
    arkts2: config.strictMode,
    timeout: 60_000,
  });

  return async (filePath: string): Promise<ReadonlyArray<Diagnostic>> => {
    // Only lint .ets files
    if (!filePath.endsWith(".ets")) {
      return [];
    }

    try {
      const problems = await linter.lintFile(filePath);
      return problems.map((p) => ({
        message: `${filePath}:${p.line}:${p.column}: [${p.rule}] ${p.problem}${p.suggest ? ` — ${p.suggest}` : ""}`,
        severity: mapSeverity(p.severity),
      }));
    } catch {
      // Subprocess failure — don't crash the agent, just skip diagnostics
      return [];
    }
  };
}
