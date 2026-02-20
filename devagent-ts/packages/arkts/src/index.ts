/**
 * @devagent/arkts — ArkTS linter integration.
 *
 * Wraps the ets2panda/linter (tslinter) as a subprocess and provides
 * a DiagnosticProvider for the DevAgent DoubleCheck validation pipeline.
 */

// Types for tslinter output
export type {
  TsLinterProblem,
  TsLinterAutofix,
  TsLinterFileResult,
} from "./rules.js";

// Parsing utilities and constants
export { parseTsLinterLine, mapSeverity, ProblemSeverity } from "./rules.js";

// Linter engine (subprocess wrapper)
export { ArkTSLinter, isTsLinterAvailable } from "./linter.js";
export type { ArkTSLinterOptions } from "./linter.js";

// DiagnosticProvider integration
export { createArkTSDiagnosticProvider } from "./diagnostic-provider.js";
export type { Diagnostic, DiagnosticProviderFn } from "./diagnostic-provider.js";
