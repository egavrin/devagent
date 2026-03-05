/**
 * @devagent/arkts — ArkTS linter integration.
 *
 * Wraps the ets2panda/linter (tslinter) as a subprocess and provides
 * a DiagnosticProvider for the DevAgent DoubleCheck validation pipeline.
 */

// DiagnosticProvider integration
export { createArkTSDiagnosticProvider } from "./diagnostic-provider.js";
