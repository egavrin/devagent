/**
 * LSP integration — client and code analysis tools.
 */

export { LSPClient } from "./client.js";
export type {
  LSPClientOptions,
  DiagnosticResult,
  SymbolResult,
  LocationResult,
} from "./client.js";

export { createLSPTools, createRoutingLSPTools } from "./tools.js";
export type { LSPClientResolver } from "./tools.js";
