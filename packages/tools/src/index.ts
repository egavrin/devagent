/**
 * @devagent/tools — Tool registry and built-in tools.
 */

export { ToolRegistry } from "./registry.js";
export {
  builtinTools,
  readFileTool,
  writeFileTool,
  replaceInFileTool,
  findFilesTool,
  searchFilesTool,
  runCommandTool,
  gitStatusTool,
  gitDiffTool,
  gitCommitTool,
  FileTime,
} from "./builtins/index.js";

// LSP integration
export { LSPClient, createLSPTools } from "./lsp/index.js";
export type {
  LSPClientOptions,
  DiagnosticResult,
  SymbolResult,
  LocationResult,
} from "./lsp/index.js";

// MCP integration
export { McpHub } from "./mcp/index.js";
export type {
  McpServerConfig,
  McpConfig,
  McpToolDefinition,
  McpServer,
  McpHubOptions,
} from "./mcp/index.js";

import { ToolRegistry } from "./registry.js";
import { builtinTools } from "./builtins/index.js";

/**
 * Create a registry with all built-in tools registered.
 */
export function createDefaultToolRegistry(): ToolRegistry {
  const registry = new ToolRegistry();
  for (const tool of builtinTools) {
    registry.register(tool);
  }
  return registry;
}
