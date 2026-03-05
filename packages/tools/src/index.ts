/**
 * @devagent/tools — Tool registry and built-in tools.
 */

export { ToolRegistry } from "./registry.js";
export { builtinTools } from "./builtins/index.js";

// LSP integration
export { LSPClient, createRoutingLSPTools } from "./lsp/index.js";

// Shared utilities
export { spawnAndCapture } from "./builtins/spawn-capture.js";

// MCP integration
export { McpHub } from "./mcp/index.js";
export type { McpServer } from "./mcp/index.js";

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
