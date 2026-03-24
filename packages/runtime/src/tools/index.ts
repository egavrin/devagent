/**
 * @devagent/runtime — Tool registry and built-in tools.
 */

export { ToolRegistry } from "./registry.js";
export { builtinTools } from "./builtins/index.js";
export {
  createReadFileTool,
  createFindFilesTool,
  createSearchFilesTool,
} from "./builtins/index.js";

// LSP integration
export { LSPClient, createRoutingLSPTools } from "./lsp/index.js";

// Shared utilities
export { spawnAndCapture } from "./builtins/spawn-capture.js";

import type { ToolSpec } from "../core/index.js";
import { ToolRegistry } from "./registry.js";
import { builtinTools } from "./builtins/index.js";

export interface DefaultToolRegistryOptions {
  readonly overrides?: ReadonlyArray<ToolSpec>;
}

/**
 * Create a registry with all built-in tools registered.
 */
export function createDefaultToolRegistry(options?: DefaultToolRegistryOptions): ToolRegistry {
  const registry = new ToolRegistry();
  const overrideMap = new Map(
    (options?.overrides ?? []).map((tool) => [tool.name, tool]),
  );
  for (const tool of builtinTools) {
    registry.register(overrideMap.get(tool.name) ?? tool);
  }
  for (const tool of options?.overrides ?? []) {
    if (!registry.has(tool.name)) {
      registry.register(tool);
    }
  }
  return registry;
}
