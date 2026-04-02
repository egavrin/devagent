/**
 * @devagent/runtime — Tool registry and built-in tools.
 */

export { ToolRegistry } from "./registry.js";
export type { DeferredToolStub } from "./registry.js";
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
import {
  readFileTool,
  writeFileTool,
  replaceInFileTool,
  findFilesTool,
  searchFilesTool,
  runCommandTool,
  gitStatusTool,
  gitDiffTool,
  gitCommitTool,
} from "./builtins/index.js";

const DEFAULT_BUILTIN_TOOLS: ReadonlyArray<ToolSpec> = [
  readFileTool,
  writeFileTool,
  replaceInFileTool,
  findFilesTool,
  searchFilesTool,
  runCommandTool,
  gitStatusTool,
  gitDiffTool,
  gitCommitTool,
];

export interface DefaultToolRegistryOptions {
  readonly overrides?: ReadonlyArray<ToolSpec>;
  /**
   * Tool names to register as deferred (stub only in prompt, full schema on demand).
   * Reduces prompt size by ~30-50% when specialized tools aren't needed.
   * If not specified, all tools are loaded eagerly (backward compatible).
   */
  readonly deferredToolNames?: ReadonlySet<string>;
}

/**
 * Create a registry with all built-in tools registered.
 * Tools listed in `deferredToolNames` are registered as deferred (stubs only).
 */
export function createDefaultToolRegistry(options?: DefaultToolRegistryOptions): ToolRegistry {
  const registry = new ToolRegistry();
  const overrideMap = new Map(
    (options?.overrides ?? []).map((tool) => [tool.name, tool]),
  );
  const deferredNames = options?.deferredToolNames ?? new Set<string>();

  for (const tool of DEFAULT_BUILTIN_TOOLS) {
    const effective = overrideMap.get(tool.name) ?? tool;
    if (deferredNames.has(tool.name)) {
      registry.registerDeferred(effective);
    } else {
      registry.register(effective);
    }
  }
  for (const tool of options?.overrides ?? []) {
    if (!registry.has(tool.name)) {
      registry.register(tool);
    }
  }
  return registry;
}
