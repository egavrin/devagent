/**
 * Tool registry — manages tool registration and lookup.
 * Fail fast: throws on duplicate names or missing tools.
 */

import type { ToolSpec, ToolCategory } from "@devagent/core";
import { ToolNotFoundError } from "@devagent/core";

export class ToolRegistry {
  private readonly tools = new Map<string, ToolSpec>();

  register(tool: ToolSpec): void {
    if (this.tools.has(tool.name)) {
      throw new Error(`Tool "${tool.name}" is already registered`);
    }
    this.tools.set(tool.name, tool);
  }

  get(name: string): ToolSpec {
    const tool = this.tools.get(name);
    if (!tool) {
      const available = this.list().join(", ");
      throw new ToolNotFoundError(name);
    }
    return tool;
  }

  has(name: string): boolean {
    return this.tools.has(name);
  }

  list(): ReadonlyArray<string> {
    return Array.from(this.tools.keys());
  }

  getAll(): ReadonlyArray<ToolSpec> {
    return Array.from(this.tools.values());
  }

  getByCategory(category: ToolCategory): ReadonlyArray<ToolSpec> {
    return Array.from(this.tools.values()).filter(
      (t) => t.category === category,
    );
  }

  getReadOnly(): ReadonlyArray<ToolSpec> {
    return this.getByCategory("readonly");
  }
}
