/**
 * Plugin system — interfaces and manager for extending DevAgent.
 * Plugins register tools, commands, and subscribe to events.
 * ArkTS-compatible: no `any`, explicit types.
 */

import type { ToolSpec, DevAgentConfig } from "./types.js";
import type { EventBus } from "./events.js";

// ─── Plugin Types ────────────────────────────────────────────

export interface PluginContext {
  readonly bus: EventBus;
  readonly config: DevAgentConfig;
  readonly repoRoot: string;
}

export interface CommandHandler {
  readonly description: string;
  readonly usage?: string;
  execute(args: string, context: PluginContext): Promise<string>;
}

export interface Plugin {
  readonly name: string;
  readonly version: string;
  readonly description?: string;
  readonly tools?: ReadonlyArray<ToolSpec>;
  readonly commands?: Record<string, CommandHandler>;
  activate(context: PluginContext): void;
  deactivate?(): void;
}

// ─── Plugin Manager ─────────────────────────────────────────

export class PluginManager {
  private readonly plugins = new Map<string, Plugin>();
  private readonly commands = new Map<string, { plugin: string; handler: CommandHandler }>();
  private context: PluginContext | null = null;

  /**
   * Initialize the plugin manager with context.
   * Must be called before activating plugins.
   */
  init(context: PluginContext): void {
    this.context = context;
  }

  /**
   * Register and activate a plugin.
   * Throws if plugin name is already registered or context not initialized.
   */
  register(plugin: Plugin): void {
    if (!this.context) {
      throw new Error("PluginManager not initialized — call init() first");
    }
    if (this.plugins.has(plugin.name)) {
      throw new Error(`Plugin "${plugin.name}" is already registered`);
    }

    // Register commands from this plugin
    if (plugin.commands) {
      for (const [cmdName, handler] of Object.entries(plugin.commands)) {
        if (this.commands.has(cmdName)) {
          const existing = this.commands.get(cmdName)!;
          throw new Error(
            `Command "/${cmdName}" already registered by plugin "${existing.plugin}"`,
          );
        }
        this.commands.set(cmdName, { plugin: plugin.name, handler });
      }
    }

    this.plugins.set(plugin.name, plugin);

    // Activate the plugin
    plugin.activate(this.context);
  }

  /**
   * Deactivate and remove a plugin.
   */
  unregister(name: string): void {
    const plugin = this.plugins.get(name);
    if (!plugin) return;

    // Deactivate
    if (plugin.deactivate) {
      plugin.deactivate();
    }

    // Remove commands owned by this plugin
    for (const [cmdName, entry] of this.commands.entries()) {
      if (entry.plugin === name) {
        this.commands.delete(cmdName);
      }
    }

    this.plugins.delete(name);
  }

  /**
   * Get a registered plugin by name.
   */
  get(name: string): Plugin | undefined {
    return this.plugins.get(name);
  }

  /**
   * Check if a plugin is registered.
   */
  has(name: string): boolean {
    return this.plugins.has(name);
  }

  /**
   * List all registered plugin names.
   */
  list(): ReadonlyArray<string> {
    return Array.from(this.plugins.keys());
  }

  /**
   * Get all tools from all registered plugins.
   */
  getAllTools(): ReadonlyArray<ToolSpec> {
    const tools: ToolSpec[] = [];
    for (const plugin of this.plugins.values()) {
      if (plugin.tools) {
        tools.push(...plugin.tools);
      }
    }
    return tools;
  }

  /**
   * Check if a command is registered.
   */
  hasCommand(name: string): boolean {
    return this.commands.has(name);
  }

  /**
   * Execute a registered command.
   * Throws if command not found or context not initialized.
   */
  async executeCommand(name: string, args: string): Promise<string> {
    if (!this.context) {
      throw new Error("PluginManager not initialized — call init() first");
    }
    const entry = this.commands.get(name);
    if (!entry) {
      const available = Array.from(this.commands.keys())
        .map((c) => `/${c}`)
        .join(", ");
      throw new Error(
        `Unknown command "/${name}". Available: ${available || "none"}`,
      );
    }
    return entry.handler.execute(args, this.context);
  }

  /**
   * List all available commands with descriptions.
   */
  listCommands(): ReadonlyArray<{ name: string; description: string; plugin: string }> {
    const result: Array<{ name: string; description: string; plugin: string }> = [];
    for (const [name, entry] of this.commands.entries()) {
      result.push({
        name,
        description: entry.handler.description,
        plugin: entry.plugin,
      });
    }
    return result;
  }

  /**
   * Deactivate all plugins and clear state.
   */
  destroy(): void {
    for (const plugin of this.plugins.values()) {
      if (plugin.deactivate) {
        plugin.deactivate();
      }
    }
    this.plugins.clear();
    this.commands.clear();
    this.context = null;
  }
}
