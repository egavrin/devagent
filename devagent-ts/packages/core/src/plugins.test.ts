import { describe, it, expect, beforeEach } from "vitest";
import { PluginManager } from "./plugins.js";
import type { Plugin, PluginContext, CommandHandler } from "./plugins.js";
import { EventBus } from "./events.js";
import type { DevAgentConfig, ToolSpec } from "./types.js";
import { ApprovalMode } from "./types.js";

function makeConfig(): DevAgentConfig {
  return {
    provider: "test",
    model: "test-model",
    providers: {},
    approval: {
      mode: ApprovalMode.SUGGEST,
      autoApprovePlan: false,
      autoApproveCode: false,
      autoApproveShell: false,
      auditLog: false,
      toolOverrides: {},
      pathRules: [],
    },
    budget: {
      maxIterations: 10,
      maxContextTokens: 4096,
      responseHeadroom: 1024,
      costWarningThreshold: 1.0,
      enableCostTracking: false,
    },
    context: {
      pruningStrategy: "sliding_window",
      triggerRatio: 0.8,
      keepRecentMessages: 10,
    },
    arkts: { enabled: false, strictMode: false, targetVersion: "5.0" },
  };
}

function makeContext(): PluginContext {
  return {
    bus: new EventBus(),
    config: makeConfig(),
    repoRoot: "/tmp/test-repo",
  };
}

function makeTool(name: string): ToolSpec {
  return {
    name,
    description: `Test tool ${name}`,
    category: "readonly",
    paramSchema: { type: "object" },
    resultSchema: { type: "object" },
    handler: async () => ({ success: true, output: "ok", error: null, artifacts: [] }),
  };
}

function makePlugin(name: string, options?: {
  tools?: ReadonlyArray<ToolSpec>;
  commands?: Record<string, CommandHandler>;
}): Plugin {
  let activated = false;
  return {
    name,
    version: "1.0.0",
    tools: options?.tools,
    commands: options?.commands,
    activate() { activated = true; },
    deactivate() { activated = false; },
  };
}

describe("PluginManager", () => {
  let manager: PluginManager;

  beforeEach(() => {
    manager = new PluginManager();
  });

  it("throws if register called before init", () => {
    expect(() => manager.register(makePlugin("test"))).toThrow(
      "PluginManager not initialized",
    );
  });

  it("registers and activates a plugin", () => {
    manager.init(makeContext());
    const plugin = makePlugin("hello");
    manager.register(plugin);
    expect(manager.has("hello")).toBe(true);
    expect(manager.list()).toContain("hello");
  });

  it("throws on duplicate plugin name", () => {
    manager.init(makeContext());
    manager.register(makePlugin("dup"));
    expect(() => manager.register(makePlugin("dup"))).toThrow(
      'Plugin "dup" is already registered',
    );
  });

  it("unregisters and deactivates a plugin", () => {
    manager.init(makeContext());
    manager.register(makePlugin("gone"));
    expect(manager.has("gone")).toBe(true);
    manager.unregister("gone");
    expect(manager.has("gone")).toBe(false);
  });

  it("collects tools from all plugins", () => {
    manager.init(makeContext());
    manager.register(makePlugin("p1", { tools: [makeTool("t1")] }));
    manager.register(makePlugin("p2", { tools: [makeTool("t2"), makeTool("t3")] }));
    const tools = manager.getAllTools();
    expect(tools).toHaveLength(3);
    expect(tools.map((t) => t.name)).toEqual(["t1", "t2", "t3"]);
  });

  it("registers and executes commands", async () => {
    manager.init(makeContext());
    const cmd: CommandHandler = {
      description: "Say hello",
      async execute(args) { return `Hello ${args}`; },
    };
    manager.register(makePlugin("greeter", { commands: { greet: cmd } }));
    expect(manager.hasCommand("greet")).toBe(true);
    const result = await manager.executeCommand("greet", "world");
    expect(result).toBe("Hello world");
  });

  it("throws on unknown command", async () => {
    manager.init(makeContext());
    await expect(manager.executeCommand("nope", "")).rejects.toThrow(
      'Unknown command "/nope"',
    );
  });

  it("throws on duplicate command names", () => {
    manager.init(makeContext());
    const cmd: CommandHandler = {
      description: "test",
      async execute() { return "ok"; },
    };
    manager.register(makePlugin("p1", { commands: { dup: cmd } }));
    expect(() => {
      manager.register(makePlugin("p2", { commands: { dup: cmd } }));
    }).toThrow('Command "/dup" already registered by plugin "p1"');
  });

  it("lists commands with descriptions", () => {
    manager.init(makeContext());
    const cmd: CommandHandler = {
      description: "Do stuff",
      async execute() { return "done"; },
    };
    manager.register(makePlugin("test-plug", { commands: { stuff: cmd } }));
    const cmds = manager.listCommands();
    expect(cmds).toHaveLength(1);
    expect(cmds[0]).toEqual({
      name: "stuff",
      description: "Do stuff",
      plugin: "test-plug",
    });
  });

  it("removes commands when plugin is unregistered", async () => {
    manager.init(makeContext());
    const cmd: CommandHandler = {
      description: "temp",
      async execute() { return "ok"; },
    };
    manager.register(makePlugin("temp", { commands: { temp: cmd } }));
    expect(manager.hasCommand("temp")).toBe(true);
    manager.unregister("temp");
    expect(manager.hasCommand("temp")).toBe(false);
  });

  it("destroy clears all plugins and commands", () => {
    manager.init(makeContext());
    manager.register(makePlugin("a"));
    manager.register(makePlugin("b"));
    manager.destroy();
    expect(manager.list()).toHaveLength(0);
  });
});
