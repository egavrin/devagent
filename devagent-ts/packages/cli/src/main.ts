/**
 * CLI main entry point — parses arguments, wires up engine, runs queries.
 * Integrates: plugins, skills, MCP, context management.
 */

import { createInterface } from "node:readline";
import {
  EventBus,
  ApprovalGate,
  PluginManager,
  SkillRegistry,
  ContextManager,
  loadConfig,
  findProjectRoot,
} from "@devagent/core";
import type { DevAgentConfig, ApprovalPolicy } from "@devagent/core";
import { ApprovalMode } from "@devagent/core";
import { createDefaultRegistry } from "@devagent/providers";
import { createDefaultToolRegistry, McpHub } from "@devagent/tools";
import { TaskLoop, createBuiltinPlugins } from "@devagent/engine";
import type { TaskMode } from "@devagent/engine";
import { runDesktopBridge } from "./desktop-bridge.js";

// ─── Argument Parsing ────────────────────────────────────────

interface CliArgs {
  query: string | null;
  interactive: boolean;
  desktop: boolean;
  mode: TaskMode;
  provider: string | null;
  model: string | null;
  maxIterations: number | null;
  reasoning: "low" | "medium" | "high" | null;
}

function parseArgs(argv: string[]): CliArgs {
  const args = argv.slice(2); // Skip bun and script path
  const result: CliArgs = {
    query: null,
    interactive: false,
    desktop: false,
    mode: "act",
    provider: null,
    model: null,
    maxIterations: null,
    reasoning: null,
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i]!;

    if (arg === "chat") {
      result.interactive = true;
    } else if (arg === "--desktop") {
      result.desktop = true;
    } else if (arg === "--plan") {
      result.mode = "plan";
    } else if (arg === "--suggest") {
      // handled in config override
    } else if (arg === "--auto-edit") {
      // handled in config override
    } else if (arg === "--full-auto") {
      // handled in config override
    } else if (arg === "--provider" && i + 1 < args.length) {
      result.provider = args[++i]!;
    } else if (arg === "--model" && i + 1 < args.length) {
      result.model = args[++i]!;
    } else if (arg === "--max-iterations" && i + 1 < args.length) {
      const val = parseInt(args[++i]!, 10);
      result.maxIterations = isNaN(val) ? null : val;
    } else if (arg === "--reasoning" && i + 1 < args.length) {
      const val = args[++i]! as "low" | "medium" | "high";
      if (["low", "medium", "high"].includes(val)) {
        result.reasoning = val;
      }
    } else if (arg === "--help" || arg === "-h") {
      printHelp();
      process.exit(0);
    } else if (!arg.startsWith("-")) {
      result.query = arg;
    }
  }

  // Default to interactive if no query
  if (!result.query) {
    result.interactive = true;
  }

  return result;
}

function getApprovalMode(argv: string[]): ApprovalMode | null {
  if (argv.includes("--suggest")) return ApprovalMode.SUGGEST;
  if (argv.includes("--auto-edit")) return ApprovalMode.AUTO_EDIT;
  if (argv.includes("--full-auto")) return ApprovalMode.FULL_AUTO;
  return null;
}

function printHelp(): void {
  console.log(`
devagent — AI-powered development agent

Usage:
  devagent "<query>"              Natural language query
  devagent chat                   Interactive chat mode
  devagent --plan "<query>"       Plan mode (read-only analysis)

Options:
  --provider <name>     LLM provider (anthropic, openai)
  --model <id>          Model ID
  --max-iterations <n>  Max tool-call iterations (default: 30)
  --reasoning <level>   Reasoning effort: low, medium, high
  --suggest             Suggest mode (show diffs, ask before writing)
  --auto-edit           Auto-edit mode (auto-approve file writes)
  --full-auto           Full-auto mode (auto-approve everything)
  --desktop             Desktop bridge mode (JSON-lines protocol over stdio)
  -h, --help            Show this help

Interactive Commands:
  /plan                 Switch to plan mode (read-only)
  /act                  Switch to act mode
  /clear                Clear conversation history
  /skills               List available skills
  /commands             List available plugin commands
  /<command> [args]      Run a plugin command
  exit                  Quit

Environment:
  DEVAGENT_PROVIDER     Default provider
  DEVAGENT_MODEL        Default model
  DEVAGENT_API_KEY      API key for the default provider
`.trim());
}

// ─── System Prompt ──────────────────────────────────────────

function getSystemPrompt(
  mode: TaskMode,
  repoRoot: string,
  skills: SkillRegistry,
): string {
  const modeLabel = mode === "plan" ? "PLAN (read-only)" : "ACT";

  let skillsSection = "";
  const skillList = skills.list();
  if (skillList.length > 0) {
    const skillNames = skillList.map((s) => `- ${s.name}: ${s.description}`).join("\n");
    skillsSection = `\n\nAvailable skills:\n${skillNames}\nYou can reference these skills when the user asks about related topics.`;
  }

  return `You are DevAgent, an AI-powered development assistant.

Mode: ${modeLabel}
Working directory: ${repoRoot}

You have access to tools for reading files, writing files, searching code, running commands, and git operations.
${mode === "plan" ? "In plan mode, you can only use read-only tools (read_file, find_files, search_files, git_status, git_diff)." : ""}
When the user asks you to perform a task:
1. Understand the request
2. Use tools to explore the codebase and gather information
3. Make changes or provide analysis as requested
4. Report what you did

Be concise and direct. Fail fast — report errors immediately rather than guessing.${skillsSection}`;
}

// ─── Main ──────────────────────────────────────────────────

export async function main(): Promise<void> {
  const cliArgs = parseArgs(process.argv);

  // Desktop bridge mode — JSON-lines protocol for Tauri IPC
  if (cliArgs.desktop) {
    await runDesktopBridge({
      provider: cliArgs.provider ?? undefined,
      model: cliArgs.model ?? undefined,
      mode: cliArgs.mode,
    });
    return;
  }

  const projectRoot = findProjectRoot() ?? process.cwd();

  // Load config with CLI overrides
  const approvalMode = getApprovalMode(process.argv);
  const configOverrides: Partial<DevAgentConfig> = {
    ...(cliArgs.provider ? { provider: cliArgs.provider } : {}),
    ...(cliArgs.model ? { model: cliArgs.model } : {}),
    ...(approvalMode ? { approval: { mode: approvalMode } as ApprovalPolicy } : {}),
  };

  let config = loadConfig(projectRoot, configOverrides);

  // Apply CLI overrides that loadConfig doesn't handle
  if (cliArgs.maxIterations !== null) {
    config = {
      ...config,
      budget: { ...config.budget, maxIterations: cliArgs.maxIterations },
    };
  }

  // Set up providers
  const providerRegistry = createDefaultRegistry();
  const providerConfig = {
    ...(config.providers[config.provider] ?? {
      model: config.model,
      apiKey: process.env["DEVAGENT_API_KEY"],
    }),
    ...(cliArgs.reasoning ? { reasoningEffort: cliArgs.reasoning } : {}),
  };

  if (!providerConfig.apiKey) {
    console.error(
      `Error: No API key configured for provider "${config.provider}".`,
    );
    console.error(
      "Set DEVAGENT_API_KEY or configure in .devagent.toml",
    );
    process.exit(1);
  }

  const provider = providerRegistry.get(config.provider, providerConfig);

  // Set up tools, bus, approval
  const toolRegistry = createDefaultToolRegistry();
  const bus = new EventBus();
  const gate = new ApprovalGate(config.approval, bus);

  // ─── Skills ────────────────────────────────────────────────
  const skills = new SkillRegistry();
  skills.discover(projectRoot);
  if (skills.size > 0) {
    process.stderr.write(
      `\x1b[90m[skills] Discovered ${skills.size} skill(s)\x1b[0m\n`,
    );
  }

  // ─── Plugins ───────────────────────────────────────────────
  const pluginManager = new PluginManager();
  pluginManager.init({ bus, config, repoRoot: projectRoot });

  // Register built-in plugins
  const builtinPlugins = createBuiltinPlugins();
  for (const plugin of builtinPlugins) {
    pluginManager.register(plugin);

    // Register plugin tools in the tool registry
    if (plugin.tools) {
      for (const tool of plugin.tools) {
        toolRegistry.register(tool);
      }
    }
  }

  // ─── MCP ───────────────────────────────────────────────────
  const mcpHub = new McpHub({ repoRoot: projectRoot, watchConfig: true });
  await mcpHub.init();

  // Register MCP tools
  const mcpTools = mcpHub.getToolSpecs();
  for (const tool of mcpTools) {
    toolRegistry.register(tool);
  }

  const mcpServers = mcpHub.getServers();
  if (mcpServers.length > 0) {
    process.stderr.write(
      `\x1b[90m[mcp] ${mcpServers.length} server(s) connected\x1b[0m\n`,
    );
  }

  // ─── Context Management ────────────────────────────────────
  const contextManager = new ContextManager(config.context);

  // Set up event handlers for CLI output
  setupEventHandlers(bus, config);

  try {
    if (cliArgs.interactive) {
      await runInteractive(
        provider,
        toolRegistry,
        bus,
        gate,
        config,
        projectRoot,
        cliArgs.mode,
        pluginManager,
        skills,
        contextManager,
      );
    } else if (cliArgs.query) {
      await runSingleQuery(
        cliArgs.query,
        provider,
        toolRegistry,
        bus,
        gate,
        config,
        projectRoot,
        cliArgs.mode,
        skills,
      );
    }
  } finally {
    // Cleanup
    pluginManager.destroy();
    mcpHub.dispose();
  }
}

// ─── Event Handlers ──────────────────────────────────────────

function setupEventHandlers(bus: EventBus, _config: DevAgentConfig): void {
  bus.on("tool:before", (event) => {
    if (!event.name.startsWith("audit:")) {
      process.stderr.write(`\x1b[90m[tool] ${event.name}\x1b[0m\n`);
    }
  });

  bus.on("tool:after", (event) => {
    if (event.result.error) {
      process.stderr.write(
        `\x1b[31m[error] ${event.name}: ${event.result.error}\x1b[0m\n`,
      );
    }
  });

  bus.on("error", (event) => {
    process.stderr.write(`\x1b[31m[error] ${event.message}\x1b[0m\n`);
  });

  // Approval prompts
  bus.on("approval:request", (event) => {
    const rl = createInterface({
      input: process.stdin,
      output: process.stdout,
    });
    rl.question(
      `\x1b[33m[approval] ${event.toolName}: ${event.details}\nApprove? (y/n): \x1b[0m`,
      (answer) => {
        rl.close();
        bus.emit("approval:response", {
          id: event.id,
          approved: answer.toLowerCase().startsWith("y"),
        });
      },
    );
  });
}

// ─── Single Query ────────────────────────────────────────────

async function runSingleQuery(
  query: string,
  provider: import("@devagent/core").LLMProvider,
  toolRegistry: import("@devagent/tools").ToolRegistry,
  bus: EventBus,
  gate: ApprovalGate,
  config: DevAgentConfig,
  repoRoot: string,
  mode: TaskMode,
  skills: SkillRegistry,
): Promise<void> {
  const systemPrompt = getSystemPrompt(mode, repoRoot, skills);

  // Stream assistant output to stdout
  let isFirstChunk = true;
  bus.on("message:assistant", (event) => {
    if (event.partial) {
      process.stdout.write(event.content);
      isFirstChunk = false;
    }
  });

  const loop = new TaskLoop({
    provider,
    tools: toolRegistry,
    bus,
    approvalGate: gate,
    config,
    systemPrompt,
    repoRoot,
    mode,
  });

  const result = await loop.run(query);
  if (!isFirstChunk) process.stdout.write("\n");

  if (config.budget.enableCostTracking) {
    process.stderr.write(
      `\x1b[90m[${result.iterations} iterations]\x1b[0m\n`,
    );
  }
}

// ─── Interactive Mode ────────────────────────────────────────

async function runInteractive(
  provider: import("@devagent/core").LLMProvider,
  toolRegistry: import("@devagent/tools").ToolRegistry,
  bus: EventBus,
  gate: ApprovalGate,
  config: DevAgentConfig,
  repoRoot: string,
  mode: TaskMode,
  pluginManager: PluginManager,
  skills: SkillRegistry,
  contextManager: ContextManager,
): Promise<void> {
  console.log("DevAgent interactive mode");
  console.log(`Mode: ${mode} | Provider: ${config.provider} | Model: ${config.model}`);

  // Show available plugins and skills
  const pluginNames = pluginManager.list();
  if (pluginNames.length > 0) {
    console.log(`Plugins: ${pluginNames.join(", ")}`);
  }
  if (skills.size > 0) {
    console.log(`Skills: ${skills.list().map((s) => s.name).join(", ")}`);
  }

  console.log('Type your query, "/commands" for commands, or "exit" to quit.\n');

  const rl = createInterface({
    input: process.stdin,
    output: process.stdout,
    prompt: "\x1b[36m> \x1b[0m",
  });

  const systemPrompt = getSystemPrompt(mode, repoRoot, skills);

  // Create a single TaskLoop that persists across turns (multi-turn conversation)
  const loop = new TaskLoop({
    provider,
    tools: toolRegistry,
    bus,
    approvalGate: gate,
    config,
    systemPrompt,
    repoRoot,
    mode,
  });

  // Stream assistant output to stdout
  bus.on("message:assistant", (event) => {
    if (event.partial) {
      process.stdout.write(event.content);
    }
  });

  rl.prompt();

  for await (const line of rl) {
    const input = line.trim();
    if (!input) {
      rl.prompt();
      continue;
    }

    if (input === "exit" || input === "quit" || input === "/exit") {
      console.log("Goodbye!");
      break;
    }

    // ─── Built-in CLI Commands ─────────────────────────────
    if (input === "/plan") {
      mode = "plan";
      loop.setMode("plan");
      console.log("Switched to plan mode (read-only).");
      rl.prompt();
      continue;
    }

    if (input === "/act") {
      mode = "act";
      loop.setMode("act");
      console.log("Switched to act mode.");
      rl.prompt();
      continue;
    }

    if (input === "/clear") {
      console.log("Conversation cleared. Starting fresh.");
      rl.prompt();
      continue;
    }

    if (input === "/commands") {
      const cmds = pluginManager.listCommands();
      if (cmds.length === 0) {
        console.log("No plugin commands available.");
      } else {
        console.log("Available commands:");
        for (const cmd of cmds) {
          console.log(`  /${cmd.name} — ${cmd.description} (${cmd.plugin})`);
        }
      }
      rl.prompt();
      continue;
    }

    if (input === "/skills") {
      const skillList = skills.list();
      if (skillList.length === 0) {
        console.log("No skills discovered. Add .md files to .devagent/skills/");
      } else {
        console.log("Available skills:");
        for (const skill of skillList) {
          console.log(`  ${skill.name} — ${skill.description} (${skill.source})`);
        }
      }
      rl.prompt();
      continue;
    }

    // ─── Plugin Commands ──────────────────────────────────
    if (input.startsWith("/")) {
      const spaceIdx = input.indexOf(" ");
      const cmdName = spaceIdx > 0 ? input.substring(1, spaceIdx) : input.substring(1);
      const cmdArgs = spaceIdx > 0 ? input.substring(spaceIdx + 1) : "";

      if (pluginManager.hasCommand(cmdName)) {
        try {
          const result = await pluginManager.executeCommand(cmdName, cmdArgs);
          console.log(result);
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          console.error(`\x1b[31mCommand error: ${msg}\x1b[0m`);
        }
        rl.prompt();
        continue;
      }

      // Unknown slash command
      console.log(`Unknown command: /${cmdName}. Use /commands to see available commands.`);
      rl.prompt();
      continue;
    }

    // ─── LLM Query ────────────────────────────────────────
    try {
      // Reset iteration budget per turn, but keep message history
      loop.resetIterations();
      await loop.run(input);
      process.stdout.write("\n\n");
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error(`\x1b[31mError: ${msg}\x1b[0m`);
    }

    rl.prompt();
  }

  rl.close();
}
