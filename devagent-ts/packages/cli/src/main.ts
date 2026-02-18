/**
 * CLI main entry point — parses arguments, wires up engine, runs queries.
 * Integrates: plugins, skills, MCP, context management.
 */

import { createInterface } from "node:readline";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  EventBus,
  ApprovalGate,
  PluginManager,
  SkillRegistry,
  ContextManager,
  loadConfig,
  resolveProviderCredentials,
  findProjectRoot,
  loadModelRegistry,
  lookupModelCapabilities,
} from "@devagent/core";
import type { DevAgentConfig, ApprovalPolicy } from "@devagent/core";
import { ApprovalMode } from "@devagent/core";
import { createDefaultRegistry, validateOllamaModel } from "@devagent/providers";
import { createDefaultToolRegistry, McpHub } from "@devagent/tools";
import { TaskLoop, createBuiltinPlugins, createPlanTool } from "@devagent/engine";
import type { TaskMode } from "@devagent/engine";
import { runDesktopBridge } from "./desktop-bridge.js";
import { assembleSystemPrompt } from "./prompts/index.js";
import {
  Spinner,
  dim, red, cyan, green, yellow, bold,
  formatToolStart,
  formatToolEnd,
  formatPlan,
  formatSummary,
  formatError,
} from "./format.js";

// ─── Argument Parsing ────────────────────────────────────────

type Verbosity = "quiet" | "normal" | "verbose";

interface CliArgs {
  query: string | null;
  interactive: boolean;
  desktop: boolean;
  mode: TaskMode;
  provider: string | null;
  model: string | null;
  maxIterations: number | null;
  reasoning: "low" | "medium" | "high" | null;
  verbosity: Verbosity;
  authCommand: string | null;
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
    verbosity: "normal",
    authCommand: null,
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i]!;

    if (arg === "auth") {
      result.authCommand = args[i + 1] ?? "login";
      i++;
      return result; // auth is handled before anything else
    } else if (arg === "chat") {
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
    } else if (arg === "-q" || arg === "--quiet") {
      result.verbosity = "quiet";
    } else if (arg === "-v" || arg === "--verbose") {
      result.verbosity = "verbose";
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

Auth:
  devagent auth login             Store API key for a provider
  devagent auth status            Show configured credentials
  devagent auth logout            Remove stored credentials

Options:
  --provider <name>     LLM provider (anthropic, openai, ollama, chatgpt, github-copilot)
  --model <id>          Model ID
  --max-iterations <n>  Max tool-call iterations (default: 30)
  --reasoning <level>   Reasoning effort: low, medium, high
  --suggest             Suggest mode (show diffs, ask before writing)
  --auto-edit           Auto-edit mode (auto-approve file writes)
  --full-auto           Full-auto mode (auto-approve everything)
  --desktop             Desktop bridge mode (JSON-lines protocol over stdio)
  -v, --verbose         Verbose output (show full tool params and results)
  -q, --quiet           Quiet output (errors only)
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

Installation:
  bun run build && bun run install-cli    Install as 'devagent' command
`.trim());
}

// ─── Main ──────────────────────────────────────────────────

export async function main(): Promise<void> {
  const cliArgs = parseArgs(process.argv);

  // Auth commands — handle before config loading (doesn't need provider setup)
  if (cliArgs.authCommand) {
    const { runAuthCommand } = await import("./auth.js");
    await runAuthCommand(cliArgs.authCommand);
    return;
  }

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

  // Resolve OAuth credentials (refresh expired tokens) — must come before provider setup
  try {
    config = await resolveProviderCredentials(config);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    process.stderr.write(formatError(`OAuth credential error: ${msg}`) + "\n");
    process.stderr.write(dim('Run "devagent auth login" to re-authenticate.') + "\n");
    process.exit(1);
  }

  // Apply CLI overrides that loadConfig doesn't handle
  if (cliArgs.maxIterations !== null) {
    config = {
      ...config,
      budget: { ...config.budget, maxIterations: cliArgs.maxIterations },
    };
  }

  // Load model registry (models/*.toml files with per-model capabilities)
  // Search: devagent repo models/ dir, project models/ dir, ~/.config/devagent/models/
  const cliDir = dirname(fileURLToPath(import.meta.url));
  const devagentModelsDir = join(cliDir, "..", "..", "..", "..", "models");
  loadModelRegistry(projectRoot, [devagentModelsDir]);

  // Set up providers
  const providerRegistry = createDefaultRegistry();
  const baseProviderConfig = config.providers[config.provider] ?? {
    model: config.model,
    apiKey: process.env["DEVAGENT_API_KEY"],
  };

  // Auto-resolve capabilities from model registry if not explicitly configured
  const registryCaps = lookupModelCapabilities(config.model);
  const providerConfig = {
    ...baseProviderConfig,
    ...(cliArgs.reasoning ? { reasoningEffort: cliArgs.reasoning } : {}),
    ...(!baseProviderConfig.capabilities && registryCaps
      ? { capabilities: registryCaps }
      : {}),
  };

  // Providers that don't require an API key (local endpoints or OAuth-based)
  const noKeyProviders = new Set(["ollama", "chatgpt", "github-copilot"]);
  if (!providerConfig.apiKey && !providerConfig.oauthToken && !noKeyProviders.has(config.provider)) {
    process.stderr.write(
      formatError(`No API key configured for provider "${config.provider}".`) + "\n",
    );
    process.stderr.write(
      dim('Run "devagent auth login" to store a key, or set DEVAGENT_API_KEY') + "\n",
    );
    process.exit(1);
  }

  const provider = providerRegistry.get(config.provider, providerConfig);

  // Pre-flight: validate Ollama model availability before session setup
  if (config.provider === "ollama") {
    try {
      const ollamaBaseUrl = providerConfig.baseUrl ?? "http://localhost:11434/v1";
      await validateOllamaModel(config.model, ollamaBaseUrl);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      process.stderr.write(formatError(msg) + "\n");
      process.exit(1);
    }
  }

  // Set up tools, bus, approval
  const toolRegistry = createDefaultToolRegistry();
  const bus = new EventBus();
  const gate = new ApprovalGate(config.approval, bus);

  // Register workflow tools
  toolRegistry.register(createPlanTool(bus));

  // ─── Skills ────────────────────────────────────────────────
  const skills = new SkillRegistry();
  skills.discover(projectRoot);
  if (skills.size > 0 && cliArgs.verbosity !== "quiet") {
    process.stderr.write(dim(`[skills] Discovered ${skills.size} skill(s)`) + "\n");
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
  if (mcpServers.length > 0 && cliArgs.verbosity !== "quiet") {
    process.stderr.write(dim(`[mcp] ${mcpServers.length} server(s) connected`) + "\n");
  }

  // ─── Context Management ────────────────────────────────────
  const contextManager = new ContextManager(config.context);

  // Set up event handlers for CLI output
  setupEventHandlers(bus, config, cliArgs.verbosity);

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
        cliArgs.verbosity,
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
        cliArgs.verbosity,
      );
    }
  } finally {
    // Cleanup
    pluginManager.destroy();
    mcpHub.dispose();
  }
}

// ─── Event Handlers ──────────────────────────────────────────

/** Shared spinner instance — started during LLM thinking, stopped on tool/text events. */
const spinner = new Spinner();

/** Mutable iteration counter — updated by tool:before events, reset per query turn. */
let currentIteration = 0;

/** Tracks whether any tool was called this turn — for visual separator before final response. */
let hadToolCalls = false;

/**
 * Buffer for streamed assistant text. Text is buffered during each LLM iteration.
 * If tool calls follow (tool:before fires), the buffer is discarded (it was thinking text).
 * If no tool calls follow (final response), the buffer is flushed to stdout.
 */
let textBuffer = "";

/** Whether we're currently buffering text (between first partial text and tool:before/end). */
let isBufferingText = false;

/** Reset per query turn so formatToolStart shows correct [n/max] counter. */
export function resetOutputIteration(): void {
  currentIteration = 0;
  hadToolCalls = false;
  textBuffer = "";
  isBufferingText = false;
}

function setupEventHandlers(
  bus: EventBus,
  config: DevAgentConfig,
  verbosity: Verbosity,
): void {
  const maxIter = config.budget.maxIterations;

  // ─── Tool events ──────────────────────────────────────────

  bus.on("tool:before", (event) => {
    if (event.name.startsWith("audit:")) return;

    // Stop spinner — a tool call arrived
    spinner.stop();

    // Discard any buffered thinking text that preceded these tool calls
    textBuffer = "";
    isBufferingText = false;

    currentIteration++;
    hadToolCalls = true;

    if (verbosity === "quiet") return;

    if (verbosity === "verbose") {
      // Verbose: show full params as JSON
      const line = formatToolStart(event.name, event.params, currentIteration, maxIter);
      process.stderr.write(line + "\n");
      process.stderr.write(dim(`  params: ${JSON.stringify(event.params, null, 2)}`) + "\n");
    } else {
      // Normal: concise summary
      process.stderr.write(
        formatToolStart(event.name, event.params, currentIteration, maxIter) + "\n",
      );
    }
  });

  bus.on("tool:after", (event) => {
    if (event.name.startsWith("audit:")) return;
    if (verbosity === "quiet" && event.result.success) return;

    const line = formatToolEnd(
      event.name,
      event.result.success,
      event.durationMs,
      event.result.error ?? undefined,
    );
    process.stderr.write(line + "\n");

    if (verbosity === "verbose" && event.result.output) {
      // Show truncated output in verbose mode
      const output = event.result.output.length > 500
        ? event.result.output.substring(0, 500) + "…"
        : event.result.output;
      process.stderr.write(dim(`  output: ${output}`) + "\n");
    }

    // Restart spinner after tool completes (waiting for next LLM response)
    if (verbosity !== "quiet") {
      spinner.start("Thinking…");
    }
  });

  // ─── Plan updates ──────────────────────────────────────────

  bus.on("plan:updated", (event) => {
    if (verbosity === "quiet") return;

    spinner.stop();
    process.stderr.write("\n" + dim("── Plan ──") + "\n");
    process.stderr.write(formatPlan(event.steps) + "\n\n");
  });

  // ─── Message events (spinner management) ───────────────────

  bus.on("message:user", () => {
    // Start spinner when user query is sent (waiting for LLM)
    if (verbosity !== "quiet") {
      spinner.start("Thinking…");
    }
  });

  bus.on("message:assistant", (event) => {
    if (event.partial) {
      // Stop spinner on first text chunk
      spinner.stop();
      // Buffer the text — it might be thinking text before tool calls,
      // or it might be the final response. We'll know when tool:before
      // fires (discard buffer) or the loop ends (flush buffer to stdout).
      textBuffer += event.content;
      isBufferingText = true;
    }
  });

  // ─── Errors ────────────────────────────────────────────────

  bus.on("error", (event) => {
    spinner.stop();
    process.stderr.write(formatError(event.message) + "\n");
  });

  // ─── Approval prompts ─────────────────────────────────────

  bus.on("approval:request", (event) => {
    spinner.stop();
    const rl = createInterface({
      input: process.stdin,
      output: process.stdout,
    });
    rl.question(
      yellow(`[approval] ${event.toolName}: ${event.details}\nApprove? (y/n): `),
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
  verbosity: Verbosity,
): Promise<void> {
  const systemPrompt = assembleSystemPrompt({ mode, repoRoot, skills });

  resetOutputIteration();
  const startTime = Date.now();

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

  // Stop spinner in case LLM finished without emitting text
  spinner.stop();

  // Flush buffered final response to stdout
  if (textBuffer.trim()) {
    if (hadToolCalls) process.stderr.write("\n");
    process.stdout.write(textBuffer + "\n");
  }

  if (verbosity !== "quiet") {
    const elapsed = Date.now() - startTime;
    process.stderr.write(formatSummary(result.iterations, elapsed) + "\n");
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
  verbosity: Verbosity,
): Promise<void> {
  process.stderr.write(bold("DevAgent") + " interactive mode\n");
  process.stderr.write(
    dim(`Mode: ${mode} | Provider: ${config.provider} | Model: ${config.model}`) + "\n",
  );

  // Show available plugins and skills
  const pluginNames = pluginManager.list();
  if (pluginNames.length > 0) {
    process.stderr.write(dim(`Plugins: ${pluginNames.join(", ")}`) + "\n");
  }
  if (skills.size > 0) {
    process.stderr.write(
      dim(`Skills: ${skills.list().map((s) => s.name).join(", ")}`) + "\n",
    );
  }

  process.stderr.write(
    dim('Type your query, "/commands" for commands, or "exit" to quit.') + "\n\n",
  );

  const rl = createInterface({
    input: process.stdin,
    output: process.stdout,
    prompt: cyan("> "),
  });

  const systemPrompt = assembleSystemPrompt({ mode, repoRoot, skills });

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

  rl.prompt();

  for await (const line of rl) {
    const input = line.trim();
    if (!input) {
      rl.prompt();
      continue;
    }

    if (input === "exit" || input === "quit" || input === "/exit") {
      process.stderr.write(dim("Goodbye!") + "\n");
      break;
    }

    // ─── Built-in CLI Commands ─────────────────────────────
    if (input === "/plan") {
      mode = "plan";
      loop.setMode("plan");
      process.stderr.write(yellow("Switched to plan mode (read-only).") + "\n");
      rl.prompt();
      continue;
    }

    if (input === "/act") {
      mode = "act";
      loop.setMode("act");
      process.stderr.write(green("Switched to act mode.") + "\n");
      rl.prompt();
      continue;
    }

    if (input === "/clear") {
      process.stderr.write(dim("Conversation cleared. Starting fresh.") + "\n");
      rl.prompt();
      continue;
    }

    if (input === "/commands") {
      const cmds = pluginManager.listCommands();
      if (cmds.length === 0) {
        process.stderr.write(dim("No plugin commands available.") + "\n");
      } else {
        process.stderr.write(bold("Available commands:") + "\n");
        for (const cmd of cmds) {
          process.stderr.write(`  ${cyan("/" + cmd.name)} ${dim("—")} ${cmd.description} ${dim(`(${cmd.plugin})`)}` + "\n");
        }
      }
      rl.prompt();
      continue;
    }

    if (input === "/skills") {
      const skillList = skills.list();
      if (skillList.length === 0) {
        process.stderr.write(dim("No skills discovered. Add .md files to .devagent/skills/") + "\n");
      } else {
        process.stderr.write(bold("Available skills:") + "\n");
        for (const skill of skillList) {
          process.stderr.write(`  ${cyan(skill.name)} ${dim("—")} ${skill.description} ${dim(`(${skill.source})`)}` + "\n");
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
          process.stderr.write(result + "\n");
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          process.stderr.write(formatError(`Command error: ${msg}`) + "\n");
        }
        rl.prompt();
        continue;
      }

      // Unknown slash command
      process.stderr.write(
        yellow(`Unknown command: /${cmdName}. Use /commands to see available commands.`) + "\n",
      );
      rl.prompt();
      continue;
    }

    // ─── LLM Query ────────────────────────────────────────
    try {
      // Reset iteration budget per turn, but keep message history
      loop.resetIterations();
      resetOutputIteration();
      const turnStart = Date.now();
      await loop.run(input);

      // Stop spinner in case LLM finished without emitting text
      spinner.stop();

      // Flush buffered final response to stdout
      if (textBuffer.trim()) {
        if (hadToolCalls) process.stderr.write("\n");
        process.stdout.write(textBuffer + "\n");
      }

      if (verbosity !== "quiet") {
        const elapsed = Date.now() - turnStart;
        const iterations = loop.getIterations();
        process.stderr.write(formatSummary(iterations, elapsed) + "\n");
      }
    } catch (err) {
      spinner.stop();
      const msg = err instanceof Error ? err.message : String(err);
      process.stderr.write(formatError(msg) + "\n");
    }

    rl.prompt();
  }

  rl.close();
}
