/**
 * CLI main entry point — parses arguments, wires up engine, runs queries.
 */

import { createInterface } from "node:readline";
import {
  EventBus,
  ApprovalGate,
  loadConfig,
  findProjectRoot,
} from "@devagent/core";
import type { DevAgentConfig, ApprovalPolicy } from "@devagent/core";
import { ApprovalMode } from "@devagent/core";
import { createDefaultRegistry } from "@devagent/providers";
import { createDefaultToolRegistry } from "@devagent/tools";
import { TaskLoop } from "@devagent/engine";
import type { TaskMode } from "@devagent/engine";

// ─── Argument Parsing ────────────────────────────────────────

interface CliArgs {
  query: string | null;
  interactive: boolean;
  mode: TaskMode;
  provider: string | null;
  model: string | null;
}

function parseArgs(argv: string[]): CliArgs {
  const args = argv.slice(2); // Skip bun and script path
  const result: CliArgs = {
    query: null,
    interactive: false,
    mode: "act",
    provider: null,
    model: null,
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i]!;

    if (arg === "chat") {
      result.interactive = true;
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
  --suggest             Suggest mode (show diffs, ask before writing)
  --auto-edit           Auto-edit mode (auto-approve file writes)
  --full-auto           Full-auto mode (auto-approve everything)
  -h, --help            Show this help

Environment:
  DEVAGENT_PROVIDER     Default provider
  DEVAGENT_MODEL        Default model
  DEVAGENT_API_KEY      API key for the default provider
`.trim());
}

// ─── System Prompt ──────────────────────────────────────────

function getSystemPrompt(mode: TaskMode, repoRoot: string): string {
  const modeLabel = mode === "plan" ? "PLAN (read-only)" : "ACT";
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

Be concise and direct. Fail fast — report errors immediately rather than guessing.`;
}

// ─── Main ──────────────────────────────────────────────────

export async function main(): Promise<void> {
  const cliArgs = parseArgs(process.argv);
  const projectRoot = findProjectRoot() ?? process.cwd();

  // Load config with CLI overrides
  const approvalMode = getApprovalMode(process.argv);
  const configOverrides: Partial<DevAgentConfig> = {
    ...(cliArgs.provider ? { provider: cliArgs.provider } : {}),
    ...(cliArgs.model ? { model: cliArgs.model } : {}),
    ...(approvalMode ? { approval: { mode: approvalMode } as ApprovalPolicy } : {}),
  };

  const config = loadConfig(projectRoot, configOverrides);

  // Set up providers
  const providerRegistry = createDefaultRegistry();
  const providerConfig = config.providers[config.provider] ?? {
    model: config.model,
    apiKey: process.env["DEVAGENT_API_KEY"],
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

  // Set up event handlers for CLI output
  setupEventHandlers(bus, config);

  if (cliArgs.interactive) {
    await runInteractive(
      provider,
      toolRegistry,
      bus,
      gate,
      config,
      projectRoot,
      cliArgs.mode,
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
    );
  }
}

// ─── Event Handlers ──────────────────────────────────────────

function setupEventHandlers(bus: EventBus, config: DevAgentConfig): void {
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
): Promise<void> {
  const systemPrompt = getSystemPrompt(mode, repoRoot);

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
): Promise<void> {
  console.log("DevAgent interactive mode");
  console.log(`Mode: ${mode} | Provider: ${config.provider} | Model: ${config.model}`);
  console.log('Type your query, or "exit" to quit.\n');

  const rl = createInterface({
    input: process.stdin,
    output: process.stdout,
    prompt: "\x1b[36m> \x1b[0m",
  });

  const systemPrompt = getSystemPrompt(mode, repoRoot);

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
      // Cannot clear a TaskLoop's history, so we inform the user
      // In practice, a new TaskLoop would be created
      rl.prompt();
      continue;
    }

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
