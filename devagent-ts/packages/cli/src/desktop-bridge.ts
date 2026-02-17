/**
 * Desktop bridge — JSON-lines protocol for Tauri IPC.
 *
 * The desktop app spawns the CLI with `--desktop` flag.
 * Communication is via stdin/stdout using JSON lines:
 *
 * Incoming (stdin):
 *   {"type":"query","content":"...","mode":"act"}
 *   {"type":"set_mode","mode":"plan"}
 *   {"type":"set_provider","provider":"anthropic","model":"claude-sonnet-4-20250514","apiKey":"sk-..."}
 *   {"type":"set_approval","mode":"suggest"}
 *   {"type":"abort"}
 *   {"type":"exit"}
 *
 * Outgoing (stdout):
 *   {"type":"text","content":"partial text..."}
 *   {"type":"tool_start","name":"read_file","callId":"...","params":{}}
 *   {"type":"tool_end","name":"read_file","callId":"...","success":true,"output":"...","error":null,"durationMs":42}
 *   {"type":"approval_request","id":"...","toolName":"write_file","details":"..."}
 *   {"type":"done","iterations":3}
 *   {"type":"error","message":"...","fatal":false}
 *   {"type":"ready"}
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
import type { ProviderRegistry } from "@devagent/providers";
import { createDefaultToolRegistry, McpHub } from "@devagent/tools";
import type { ToolRegistry } from "@devagent/tools";
import { TaskLoop, createBuiltinPlugins } from "@devagent/engine";
import type { TaskMode } from "@devagent/engine";
import type { LLMProvider } from "@devagent/core";

// ─── JSON-lines Protocol Types ──────────────────────────────

interface IncomingQuery {
  type: "query";
  content: string;
  mode?: TaskMode;
}

interface IncomingSetMode {
  type: "set_mode";
  mode: TaskMode;
}

interface IncomingSetProvider {
  type: "set_provider";
  provider: string;
  model: string;
  apiKey?: string;
}

interface IncomingSetApproval {
  type: "set_approval";
  mode: string;
}

interface IncomingApprovalResponse {
  type: "approval_response";
  id: string;
  approved: boolean;
}

interface IncomingAbort {
  type: "abort";
}

interface IncomingExit {
  type: "exit";
}

type IncomingMessage =
  | IncomingQuery
  | IncomingSetMode
  | IncomingSetProvider
  | IncomingSetApproval
  | IncomingApprovalResponse
  | IncomingAbort
  | IncomingExit;

// ─── Outgoing helpers ──────────────────────────────────────

function send(data: Record<string, unknown>): void {
  process.stdout.write(JSON.stringify(data) + "\n");
}

// ─── System Prompt ─────────────────────────────────────────

function getSystemPrompt(mode: TaskMode, repoRoot: string, skills: SkillRegistry): string {
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

// ─── Desktop Bridge ─────────────────────────────────────────

export async function runDesktopBridge(initialArgs: {
  provider?: string;
  model?: string;
  mode?: TaskMode;
}): Promise<void> {
  const projectRoot = findProjectRoot() ?? process.cwd();

  // Initial config
  const configOverrides: Partial<DevAgentConfig> = {
    ...(initialArgs.provider ? { provider: initialArgs.provider } : {}),
    ...(initialArgs.model ? { model: initialArgs.model } : {}),
  };

  let config = loadConfig(projectRoot, configOverrides);
  let currentMode: TaskMode = initialArgs.mode ?? "act";

  // Set up providers
  const providerRegistry = createDefaultRegistry();
  let provider = createProvider(providerRegistry, config);

  // Set up tools, bus, approval
  const toolRegistry = createDefaultToolRegistry();
  const bus = new EventBus();
  let gate = new ApprovalGate(config.approval, bus);

  // Skills
  const skills = new SkillRegistry();
  skills.discover(projectRoot);

  // Plugins
  const pluginManager = new PluginManager();
  pluginManager.init({ bus, config, repoRoot: projectRoot });
  const builtinPlugins = createBuiltinPlugins();
  for (const plugin of builtinPlugins) {
    pluginManager.register(plugin);
    if (plugin.tools) {
      for (const tool of plugin.tools) {
        toolRegistry.register(tool);
      }
    }
  }

  // MCP
  const mcpHub = new McpHub({ repoRoot: projectRoot, watchConfig: true });
  await mcpHub.init();
  const mcpTools = mcpHub.getToolSpecs();
  for (const tool of mcpTools) {
    toolRegistry.register(tool);
  }

  // Context management
  const _contextManager = new ContextManager(config.context);

  // Wire up event bus → JSON-lines output
  setupBusToJsonLines(bus);

  // Create initial task loop
  let loop = createLoop(
    provider,
    toolRegistry,
    bus,
    gate,
    config,
    projectRoot,
    currentMode,
    skills,
  );

  // Signal ready
  send({ type: "ready" });

  // Read stdin line by line using event-based handler.
  // IMPORTANT: We use rl.on("line") instead of `for await (const line of rl)`
  // because the for-await loop blocks on `await loop.run()` which prevents
  // processing approval_response and abort messages while the task loop is
  // waiting for user approval. Event-based handling processes all messages
  // concurrently — approval responses arrive even during loop.run().
  const rl = createInterface({
    input: process.stdin,
    output: process.stdout,
    terminal: false,
  });

  let running = false; // Guard against concurrent queries

  rl.on("line", (line: string) => {
    const trimmed = line.trim();
    if (!trimmed) return;

    let msg: IncomingMessage;
    try {
      msg = JSON.parse(trimmed) as IncomingMessage;
    } catch {
      send({ type: "error", message: `Invalid JSON: ${trimmed}`, fatal: false });
      return;
    }

    switch (msg.type) {
      case "query": {
        if (running) {
          send({ type: "error", message: "A query is already running", fatal: false });
          return;
        }
        running = true;

        if (msg.mode) {
          currentMode = msg.mode;
          loop.setMode(currentMode);
        }

        // Run the query asynchronously so stdin remains responsive
        // for approval_response and abort messages during execution.
        void (async () => {
          try {
            loop.resetIterations();
            const result = await loop.run(msg.content);
            send({
              type: "done",
              iterations: result.iterations,
              cost: result.cost,
              aborted: result.aborted,
            });
          } catch (err) {
            const errMsg = err instanceof Error ? err.message : String(err);
            send({ type: "error", message: errMsg, fatal: false });
          } finally {
            running = false;
          }
        })();
        break;
      }

      case "set_mode": {
        currentMode = msg.mode;
        loop.setMode(currentMode);
        send({ type: "mode_changed", mode: currentMode });
        break;
      }

      case "set_provider": {
        try {
          // Rebuild config with new provider/model
          const newOverrides: Partial<DevAgentConfig> = {
            provider: msg.provider,
            model: msg.model,
          };
          config = loadConfig(projectRoot, newOverrides);

          // Override API key if provided
          if (msg.apiKey) {
            const providerConfig = {
              ...config.providers[msg.provider],
              model: msg.model,
              apiKey: msg.apiKey,
            };
            provider = providerRegistry.get(msg.provider, providerConfig);
          } else {
            provider = createProvider(providerRegistry, config);
          }

          // Recreate task loop with new provider (fresh conversation)
          gate = new ApprovalGate(config.approval, bus);
          loop = createLoop(
            provider,
            toolRegistry,
            bus,
            gate,
            config,
            projectRoot,
            currentMode,
            skills,
          );

          send({ type: "provider_changed", provider: msg.provider, model: msg.model });
        } catch (err) {
          const errMsg = err instanceof Error ? err.message : String(err);
          send({ type: "error", message: `Failed to set provider: ${errMsg}`, fatal: false });
        }
        break;
      }

      case "set_approval": {
        const approvalMap: Record<string, ApprovalMode> = {
          suggest: ApprovalMode.SUGGEST,
          "auto-edit": ApprovalMode.AUTO_EDIT,
          "full-auto": ApprovalMode.FULL_AUTO,
        };
        const newMode = approvalMap[msg.mode];
        if (newMode) {
          const newApproval = { ...config.approval, mode: newMode } as ApprovalPolicy;
          config = { ...config, approval: newApproval } as DevAgentConfig;
          gate = new ApprovalGate(config.approval, bus);
          send({ type: "approval_changed", mode: msg.mode });
        }
        break;
      }

      case "approval_response": {
        bus.emit("approval:response", {
          id: msg.id,
          approved: msg.approved,
        });
        break;
      }

      case "abort": {
        loop.abort();
        break;
      }

      case "exit": {
        pluginManager.destroy();
        mcpHub.dispose();
        process.exit(0);
        break;
      }

      default: {
        send({ type: "error", message: `Unknown message type: ${(msg as IncomingMessage).type}`, fatal: false });
      }
    }
  });

  rl.on("close", () => {
    // Stdin closed — clean exit
    pluginManager.destroy();
    mcpHub.dispose();
  });

  // Keep process alive (stdin event loop handles messages)
  await new Promise<void>(() => {
    // Never resolves — process exits via "exit" message or stdin close
  });
}

// ─── Helpers ───────────────────────────────────────────────

function createProvider(
  registry: ProviderRegistry,
  config: DevAgentConfig,
): LLMProvider {
  const providerConfig = config.providers[config.provider] ?? {
    model: config.model,
    apiKey: process.env["DEVAGENT_API_KEY"],
  };

  if (!providerConfig.apiKey) {
    throw new Error(
      `No API key for provider "${config.provider}". Set DEVAGENT_API_KEY or configure in .devagent.toml`,
    );
  }

  return registry.get(config.provider, providerConfig);
}

function createLoop(
  provider: LLMProvider,
  tools: ToolRegistry,
  bus: EventBus,
  gate: ApprovalGate,
  config: DevAgentConfig,
  repoRoot: string,
  mode: TaskMode,
  skills: SkillRegistry,
): TaskLoop {
  const systemPrompt = getSystemPrompt(mode, repoRoot, skills);
  return new TaskLoop({
    provider,
    tools,
    bus,
    approvalGate: gate,
    config,
    systemPrompt,
    repoRoot,
    mode,
  });
}

function setupBusToJsonLines(bus: EventBus): void {
  // Stream text chunks
  bus.on("message:assistant", (event) => {
    if (event.partial) {
      send({ type: "text", content: event.content });
    }
  });

  // Tool execution events
  bus.on("tool:before", (event) => {
    send({
      type: "tool_start",
      name: event.name,
      callId: event.callId,
      params: event.params,
    });
  });

  bus.on("tool:after", (event) => {
    send({
      type: "tool_end",
      name: event.name,
      callId: event.callId,
      success: event.result.success,
      output: event.result.output.substring(0, 2000), // Truncate for transport
      error: event.result.error,
      durationMs: event.durationMs,
    });
  });

  // Approval requests
  bus.on("approval:request", (event) => {
    send({
      type: "approval_request",
      id: event.id,
      toolName: event.toolName,
      details: event.details,
    });
  });

  // Errors
  bus.on("error", (event) => {
    send({
      type: "error",
      message: event.message,
      fatal: event.fatal,
    });
  });

  // Cost updates
  bus.on("cost:update", (event) => {
    send({
      type: "cost_update",
      inputTokens: event.inputTokens,
      outputTokens: event.outputTokens,
      totalCost: event.totalCost,
    });
  });
}
