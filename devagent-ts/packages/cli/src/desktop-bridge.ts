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
 *   {"type":"approval_response","id":"...","approved":true}
 *   {"type":"abort"}
 *   {"type":"exit"}
 *   {"type":"list_skills"}
 *   {"type":"load_skill","name":"..."}
 *   {"type":"list_mcp_servers"}
 *   {"type":"restart_mcp_server","name":"..."}
 *   {"type":"toggle_mcp_server","name":"...","enabled":true}
 *   {"type":"search_memories","query":"...","category":"pattern"}
 *   {"type":"get_memory_summary"}
 *   {"type":"delete_memory","id":"..."}
 *   {"type":"list_commands"}
 *   {"type":"execute_command","command":"...","args":"..."}
 *   {"type":"get_working_dir"}
 *   {"type":"set_working_dir","dir":"..."}
 *   {"type":"get_config"}
 *
 * Outgoing (stdout):
 *   {"type":"text","content":"partial text..."}
 *   {"type":"tool_start","name":"read_file","callId":"...","params":{}}
 *   {"type":"tool_end","name":"read_file","callId":"...","success":true,"output":"...","error":null,"durationMs":42}
 *   {"type":"approval_request","id":"...","toolName":"write_file","details":"..."}
 *   {"type":"done","iterations":3}
 *   {"type":"error","message":"...","fatal":false}
 *   {"type":"ready"}
 *   {"type":"skills_list","skills":[...]}
 *   {"type":"skill_loaded","name":"...","instructions":"..."}
 *   {"type":"mcp_servers","servers":[...]}
 *   {"type":"memories","entries":[...]}
 *   {"type":"memory_summary","summary":{...}}
 *   {"type":"memory_deleted","id":"..."}
 *   {"type":"commands_list","commands":[...]}
 *   {"type":"command_result","command":"...","output":"..."}
 *   {"type":"working_dir","dir":"..."}
 *   {"type":"config","config":{...}}
 *   {"type":"file_diff","filePath":"...","diff":"...","toolCallId":"..."}
 *   {"type":"cost_update","inputTokens":0,"outputTokens":0,"totalCost":0}
 */

import { createInterface } from "node:readline";
import { execSync } from "node:child_process";
import {
  EventBus,
  ApprovalGate,
  PluginManager,
  SkillRegistry,
  ContextManager,
  MemoryStore,
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

interface IncomingListSkills {
  type: "list_skills";
}

interface IncomingLoadSkill {
  type: "load_skill";
  name: string;
}

interface IncomingListMcpServers {
  type: "list_mcp_servers";
}

interface IncomingRestartMcpServer {
  type: "restart_mcp_server";
  name: string;
}

interface IncomingToggleMcpServer {
  type: "toggle_mcp_server";
  name: string;
  enabled: boolean;
}

interface IncomingSearchMemories {
  type: "search_memories";
  query?: string;
  category?: string;
}

interface IncomingGetMemorySummary {
  type: "get_memory_summary";
}

interface IncomingDeleteMemory {
  type: "delete_memory";
  id: string;
}

interface IncomingListCommands {
  type: "list_commands";
}

interface IncomingExecuteCommand {
  type: "execute_command";
  command: string;
  args: string;
}

interface IncomingGetWorkingDir {
  type: "get_working_dir";
}

interface IncomingSetWorkingDir {
  type: "set_working_dir";
  dir: string;
}

interface IncomingGetConfig {
  type: "get_config";
}

type IncomingMessage =
  | IncomingQuery
  | IncomingSetMode
  | IncomingSetProvider
  | IncomingSetApproval
  | IncomingApprovalResponse
  | IncomingAbort
  | IncomingExit
  | IncomingListSkills
  | IncomingLoadSkill
  | IncomingListMcpServers
  | IncomingRestartMcpServer
  | IncomingToggleMcpServer
  | IncomingSearchMemories
  | IncomingGetMemorySummary
  | IncomingDeleteMemory
  | IncomingListCommands
  | IncomingExecuteCommand
  | IncomingGetWorkingDir
  | IncomingSetWorkingDir
  | IncomingGetConfig;

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

  // Set up providers — defer provider creation until needed.
  // The desktop app may not have an API key configured on startup;
  // the user sets it via the Settings panel which sends "set_provider".
  const providerRegistry = createDefaultRegistry();
  let provider: LLMProvider | null = null;
  try {
    provider = createProvider(providerRegistry, config);
  } catch {
    // No API key yet — that's OK, user will configure via Settings
  }

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

  // Memory store
  const memoryStore = new MemoryStore();

  // Wire up event bus → JSON-lines output
  setupBusToJsonLines(bus, projectRoot);

  // Create initial task loop — deferred if no provider yet
  let loop: TaskLoop | null = null;
  if (provider) {
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
  }

  // Helper: ensure loop exists (throws user-friendly error if not configured)
  function ensureLoop(): TaskLoop {
    if (!loop) {
      throw new Error(
        "No LLM provider configured. Open Settings and set your API key first.",
      );
    }
    return loop;
  }

  // Signal ready + initial state
  send({ type: "ready" });
  send({ type: "working_dir", dir: projectRoot });
  if (!provider) {
    send({
      type: "error",
      message: "No API key configured. Open Settings to set your provider and API key.",
      fatal: false,
    });
  }

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

        let activeLoop: TaskLoop;
        try {
          activeLoop = ensureLoop();
        } catch (err) {
          const errMsg = err instanceof Error ? err.message : String(err);
          send({ type: "error", message: errMsg, fatal: false });
          return;
        }

        running = true;

        if (msg.mode) {
          currentMode = msg.mode;
          activeLoop.setMode(currentMode);
        }

        // Run the query asynchronously so stdin remains responsive
        // for approval_response and abort messages during execution.
        void (async () => {
          try {
            activeLoop.resetIterations();
            const result = await activeLoop.run(msg.content);
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
        if (loop) loop.setMode(currentMode);
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
        if (loop) loop.abort();
        break;
      }

      case "list_skills": {
        const skillList = skills.list();
        send({
          type: "skills_list",
          skills: skillList.map((s) => ({
            name: s.name,
            description: s.description,
            source: s.source,
          })),
        });
        break;
      }

      case "load_skill": {
        try {
          const skill = skills.load(msg.name);
          send({
            type: "skill_loaded",
            name: skill.name,
            description: skill.description,
            source: skill.source,
            instructions: skill.instructions,
          });
        } catch (err) {
          const errMsg = err instanceof Error ? err.message : String(err);
          send({ type: "error", message: `Failed to load skill: ${errMsg}`, fatal: false });
        }
        break;
      }

      case "list_mcp_servers": {
        const servers = mcpHub.getServers();
        send({
          type: "mcp_servers",
          servers: servers.map((s) => ({
            name: s.name,
            status: s.status,
            toolCount: s.tools.length,
            tools: s.tools.map((t) => ({
              name: t.name,
              description: t.description,
            })),
            error: s.error,
          })),
        });
        break;
      }

      case "restart_mcp_server": {
        // Dispose and re-init the hub (individual server restart not yet supported)
        try {
          mcpHub.dispose();
          void (async () => {
            await mcpHub.init();
            const servers = mcpHub.getServers();
            send({
              type: "mcp_servers",
              servers: servers.map((s) => ({
                name: s.name,
                status: s.status,
                toolCount: s.tools.length,
                tools: s.tools.map((t) => ({
                  name: t.name,
                  description: t.description,
                })),
                error: s.error,
              })),
            });
          })();
        } catch (err) {
          const errMsg = err instanceof Error ? err.message : String(err);
          send({ type: "error", message: `Failed to restart MCP server: ${errMsg}`, fatal: false });
        }
        break;
      }

      case "toggle_mcp_server": {
        // Toggle not yet supported at hub level — send current state
        send({
          type: "error",
          message: `MCP server toggle not yet implemented. Edit mcp.json to enable/disable servers.`,
          fatal: false,
        });
        break;
      }

      case "search_memories": {
        try {
          const searchOpts: Record<string, unknown> = {};
          if (msg.query) searchOpts["query"] = msg.query;
          if (msg.category) searchOpts["category"] = msg.category;
          const entries = memoryStore.search(searchOpts);
          send({
            type: "memories",
            entries: entries.map((m) => ({
              id: m.id,
              category: m.category,
              key: m.key,
              content: m.content,
              relevance: m.relevance,
              tags: m.tags,
              updatedAt: m.updatedAt,
              accessCount: m.accessCount,
            })),
          });
        } catch (err) {
          const errMsg = err instanceof Error ? err.message : String(err);
          send({ type: "error", message: `Memory search failed: ${errMsg}`, fatal: false });
        }
        break;
      }

      case "get_memory_summary": {
        try {
          const summary = memoryStore.summary();
          send({ type: "memory_summary", summary });
        } catch (err) {
          const errMsg = err instanceof Error ? err.message : String(err);
          send({ type: "error", message: `Memory summary failed: ${errMsg}`, fatal: false });
        }
        break;
      }

      case "delete_memory": {
        try {
          const deleted = memoryStore.delete(msg.id);
          if (deleted) {
            send({ type: "memory_deleted", id: msg.id });
          } else {
            send({ type: "error", message: `Memory not found: ${msg.id}`, fatal: false });
          }
        } catch (err) {
          const errMsg = err instanceof Error ? err.message : String(err);
          send({ type: "error", message: `Memory delete failed: ${errMsg}`, fatal: false });
        }
        break;
      }

      case "list_commands": {
        const commands = pluginManager.listCommands();
        send({ type: "commands_list", commands });
        break;
      }

      case "execute_command": {
        void (async () => {
          try {
            const output = await pluginManager.executeCommand(msg.command, msg.args);
            send({ type: "command_result", command: msg.command, output });
          } catch (err) {
            const errMsg = err instanceof Error ? err.message : String(err);
            send({ type: "error", message: `Command failed: ${errMsg}`, fatal: false });
          }
        })();
        break;
      }

      case "get_working_dir": {
        send({ type: "working_dir", dir: projectRoot });
        break;
      }

      case "set_working_dir": {
        // Working directory change requires process restart.
        // The Tauri frontend handles this by killing and respawning
        // the CLI child process with the new directory.
        send({
          type: "error",
          message: "Working directory change requires engine restart. The desktop app will respawn the CLI process.",
          fatal: false,
        });
        break;
      }

      case "get_config": {
        send({
          type: "config",
          config: {
            provider: config.provider,
            model: config.model,
            approval: config.approval,
            context: config.context,
            budget: config.budget,
          },
        });
        break;
      }

      case "exit": {
        pluginManager.destroy();
        mcpHub.dispose();
        memoryStore.close();
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
    memoryStore.close();
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

function setupBusToJsonLines(bus: EventBus, repoRoot: string): void {
  // Track tool params by callId so we can access them in tool:after
  const pendingToolParams = new Map<string, Record<string, unknown>>();

  // Stream text chunks
  bus.on("message:assistant", (event) => {
    if (event.partial) {
      send({ type: "text", content: event.content });
    }
  });

  // Tool execution events
  bus.on("tool:before", (event) => {
    pendingToolParams.set(event.callId, event.params);
    send({
      type: "tool_start",
      name: event.name,
      callId: event.callId,
      params: event.params,
    });
  });

  bus.on("tool:after", (event) => {
    const params = pendingToolParams.get(event.callId);
    pendingToolParams.delete(event.callId);

    send({
      type: "tool_end",
      name: event.name,
      callId: event.callId,
      success: event.result.success,
      output: event.result.output.substring(0, 2000), // Truncate for transport
      error: event.result.error,
      durationMs: event.durationMs,
    });

    // Emit file diffs for mutating file tools
    if (
      event.result.success &&
      (event.name === "write_file" || event.name === "replace_in_file") &&
      params
    ) {
      const filePath = (params["path"] ?? params["filePath"] ?? "") as string;
      if (filePath) {
        try {
          const diff = execSync(`git diff -- "${filePath}"`, {
            cwd: repoRoot,
            encoding: "utf-8",
            timeout: 5000,
          });
          if (diff.trim()) {
            send({
              type: "file_diff",
              filePath,
              diff,
              toolCallId: event.callId,
            });
          }
        } catch {
          // Not a git repo or git diff failed — skip silently
        }
      }
    }
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
