/**
 * MCP (Model Context Protocol) client — manages MCP server connections.
 * Servers are configured in .devagent/mcp.json and expose tools that
 * can be registered in the ToolRegistry.
 *
 * Follows Cline's McpHub pattern: auto-restart on config change,
 * server lifecycle management.
 *
 * ArkTS-compatible: no `any`, explicit types.
 */

import { existsSync, readFileSync, watchFile, unwatchFile } from "node:fs";
import { join } from "node:path";
import { spawn, type ChildProcess } from "node:child_process";
import type { ToolSpec, ToolResult } from "@devagent/core";

// ─── MCP Types ───────────────────────────────────────────────

export interface McpServerConfig {
  readonly command: string;
  readonly args?: ReadonlyArray<string>;
  readonly env?: Record<string, string>;
  readonly enabled?: boolean;
}

export interface McpConfig {
  readonly mcpServers: Record<string, McpServerConfig>;
}

export interface McpToolDefinition {
  readonly name: string;
  readonly description: string;
  readonly inputSchema: Record<string, unknown>;
}

export interface McpServer {
  readonly name: string;
  readonly config: McpServerConfig;
  readonly status: "running" | "stopped" | "error";
  readonly tools: ReadonlyArray<McpToolDefinition>;
  readonly error?: string;
}

export interface McpHubOptions {
  readonly repoRoot: string;
  readonly configPath?: string;
  readonly watchConfig?: boolean;
}

// ─── MCP Hub ─────────────────────────────────────────────────

/**
 * MCP Hub — manages multiple MCP server connections.
 * Discovers servers from .devagent/mcp.json, starts them,
 * and exposes their tools for registration in ToolRegistry.
 */
export class McpHub {
  private readonly repoRoot: string;
  private readonly configPath: string;
  private readonly watchEnabled: boolean;
  private servers: Map<string, McpServerState> = new Map();
  private disposed = false;

  constructor(options: McpHubOptions) {
    this.repoRoot = options.repoRoot;
    this.configPath =
      options.configPath ?? join(options.repoRoot, ".devagent", "mcp.json");
    this.watchEnabled = options.watchConfig ?? false;
  }

  /**
   * Initialize the hub: load config, start servers, optionally watch config.
   */
  async init(): Promise<void> {
    await this.loadAndStartServers();

    if (this.watchEnabled && existsSync(this.configPath)) {
      watchFile(this.configPath, { interval: 2000 }, () => {
        if (!this.disposed) {
          this.loadAndStartServers().catch((err) => {
            console.error(`[McpHub] Config reload error: ${err}`);
          });
        }
      });
    }
  }

  /**
   * Get the status of all servers.
   */
  getServers(): ReadonlyArray<McpServer> {
    const result: McpServer[] = [];
    for (const [name, state] of this.servers.entries()) {
      result.push({
        name,
        config: state.config,
        status: state.status,
        tools: state.tools,
        error: state.error,
      });
    }
    return result;
  }

  /**
   * Convert all MCP tools to DevAgent ToolSpecs for registration.
   */
  getToolSpecs(): ReadonlyArray<ToolSpec> {
    const specs: ToolSpec[] = [];

    for (const [serverName, state] of this.servers.entries()) {
      if (state.status !== "running") continue;

      for (const tool of state.tools) {
        specs.push({
          name: `mcp_${serverName}_${tool.name}`,
          description: `[MCP:${serverName}] ${tool.description}`,
          category: "external",
          paramSchema: {
            type: "object",
            properties: tool.inputSchema["properties"] as Record<string, unknown> | undefined,
            required: tool.inputSchema["required"] as ReadonlyArray<string> | undefined,
          },
          resultSchema: { type: "object" },
          handler: async (params) => this.callTool(serverName, tool.name, params),
        });
      }
    }

    return specs;
  }

  /**
   * Call a tool on an MCP server.
   */
  private async callTool(
    serverName: string,
    toolName: string,
    params: Record<string, unknown>,
  ): Promise<ToolResult> {
    const state = this.servers.get(serverName);
    if (!state || state.status !== "running" || !state.process) {
      return {
        success: false,
        output: "",
        error: `MCP server "${serverName}" is not running`,
        artifacts: [],
      };
    }

    try {
      // Send JSON-RPC request via stdio
      const request = {
        jsonrpc: "2.0",
        id: Date.now(),
        method: "tools/call",
        params: {
          name: toolName,
          arguments: params,
        },
      };

      const response = await this.sendRequest(state, request);

      const responseError = response["error"] as Record<string, unknown> | undefined;
      if (responseError) {
        return {
          success: false,
          output: "",
          error: `MCP error: ${(responseError["message"] as string) ?? JSON.stringify(responseError)}`,
          artifacts: [],
        };
      }

      const responseResult = response["result"] as Record<string, unknown> | undefined;
      const content = responseResult ? responseResult["content"] : undefined;
      if (Array.isArray(content) && content.length > 0) {
        const textParts = content
          .filter((c: Record<string, unknown>) => c["type"] === "text")
          .map((c: Record<string, unknown>) => c["text"] as string);
        return {
          success: true,
          output: textParts.join("\n"),
          error: null,
          artifacts: [],
        };
      }

      return {
        success: true,
        output: JSON.stringify(responseResult ?? {}),
        error: null,
        artifacts: [],
      };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      return {
        success: false,
        output: "",
        error: `MCP call failed: ${msg}`,
        artifacts: [],
      };
    }
  }

  /**
   * Send a JSON-RPC request to an MCP server and await the response.
   */
  private sendRequest(
    state: McpServerState,
    request: Record<string, unknown>,
  ): Promise<Record<string, unknown>> {
    return new Promise((resolve, reject) => {
      if (!state.process?.stdin || !state.process?.stdout) {
        reject(new Error("Server process not available"));
        return;
      }

      const requestId = request["id"];
      let buffer = "";

      const onData = (data: Buffer) => {
        buffer += data.toString();
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const response = JSON.parse(line) as Record<string, unknown>;
            if (response["id"] === requestId) {
              state.process?.stdout?.removeListener("data", onData);
              resolve(response);
            }
          } catch {
            // Not a complete JSON line, continue buffering
          }
        }
      };

      state.process.stdout.on("data", onData);

      // Timeout after 30 seconds
      setTimeout(() => {
        state.process?.stdout?.removeListener("data", onData);
        reject(new Error("MCP request timed out (30s)"));
      }, 30000);

      // Send request
      const payload = JSON.stringify(request) + "\n";
      state.process.stdin.write(payload);
    });
  }

  /**
   * Load config and start/restart servers.
   */
  private async loadAndStartServers(): Promise<void> {
    const config = this.loadConfig();
    if (!config) return;

    // Stop servers that are no longer in config
    for (const name of this.servers.keys()) {
      if (!config.mcpServers[name]) {
        this.stopServer(name);
      }
    }

    // Start or restart servers
    for (const [name, serverConfig] of Object.entries(config.mcpServers)) {
      if (serverConfig.enabled === false) {
        this.stopServer(name);
        continue;
      }

      // Check if config changed
      const existing = this.servers.get(name);
      if (existing && JSON.stringify(existing.config) === JSON.stringify(serverConfig)) {
        continue; // No change
      }

      // Stop existing and start fresh
      this.stopServer(name);
      await this.startServer(name, serverConfig);
    }
  }

  /**
   * Load MCP config from disk.
   */
  private loadConfig(): McpConfig | null {
    if (!existsSync(this.configPath)) {
      return null;
    }

    try {
      const content = readFileSync(this.configPath, "utf-8");
      return JSON.parse(content) as McpConfig;
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error(`[McpHub] Failed to load config: ${msg}`);
      return null;
    }
  }

  /**
   * Start an MCP server process.
   */
  private async startServer(name: string, config: McpServerConfig): Promise<void> {
    const state: McpServerState = {
      config,
      status: "stopped",
      tools: [],
      process: null,
    };

    try {
      const proc = spawn(config.command, [...(config.args ?? [])], {
        cwd: this.repoRoot,
        env: { ...process.env, ...(config.env ?? {}) },
        stdio: ["pipe", "pipe", "pipe"],
      });

      state.process = proc;

      proc.on("error", (err) => {
        state.status = "error";
        state.error = err.message;
      });

      proc.on("exit", (code) => {
        if (!this.disposed) {
          state.status = "stopped";
          state.error = code !== 0 ? `Exited with code ${code}` : undefined;
        }
      });

      // Initialize MCP protocol
      const initRequest = {
        jsonrpc: "2.0",
        id: 1,
        method: "initialize",
        params: {
          protocolVersion: "2024-11-05",
          capabilities: {},
          clientInfo: { name: "devagent", version: "2.0.0" },
        },
      };

      try {
        const initResponse = await this.sendRequest(state, initRequest);
        if (initResponse["error"]) {
          state.status = "error";
          state.error = `Init failed: ${JSON.stringify(initResponse["error"])}`;
          this.servers.set(name, state);
          return;
        }

        // Send initialized notification
        const notification = JSON.stringify({
          jsonrpc: "2.0",
          method: "notifications/initialized",
        }) + "\n";
        proc.stdin?.write(notification);

        // List tools
        const toolsRequest = {
          jsonrpc: "2.0",
          id: 2,
          method: "tools/list",
          params: {},
        };

        const toolsResponse = await this.sendRequest(state, toolsRequest);
        const toolsList = toolsResponse["result"] as Record<string, unknown> | undefined;
        if (toolsList && Array.isArray(toolsList["tools"])) {
          state.tools = (toolsList["tools"] as Array<Record<string, unknown>>).map((t) => ({
            name: t["name"] as string,
            description: (t["description"] as string) ?? "",
            inputSchema: (t["inputSchema"] as Record<string, unknown>) ?? { type: "object" },
          }));
        }

        state.status = "running";
      } catch (err) {
        state.status = "error";
        state.error = err instanceof Error ? err.message : String(err);
        proc.kill();
      }
    } catch (err) {
      state.status = "error";
      state.error = err instanceof Error ? err.message : String(err);
    }

    this.servers.set(name, state);
  }

  /**
   * Stop a server process.
   */
  private stopServer(name: string): void {
    const state = this.servers.get(name);
    if (!state) return;

    if (state.process) {
      state.process.kill();
      state.process = null;
    }

    state.status = "stopped";
    this.servers.delete(name);
  }

  /**
   * Shut down the hub: stop all servers, stop watching config.
   */
  dispose(): void {
    this.disposed = true;
    unwatchFile(this.configPath);

    for (const name of this.servers.keys()) {
      this.stopServer(name);
    }
  }
}

// ─── Internal State ──────────────────────────────────────────

interface McpServerState {
  readonly config: McpServerConfig;
  status: "running" | "stopped" | "error";
  tools: McpToolDefinition[];
  process: ChildProcess | null;
  error?: string;
}
