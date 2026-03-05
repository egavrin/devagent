import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { McpHub } from "./client.js";
import { mkdirSync, writeFileSync, rmSync } from "node:fs";
import { join } from "node:path";
import { EventEmitter } from "node:events";

const TEST_DIR = "/tmp/devagent-mcp-test";
const CONFIG_DIR = join(TEST_DIR, ".devagent");
const CONFIG_PATH = join(CONFIG_DIR, "mcp.json");

describe("McpHub", () => {
  beforeEach(() => {
    mkdirSync(CONFIG_DIR, { recursive: true });
  });

  afterEach(() => {
    rmSync(TEST_DIR, { recursive: true, force: true });
  });

  it("initializes with no config file", async () => {
    rmSync(CONFIG_PATH, { force: true });
    const hub = new McpHub({ repoRoot: TEST_DIR });
    await hub.init();
    expect(hub.getServers()).toHaveLength(0);
    expect(hub.getToolSpecs()).toHaveLength(0);
    hub.dispose();
  });

  it("loads config with disabled servers", async () => {
    writeFileSync(
      CONFIG_PATH,
      JSON.stringify({
        mcpServers: {
          test: {
            command: "echo",
            args: ["hello"],
            enabled: false,
          },
        },
      }),
      "utf-8",
    );

    const hub = new McpHub({ repoRoot: TEST_DIR });
    await hub.init();
    // Disabled server should not be started
    expect(hub.getServers()).toHaveLength(0);
    hub.dispose();
  });

  it("handles invalid config JSON gracefully", async () => {
    writeFileSync(CONFIG_PATH, "not json{{{", "utf-8");
    const hub = new McpHub({ repoRoot: TEST_DIR });
    // Should not throw
    await hub.init();
    expect(hub.getServers()).toHaveLength(0);
    hub.dispose();
  });

  it("creates tool specs from server tools", () => {
    // Test the tool spec generation logic by creating a hub
    // and checking that getToolSpecs returns empty for no running servers
    const hub = new McpHub({ repoRoot: TEST_DIR });
    expect(hub.getToolSpecs()).toHaveLength(0);
    hub.dispose();
  });

  it("disposes cleanly", async () => {
    const hub = new McpHub({ repoRoot: TEST_DIR });
    await hub.init();
    expect(() => hub.dispose()).not.toThrow();
  });

  it("handles missing config directory", async () => {
    rmSync(TEST_DIR, { recursive: true, force: true });
    const hub = new McpHub({ repoRoot: TEST_DIR });
    await hub.init();
    expect(hub.getServers()).toHaveLength(0);
    hub.dispose();
  });

  it("clears request timeout after successful response", async () => {
    vi.useFakeTimers();
    const hub = new McpHub({ repoRoot: TEST_DIR });
    const stdout = new EventEmitter();
    const stdin = { write: vi.fn() };
    const baselineTimers = vi.getTimerCount();
    const baselineListeners = stdout.listenerCount("data");

    const state = {
      process: {
        stdin,
        stdout,
      },
    };

    try {
      const sendRequest = (
        hub as unknown as {
          sendRequest: (
            stateArg: unknown,
            request: Record<string, unknown>,
          ) => Promise<Record<string, unknown>>;
        }
      ).sendRequest.bind(hub);

      const request = {
        jsonrpc: "2.0",
        id: 42,
        method: "tools/list",
        params: {},
      };

      const responsePromise = sendRequest(state, request);
      expect(stdin.write).toHaveBeenCalledOnce();

      stdout.emit(
        "data",
        Buffer.from(
          `${JSON.stringify({ jsonrpc: "2.0", id: 42, result: { tools: [] } })}\n`,
        ),
      );

      const response = await responsePromise;
      expect(response["result"]).toEqual({ tools: [] });
      expect(stdout.listenerCount("data")).toBe(baselineListeners);
      expect(vi.getTimerCount()).toBe(baselineTimers);
    } finally {
      vi.useRealTimers();
    }
  });

  it("does not discard interleaved responses for other request IDs", async () => {
    vi.useFakeTimers();
    const hub = new McpHub({ repoRoot: TEST_DIR });
    const stdout = new EventEmitter();
    const stdin = { write: vi.fn() };

    const state = {
      process: {
        stdin,
        stdout,
      },
    };

    try {
      const sendRequest = (
        hub as unknown as {
          sendRequest: (
            stateArg: unknown,
            request: Record<string, unknown>,
          ) => Promise<Record<string, unknown>>;
        }
      ).sendRequest.bind(hub);

      // Fire two concurrent requests with different IDs
      const promiseA = sendRequest(state, {
        jsonrpc: "2.0",
        id: 100,
        method: "tools/call",
        params: { name: "a" },
      });

      const promiseB = sendRequest(state, {
        jsonrpc: "2.0",
        id: 200,
        method: "tools/call",
        params: { name: "b" },
      });

      // Both listeners are registered. Emit a single chunk containing
      // both responses interleaved (response for 200 arrives first).
      const lineB = JSON.stringify({
        jsonrpc: "2.0",
        id: 200,
        result: { value: "b-result" },
      });
      const lineA = JSON.stringify({
        jsonrpc: "2.0",
        id: 100,
        result: { value: "a-result" },
      });
      stdout.emit("data", Buffer.from(`${lineB}\n${lineA}\n`));

      const [responseA, responseB] = await Promise.all([promiseA, promiseB]);

      expect(
        (responseA["result"] as Record<string, unknown>)["value"],
      ).toBe("a-result");
      expect(
        (responseB["result"] as Record<string, unknown>)["value"],
      ).toBe("b-result");
    } finally {
      vi.useRealTimers();
    }
  });

  it("includes errorGuidance in generated tool specs", () => {
    const hub = new McpHub({ repoRoot: TEST_DIR });

    // Inject a fake running server with one tool via internal state
    const servers = (hub as unknown as { servers: Map<string, unknown> }).servers;
    servers.set("test-server", {
      config: { command: "echo", args: [] },
      status: "running",
      tools: [
        {
          name: "do_thing",
          description: "Does a thing",
          inputSchema: { type: "object", properties: {}, required: [] },
        },
      ],
      process: null,
    });

    const specs = hub.getToolSpecs();
    expect(specs).toHaveLength(1);

    const g = specs[0].errorGuidance;
    expect(g).toBeDefined();
    expect(g!.common).toContain("MCP");

    // Must cover the three MCP-specific failure modes
    const patterns = g!.patterns!;
    expect(patterns.length).toBeGreaterThanOrEqual(3);
    const matches = patterns.map((p) => p.match);
    expect(matches).toContain("not running");
    expect(matches).toContain("timed out");
    expect(matches).toContain("MCP error");

    hub.dispose();
  });

  it("preserves unmatched lines across multiple data events", async () => {
    vi.useFakeTimers();
    const hub = new McpHub({ repoRoot: TEST_DIR });
    const stdout = new EventEmitter();
    const stdin = { write: vi.fn() };

    const state = {
      process: {
        stdin,
        stdout,
      },
    };

    try {
      const sendRequest = (
        hub as unknown as {
          sendRequest: (
            stateArg: unknown,
            request: Record<string, unknown>,
          ) => Promise<Record<string, unknown>>;
        }
      ).sendRequest.bind(hub);

      // Start a request that is waiting for id=300
      const promise = sendRequest(state, {
        jsonrpc: "2.0",
        id: 300,
        method: "tools/call",
        params: {},
      });

      // First data event: an unrelated response line (id=999) only
      const unrelated = JSON.stringify({
        jsonrpc: "2.0",
        id: 999,
        result: { ignored: true },
      });
      stdout.emit("data", Buffer.from(`${unrelated}\n`));

      // Second data event: the matching response for id=300
      const matching = JSON.stringify({
        jsonrpc: "2.0",
        id: 300,
        result: { value: "found" },
      });
      stdout.emit("data", Buffer.from(`${matching}\n`));

      const response = await promise;
      expect(
        (response["result"] as Record<string, unknown>)["value"],
      ).toBe("found");
    } finally {
      vi.useRealTimers();
    }
  });
});
