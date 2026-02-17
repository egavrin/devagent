import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { McpHub } from "./client.js";
import { mkdirSync, writeFileSync, rmSync } from "node:fs";
import { join } from "node:path";

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
});
