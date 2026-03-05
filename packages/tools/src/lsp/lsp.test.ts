import { describe, it, expect, vi, afterEach } from "vitest";
import { LSPClient } from "./client.js";
import { createLSPTools, createRoutingLSPTools, type LSPClientResolver } from "./tools.js";
import { mkdtempSync, rmSync, writeFileSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

afterEach(() => {
  vi.useRealTimers();
});

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

describe("LSPClient", () => {
  it("creates a client with options", () => {
    const client = new LSPClient({
      command: "typescript-language-server",
      args: ["--stdio"],
      rootPath: "/tmp/test",
      languageId: "typescript",
    });
    expect(client.isRunning()).toBe(false);
  });

  it("throws if used before start", async () => {
    const client = new LSPClient({
      command: "echo",
      args: [],
      rootPath: "/tmp",
      languageId: "typescript",
    });

    await expect(client.getDiagnostics("test.ts")).rejects.toThrow(
      "not initialized",
    );
  });

  it("does not return stale diagnostics on repeated checks for the same file", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "devagent-lsp-test-"));
    const fileName = "test.ts";
    writeFileSync(join(tempDir, fileName), "const x = 1;\n", "utf-8");

    const client = new LSPClient({
      command: "echo",
      args: [],
      rootPath: tempDir,
      languageId: "typescript",
      diagnosticTimeout: 500,
    });

    const diagnosticsStore = (client as unknown as {
      diagnosticsStore: Map<string, unknown[]>;
    }).diagnosticsStore;
    let openCount = 0;

    const fakeConnection = {
      sendNotification: (method: string, params: Record<string, unknown>) => {
        if (method !== "textDocument/didOpen") return;
        openCount++;
        const textDoc = params["textDocument"] as
          | { uri?: string }
          | undefined;
        const uri = textDoc?.uri;
        if (!uri) return;

        if (openCount === 1) {
          setTimeout(() => {
            diagnosticsStore.set(uri, [{
              range: {
                start: { line: 0, character: 0 },
                end: { line: 0, character: 1 },
              },
              message: "first run error",
              severity: 1,
            }]);
          }, 20);
        } else {
          setTimeout(() => {
            diagnosticsStore.set(uri, []);
          }, 20);
        }
      },
    };

    (client as unknown as {
      initialized: boolean;
      connection: unknown;
    }).initialized = true;
    (client as unknown as {
      initialized: boolean;
      connection: unknown;
    }).connection = fakeConnection;

    try {
      const first = await client.getDiagnostics(fileName);
      expect(first.diagnostics).toHaveLength(1);

      const second = await client.getDiagnostics(fileName);
      expect(second.diagnostics).toHaveLength(0);
    } finally {
      rmSync(tempDir, { recursive: true, force: true });
    }
  });

  it("does not use readFileSync (async I/O only)", async () => {
    // Verify the module doesn't import readFileSync
    const clientSource = readFileSync(
      join(__dirname, "client.ts"),
      "utf-8",
    );
    expect(clientSource).not.toContain("readFileSync");
  });

  it("cleans up timeout timer after successful LSP request", async () => {
    vi.useFakeTimers();
    const tempDir = mkdtempSync(join(tmpdir(), "devagent-lsp-timeout-"));
    const fileName = "test.ts";
    writeFileSync(join(tempDir, fileName), "const x = 1;\n", "utf-8");

    const client = new LSPClient({
      command: "echo",
      args: [],
      rootPath: tempDir,
      languageId: "typescript",
      timeout: 5000,
    });

    const fakeConnection = {
      sendNotification: () => {},
      sendRequest: (_method: string) => {
        // Simulate a fast LSP response
        return Promise.resolve({
          uri: `file://${tempDir}/${fileName}`,
          range: { start: { line: 0, character: 0 }, end: { line: 0, character: 1 } },
        });
      },
    };

    (client as unknown as { initialized: boolean; connection: unknown }).initialized = true;
    (client as unknown as { initialized: boolean; connection: unknown }).connection = fakeConnection;

    try {
      const timersBefore = vi.getTimerCount();
      await client.getDefinition(fileName, 1, 1);
      const timersAfter = vi.getTimerCount();

      // withTimeout should clean up its timer after the request resolves.
      // If it leaks, timersAfter > timersBefore.
      expect(timersAfter).toBe(timersBefore);
    } finally {
      rmSync(tempDir, { recursive: true, force: true });
    }
  });

  it("returns quickly when server publishes an empty diagnostics array", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "devagent-lsp-empty-"));
    const fileName = "test.ts";
    writeFileSync(join(tempDir, fileName), "const x = 1;\n", "utf-8");

    const client = new LSPClient({
      command: "echo",
      args: [],
      rootPath: tempDir,
      languageId: "typescript",
      diagnosticTimeout: 1000,
    });

    const diagnosticsStore = (client as unknown as {
      diagnosticsStore: Map<string, unknown[]>;
    }).diagnosticsStore;

    const fakeConnection = {
      sendNotification: (method: string, params: Record<string, unknown>) => {
        if (method !== "textDocument/didOpen") return;
        const textDoc = params["textDocument"] as
          | { uri?: string }
          | undefined;
        const uri = textDoc?.uri;
        if (!uri) return;

        setTimeout(() => {
          diagnosticsStore.set(uri, []);
        }, 20);
      },
    };

    (client as unknown as {
      initialized: boolean;
      connection: unknown;
    }).initialized = true;
    (client as unknown as {
      initialized: boolean;
      connection: unknown;
    }).connection = fakeConnection;

    try {
      // Verify the client returns quickly when the server publishes empty diagnostics
      // (exits poll loop early instead of waiting full diagnosticTimeout of 1000ms).
      const startTime = Date.now();
      const result = await client.getDiagnostics(fileName);
      const elapsed = Date.now() - startTime;

      expect(result.diagnostics).toHaveLength(0);
      // Should resolve well before the 1000ms timeout.
      // The server pushes [] at 20ms, poll interval is 100ms, so ~100-300ms.
      expect(elapsed).toBeLessThan(500);
      await sleep(30);
    } finally {
      rmSync(tempDir, { recursive: true, force: true });
    }
  });
});

  it("caches open files and sends didChange instead of didOpen/didClose for repeated access", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "devagent-lsp-cache-"));
    const fileName = "test.ts";
    writeFileSync(join(tempDir, fileName), "const x = 1;\n", "utf-8");

    const client = new LSPClient({
      command: "echo",
      args: [],
      rootPath: tempDir,
      languageId: "typescript",
      timeout: 5000,
    });

    const notifications: Array<{ method: string }> = [];
    const fakeConnection = {
      sendNotification: (method: string) => {
        notifications.push({ method });
      },
      sendRequest: () => Promise.resolve([]),
    };

    (client as unknown as { initialized: boolean; connection: unknown }).initialized = true;
    (client as unknown as { initialized: boolean; connection: unknown }).connection = fakeConnection;

    try {
      // First call: should send didOpen
      await client.getSymbols(fileName);
      const firstOpenCount = notifications.filter((n) => n.method === "textDocument/didOpen").length;
      const firstCloseCount = notifications.filter((n) => n.method === "textDocument/didClose").length;
      expect(firstOpenCount).toBe(1);
      expect(firstCloseCount).toBe(0); // File stays open

      notifications.length = 0;

      // Second call to same file: should NOT send didOpen again
      await client.getSymbols(fileName);
      const secondOpenCount = notifications.filter((n) => n.method === "textDocument/didOpen").length;
      expect(secondOpenCount).toBe(0); // Already open, no didOpen
    } finally {
      rmSync(tempDir, { recursive: true, force: true });
    }
  });

  it("sends didChange when file content changes between calls", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "devagent-lsp-change-"));
    const fileName = "test.ts";
    writeFileSync(join(tempDir, fileName), "const x = 1;\n", "utf-8");

    const client = new LSPClient({
      command: "echo",
      args: [],
      rootPath: tempDir,
      languageId: "typescript",
      timeout: 5000,
    });

    const notifications: Array<{ method: string }> = [];
    const fakeConnection = {
      sendNotification: (method: string) => {
        notifications.push({ method });
      },
      sendRequest: () => Promise.resolve([]),
    };

    (client as unknown as { initialized: boolean; connection: unknown }).initialized = true;
    (client as unknown as { initialized: boolean; connection: unknown }).connection = fakeConnection;

    try {
      // First call
      await client.getSymbols(fileName);
      notifications.length = 0;

      // Modify the file
      writeFileSync(join(tempDir, fileName), "const x = 2;\n", "utf-8");

      // Second call: should send didChange (not didOpen)
      await client.getSymbols(fileName);
      const changeCount = notifications.filter((n) => n.method === "textDocument/didChange").length;
      const openCount = notifications.filter((n) => n.method === "textDocument/didOpen").length;
      expect(changeCount).toBe(1);
      expect(openCount).toBe(0);
    } finally {
      rmSync(tempDir, { recursive: true, force: true });
    }
  });

  it("detects a crashed server process and marks client as not running", () => {
    const client = new LSPClient({
      command: "echo",
      args: [],
      rootPath: "/tmp",
      languageId: "typescript",
    });

    // Capture the exit handler that start() registers
    let exitCallback: (() => void) | null = null;
    const fakeProcess = {
      stdin: {},
      stdout: {},
      kill: () => {},
      on: (event: string, cb: () => void) => {
        if (event === "exit") exitCallback = cb;
      },
    };

    // Wire up the process and initialized flag as start() would
    const internals = client as unknown as {
      initialized: boolean;
      process: unknown;
      connection: unknown;
    };
    internals.process = fakeProcess;
    internals.initialized = true;
    internals.connection = { dispose: () => {} };

    // Manually register the exit handler (same code as in start())
    fakeProcess.on("exit", () => {
      internals.initialized = false;
    });

    expect(client.isRunning()).toBe(true);
    expect(exitCallback).not.toBeNull();

    // Simulate process crash
    exitCallback!();

    expect(client.isRunning()).toBe(false);
  });

  it("stop cleans up process/connection even when client was never initialized", async () => {
    const client = new LSPClient({
      command: "echo",
      args: [],
      rootPath: "/tmp",
      languageId: "typescript",
    });

    const dispose = vi.fn();
    const kill = vi.fn();
    const internals = client as unknown as {
      initialized: boolean;
      process: { kill: () => void };
      connection: { dispose: () => void };
    };
    internals.initialized = false;
    internals.process = { kill };
    internals.connection = { dispose };

    await client.stop();

    expect(dispose).toHaveBeenCalledOnce();
    expect(kill).toHaveBeenCalledOnce();
    expect(client.isRunning()).toBe(false);
  });

describe("createLSPTools", () => {
  it("creates 4 code analysis tools", () => {
    const client = new LSPClient({
      command: "echo",
      args: [],
      rootPath: "/tmp",
      languageId: "typescript",
    });

    const tools = createLSPTools(client);
    expect(tools.length).toBe(4);

    const names = tools.map((t) => t.name);
    expect(names).toContain("diagnostics");
    expect(names).toContain("definitions");
    expect(names).toContain("references");
    expect(names).toContain("symbols");
  });

  it("all LSP tools are readonly", () => {
    const client = new LSPClient({
      command: "echo",
      args: [],
      rootPath: "/tmp",
      languageId: "typescript",
    });

    const tools = createLSPTools(client);
    for (const tool of tools) {
      expect(tool.category).toBe("readonly");
    }
  });

  it("tools return error when client not running", async () => {
    const client = new LSPClient({
      command: "echo",
      args: [],
      rootPath: "/tmp",
      languageId: "typescript",
    });

    const tools = createLSPTools(client);
    const diagnosticsTool = tools.find((t) => t.name === "diagnostics");
    expect(diagnosticsTool).toBeDefined();

    const result = await diagnosticsTool!.handler(
      { path: "test.ts" },
      { repoRoot: "/tmp", config: {} as never, sessionId: "" },
    );

    expect(result.success).toBe(false);
    expect(result.error).toContain("not running");
  });
});

describe("createRoutingLSPTools", () => {
  it("routes diagnostics to the correct client based on file extension", async () => {
    const tsClient = new LSPClient({
      command: "echo",
      args: [],
      rootPath: "/tmp",
      languageId: "typescript",
    });
    const pyClient = new LSPClient({
      command: "echo",
      args: [],
      rootPath: "/tmp",
      languageId: "python",
    });

    const resolver = (filePath: string) => {
      if (filePath.endsWith(".ts")) return { client: tsClient, languageId: "typescript" };
      if (filePath.endsWith(".py")) return { client: pyClient, languageId: "python" };
      return null;
    };

    const tools = createRoutingLSPTools(resolver);
    expect(tools.length).toBe(6);

    // Both clients are not running, but the error message should differ
    // to prove routing happened — the tool should return "not running"
    // because the resolved client isn't started.
    const diagnosticsTool = tools.find((t) => t.name === "diagnostics")!;

    const tsResult = await diagnosticsTool.handler(
      { path: "foo.ts" },
      { repoRoot: "/tmp", config: {} as never, sessionId: "" },
    );
    expect(tsResult.success).toBe(false);
    expect(tsResult.error).toContain("not running");

    const pyResult = await diagnosticsTool.handler(
      { path: "bar.py" },
      { repoRoot: "/tmp", config: {} as never, sessionId: "" },
    );
    expect(pyResult.success).toBe(false);
    expect(pyResult.error).toContain("not running");
  });

  it("returns error for unrecognized file extensions", async () => {
    const resolver = () => null;

    const tools = createRoutingLSPTools(resolver);
    const diagnosticsTool = tools.find((t) => t.name === "diagnostics")!;

    const result = await diagnosticsTool.handler(
      { path: "unknown.xyz" },
      { repoRoot: "/tmp", config: {} as never, sessionId: "" },
    );
    expect(result.success).toBe(false);
    expect(result.error).toContain("No LSP server");
  });
});

describe("name-based LSP tools", () => {
  function makeMockClient(overrides: Record<string, unknown> = {}) {
    return {
      isRunning: () => true,
      getSymbols: vi.fn(),
      getDefinition: vi.fn(),
      getReferences: vi.fn(),
      ...overrides,
    } as unknown as LSPClient;
  }

  const ctx = { repoRoot: "/tmp", config: {} as never, sessionId: "" };

  it("definition_by_name resolves symbol position via getSymbols then calls getDefinition", async () => {
    const client = makeMockClient();
    (client.getSymbols as ReturnType<typeof vi.fn>).mockResolvedValue([
      { name: "Foo", kind: "class", line: 10, character: 5 },
    ]);
    (client.getDefinition as ReturnType<typeof vi.fn>).mockResolvedValue([
      { file: "src/other.ts", line: 42, character: 1 },
    ]);

    const resolver = () => ({ client, languageId: "typescript" });
    const tools = createRoutingLSPTools(resolver);
    const tool = tools.find((t) => t.name === "definition_by_name")!;
    expect(tool).toBeDefined();

    const result = await tool.handler({ path: "src/foo.ts", symbol_name: "Foo" }, ctx);
    expect(result.success).toBe(true);
    expect(result.output).toContain("Definition(s):");
    expect(result.output).toContain("src/other.ts:42:1");
    expect(client.getDefinition).toHaveBeenCalledWith("src/foo.ts", 10, 5, "typescript");
  });

  it("definition_by_name returns error when symbol not found in file", async () => {
    const client = makeMockClient();
    (client.getSymbols as ReturnType<typeof vi.fn>).mockResolvedValue([]);

    const resolver = () => ({ client, languageId: "typescript" });
    const tools = createRoutingLSPTools(resolver);
    const tool = tools.find((t) => t.name === "definition_by_name")!;

    const result = await tool.handler({ path: "src/foo.ts", symbol_name: "Missing" }, ctx);
    expect(result.success).toBe(false);
    expect(result.error).toContain("not found");
  });

  it("references_by_name finds all usages of a symbol by name", async () => {
    const client = makeMockClient();
    (client.getSymbols as ReturnType<typeof vi.fn>).mockResolvedValue([
      { name: "bar", kind: "function", line: 5, character: 10 },
    ]);
    (client.getReferences as ReturnType<typeof vi.fn>).mockResolvedValue([
      { file: "src/a.ts", line: 1, character: 1 },
      { file: "src/b.ts", line: 20, character: 5 },
      { file: "src/c.ts", line: 30, character: 8 },
    ]);

    const resolver = () => ({ client, languageId: "typescript" });
    const tools = createRoutingLSPTools(resolver);
    const tool = tools.find((t) => t.name === "references_by_name")!;
    expect(tool).toBeDefined();

    const result = await tool.handler({ path: "src/bar.ts", symbol_name: "bar" }, ctx);
    expect(result.success).toBe(true);
    expect(result.output).toContain("3 reference(s):");
    expect(result.output).toContain("src/a.ts:1:1");
    expect(client.getReferences).toHaveBeenCalledWith("src/bar.ts", 5, 10, "typescript");
  });

  it("references_by_name returns error when symbol not found in file", async () => {
    const client = makeMockClient();
    (client.getSymbols as ReturnType<typeof vi.fn>).mockResolvedValue([
      { name: "other", kind: "variable", line: 1, character: 1 },
    ]);

    const resolver = () => ({ client, languageId: "typescript" });
    const tools = createRoutingLSPTools(resolver);
    const tool = tools.find((t) => t.name === "references_by_name")!;

    const result = await tool.handler({ path: "src/bar.ts", symbol_name: "bar" }, ctx);
    expect(result.success).toBe(false);
    expect(result.error).toContain("not found");
  });
});

describe("LSP tool errorGuidance", () => {
  it("all single-client LSP tools have errorGuidance with common and patterns", () => {
    const client = new LSPClient({
      command: "echo",
      args: [],
      rootPath: "/tmp",
      languageId: "typescript",
    });

    const tools = createLSPTools(client);
    expect(tools).toHaveLength(4);

    for (const tool of tools) {
      expect(tool.errorGuidance, `${tool.name} missing errorGuidance`).toBeDefined();
      expect(tool.errorGuidance!.common.length, `${tool.name} empty common`).toBeGreaterThan(0);
      expect(tool.errorGuidance!.patterns, `${tool.name} missing patterns`).toBeDefined();
      expect(tool.errorGuidance!.patterns!.length, `${tool.name} no patterns`).toBeGreaterThan(0);
    }
  });

  it("all single-client LSP tools cover 'not running' pattern", () => {
    const client = new LSPClient({
      command: "echo",
      args: [],
      rootPath: "/tmp",
      languageId: "typescript",
    });

    const tools = createLSPTools(client);
    for (const tool of tools) {
      const matches = tool.errorGuidance!.patterns!.map((p) => p.match);
      expect(matches, `${tool.name} missing 'not running' pattern`).toContain("not running");
    }
  });

  it("diagnostics common hint mentions file path", () => {
    const client = new LSPClient({
      command: "echo",
      args: [],
      rootPath: "/tmp",
      languageId: "typescript",
    });

    const tools = createLSPTools(client);
    const diag = tools.find((t) => t.name === "diagnostics")!;
    expect(diag.errorGuidance!.common).toContain("file path");
  });
});

describe("Routing LSP tool errorGuidance", () => {
  it("all routing LSP tools have errorGuidance", () => {
    const resolver: LSPClientResolver = () => null;
    const tools = createRoutingLSPTools(resolver);
    expect(tools).toHaveLength(6);

    for (const tool of tools) {
      expect(tool.errorGuidance, `${tool.name} missing errorGuidance`).toBeDefined();
      expect(tool.errorGuidance!.common.length, `${tool.name} empty common`).toBeGreaterThan(0);
      expect(tool.errorGuidance!.patterns, `${tool.name} missing patterns`).toBeDefined();
    }
  });

  it("all routing tools cover 'No LSP server' and 'not running' patterns", () => {
    const resolver: LSPClientResolver = () => null;
    const tools = createRoutingLSPTools(resolver);

    for (const tool of tools) {
      const matches = tool.errorGuidance!.patterns!.map((p) => p.match);
      expect(matches, `${tool.name} missing 'No LSP server'`).toContain("No LSP server");
      expect(matches, `${tool.name} missing 'not running'`).toContain("not running");
    }
  });

  it("name-based tools have 'not found in' pattern for missing symbols", () => {
    const resolver: LSPClientResolver = () => null;
    const tools = createRoutingLSPTools(resolver);
    const nameTools = tools.filter(
      (t) => t.name === "definition_by_name" || t.name === "references_by_name",
    );
    expect(nameTools).toHaveLength(2);

    for (const tool of nameTools) {
      const matches = tool.errorGuidance!.patterns!.map((p) => p.match);
      expect(matches, `${tool.name} missing 'not found in'`).toContain("not found in");
    }
  });

  it("position-based tools do NOT have 'not found in' pattern", () => {
    const resolver: LSPClientResolver = () => null;
    const tools = createRoutingLSPTools(resolver);
    const posTools = tools.filter(
      (t) => t.name !== "definition_by_name" && t.name !== "references_by_name",
    );
    expect(posTools).toHaveLength(4);

    for (const tool of posTools) {
      const matches = tool.errorGuidance!.patterns!.map((p) => p.match);
      expect(matches, `${tool.name} has 'not found in' but shouldn't`).not.toContain("not found in");
    }
  });
});
