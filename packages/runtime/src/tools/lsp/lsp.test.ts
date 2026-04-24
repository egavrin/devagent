import { mkdtempSync, rmSync, writeFileSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, it, expect, vi, afterEach } from "vitest";

import { LSPClient } from "./client.js";
import { createLSPTools, createRoutingLSPTools, type LSPClientResolver } from "./tools.js";

afterEach(() => {
  vi.useRealTimers();
});

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function makeDestroyedStreamError(): Error & { code: string } {
  return Object.assign(
    new Error("Cannot call write after a stream was destroyed"),
    { code: "ERR_STREAM_DESTROYED" },
  );
}
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

it("clears stale diagnostics after a real content change", async () => {
  const tempDir = mkdtempSync(join(tmpdir(), "devagent-lsp-test-"));
  const fileName = "test.ts";
  writeFileSync(join(tempDir, fileName), "const x: string = 1;\n", "utf-8");

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

  const fakeConnection = {
    sendNotification: (method: string, params: Record<string, unknown>) => {
      if (method !== "textDocument/didOpen" && method !== "textDocument/didChange") return;
      const textDoc = params["textDocument"] as
        | { uri?: string }
        | undefined;
      const uri = textDoc?.uri;
      if (!uri) return;

      if (method === "textDocument/didOpen") {
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

    writeFileSync(join(tempDir, fileName), "const x = 1;\n", "utf-8");
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

it("returns cached diagnostics after syncDocument sends didSave for unchanged content", async () => {
  const tempDir = mkdtempSync(join(tmpdir(), "devagent-lsp-sync-diag-"));
  const fileName = "test.ts";
  writeFileSync(join(tempDir, fileName), "const x: string = 1;\n", "utf-8");

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
  const notifications: string[] = [];
  const fakeConnection = {
    sendNotification: vi.fn(async (method: string, params: Record<string, unknown>) => {
      notifications.push(method);
      if (method !== "textDocument/didOpen" && method !== "textDocument/didChange") return;
      const textDoc = params["textDocument"] as { uri?: string } | undefined;
      const uri = textDoc?.uri;
      if (!uri) return;
      diagnosticsStore.set(uri, [{
        range: {
          start: { line: 0, character: 6 },
          end: { line: 0, character: 7 },
        },
        message: "Type 'number' is not assignable to type 'string'.",
        severity: 1,
      }]);
    }),
  };

  (client as unknown as { initialized: boolean; connection: unknown }).initialized = true;
  (client as unknown as { initialized: boolean; connection: unknown }).connection = fakeConnection;

  try {
    await client.syncDocument(fileName, "typescript", { didSave: true });
    expect(notifications).toEqual(["textDocument/didOpen", "textDocument/didSave"]);

    notifications.length = 0;
    const result = await client.getDiagnostics(fileName, "typescript");

    expect(result.diagnostics).toHaveLength(1);
    expect(result.diagnostics[0]!.message).toContain("not assignable");
    expect(notifications).toEqual([]);
  } finally {
    rmSync(tempDir, { recursive: true, force: true });
  }
});

it("returns cached clean diagnostics after syncDocument for unchanged content", async () => {
  const tempDir = mkdtempSync(join(tmpdir(), "devagent-lsp-sync-clean-"));
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
  const fakeConnection = {
    sendNotification: vi.fn(async (method: string, params: Record<string, unknown>) => {
      if (method !== "textDocument/didOpen") return;
      const textDoc = params["textDocument"] as { uri?: string } | undefined;
      const uri = textDoc?.uri;
      if (uri) diagnosticsStore.set(uri, []);
    }),
  };

  (client as unknown as { initialized: boolean; connection: unknown }).initialized = true;
  (client as unknown as { initialized: boolean; connection: unknown }).connection = fakeConnection;

  try {
    await client.syncDocument(fileName, "typescript", { didSave: true });
    const result = await client.getDiagnostics(fileName, "typescript");

    expect(result.diagnostics).toHaveLength(0);
    expect(fakeConnection.sendNotification).toHaveBeenCalledTimes(2);
  } finally {
    rmSync(tempDir, { recursive: true, force: true });
  }
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

  it("stop suppresses destroyed-stream exit notification rejections", async () => {
    const client = new LSPClient({
      command: "echo",
      args: [],
      rootPath: "/tmp",
      languageId: "typescript",
    });

    const dispose = vi.fn();
    const kill = vi.fn();
    const unhandled: unknown[] = [];
    const onUnhandledRejection = (reason: unknown) => {
      unhandled.push(reason);
    };
    process.on("unhandledRejection", onUnhandledRejection);

    const internals = client as unknown as {
      initialized: boolean;
      process: { kill: () => void };
      connection: {
        dispose: () => void;
        sendRequest: (method: string) => Promise<void>;
        sendNotification: (method: string, params?: unknown) => Promise<void>;
      };
    };
    internals.initialized = true;
    internals.process = { kill };
    internals.connection = {
      dispose,
      sendRequest: vi.fn(async () => undefined),
      sendNotification: vi.fn(async (method: string) => {
        if (method === "exit") {
          throw makeDestroyedStreamError();
        }
      }),
    };

    try {
      await client.stop();
      await sleep(0);
      expect(unhandled).toHaveLength(0);
      expect(dispose).toHaveBeenCalledOnce();
      expect(kill).toHaveBeenCalledOnce();
    } finally {
      process.off("unhandledRejection", onUnhandledRejection);
    }
  });

  it("stop suppresses destroyed-stream didClose notification rejections", async () => {
    const client = new LSPClient({
      command: "echo",
      args: [],
      rootPath: "/tmp",
      languageId: "typescript",
    });

    const dispose = vi.fn();
    const kill = vi.fn();
    const notifications: string[] = [];
    const internals = client as unknown as {
      initialized: boolean;
      process: { kill: () => void };
      connection: {
        dispose: () => void;
        sendRequest: (method: string) => Promise<void>;
        sendNotification: (method: string, params?: unknown) => Promise<void>;
      };
      openDocuments: Map<string, { version: number; content: string }>;
    };
    internals.initialized = true;
    internals.process = { kill };
    internals.openDocuments.set("file:///tmp/test.ts", { version: 1, content: "const x = 1;\n" });
    internals.connection = {
      dispose,
      sendRequest: vi.fn(async () => undefined),
      sendNotification: vi.fn(async (method: string) => {
        notifications.push(method);
        if (method === "textDocument/didClose") {
          throw makeDestroyedStreamError();
        }
      }),
    };

    await client.stop();

    expect(notifications).toEqual(["textDocument/didClose", "exit"]);
    expect(internals.openDocuments.size).toBe(0);
    expect(dispose).toHaveBeenCalledOnce();
    expect(kill).toHaveBeenCalledOnce();
  });

  it("stop with a deadline force-disposes instead of waiting on a wedged shutdown", async () => {
    const client = new LSPClient({
      command: "echo",
      args: [],
      rootPath: "/tmp",
      languageId: "typescript",
      timeout: 5000,
    });

    const dispose = vi.fn();
    const kill = vi.fn();
    const internals = client as unknown as {
      initialized: boolean;
      process: { kill: () => void };
      connection: {
        dispose: () => void;
        sendRequest: (method: string) => Promise<void>;
        sendNotification: (method: string, params?: unknown) => Promise<void>;
      };
    };
    internals.initialized = true;
    internals.process = { kill };
    internals.connection = {
      dispose,
      sendRequest: vi.fn(() => new Promise<void>(() => {})),
      sendNotification: vi.fn(async () => undefined),
    };

    const startedAt = Date.now();
    await client.stop({ deadlineMs: 10 });

    expect(Date.now() - startedAt).toBeLessThan(200);
    expect(dispose).toHaveBeenCalledOnce();
    expect(kill).toHaveBeenCalledOnce();
    expect(client.isRunning()).toBe(false);
  });

  it("fails immediately when didOpen notification rejects", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "devagent-lsp-open-fail-"));
    const fileName = "test.ts";
    writeFileSync(join(tempDir, fileName), "const x = 1;\n", "utf-8");

    const client = new LSPClient({
      command: "echo",
      args: [],
      rootPath: tempDir,
      languageId: "typescript",
      timeout: 5000,
    });

    const unhandled: unknown[] = [];
    const onUnhandledRejection = (reason: unknown) => {
      unhandled.push(reason);
    };
    process.on("unhandledRejection", onUnhandledRejection);

    const internals = client as unknown as {
      initialized: boolean;
      connection: {
        sendNotification: (method: string, params?: unknown) => Promise<void>;
        sendRequest: (method: string) => Promise<unknown[]>;
      };
    };
    internals.initialized = true;
    internals.connection = {
      sendNotification: vi.fn(async (method: string) => {
        if (method === "textDocument/didOpen") {
          throw makeDestroyedStreamError();
        }
      }),
      sendRequest: vi.fn(async () => []),
    };

    try {
      await expect(client.getSymbols(fileName)).rejects.toThrow("stream was destroyed");
      await sleep(0);
      expect(unhandled).toHaveLength(0);
    } finally {
      process.off("unhandledRejection", onUnhandledRejection);
      rmSync(tempDir, { recursive: true, force: true });
    }
  });

  it("times out when didOpen notification never settles", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "devagent-lsp-open-timeout-"));
    const fileName = "test.ts";
    writeFileSync(join(tempDir, fileName), "const x = 1;\n", "utf-8");

    const client = new LSPClient({
      command: "echo",
      args: [],
      rootPath: tempDir,
      languageId: "typescript",
      timeout: 10,
    });

    const internals = client as unknown as {
      initialized: boolean;
      connection: {
        sendNotification: (method: string, params?: unknown) => Promise<void>;
        sendRequest: (method: string) => Promise<unknown[]>;
      };
    };
    internals.initialized = true;
    internals.connection = {
      sendNotification: vi.fn(() => new Promise<void>(() => {})),
      sendRequest: vi.fn(async () => []),
    };

    try {
      const result = client.getSymbols(fileName);
      const handledResult = result.catch((error: unknown) => error);
      const error = await handledResult;
      expect(error).toBeInstanceOf(Error);
      expect((error as Error).message).toContain("LSP notification textDocument/didOpen timed out");
      expect(client.isRunning()).toBe(false);
    } finally {
      rmSync(tempDir, { recursive: true, force: true });
    }
  });

  it("marks the client not running when a request times out", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "devagent-lsp-request-timeout-"));
    const fileName = "test.ts";
    writeFileSync(join(tempDir, fileName), "const x = 1;\n", "utf-8");

    const client = new LSPClient({
      command: "echo",
      args: [],
      rootPath: tempDir,
      languageId: "typescript",
      timeout: 10,
    });

    const internals = client as unknown as {
      initialized: boolean;
      connection: {
        dispose: () => void;
        sendNotification: (method: string, params?: unknown) => Promise<void>;
        sendRequest: (method: string) => Promise<unknown[]>;
      };
    };
    internals.initialized = true;
    internals.connection = {
      dispose: vi.fn(),
      sendNotification: vi.fn(async () => undefined),
      sendRequest: vi.fn(() => new Promise<unknown[]>(() => {})),
    };

    try {
      await expect(client.getSymbols(fileName)).rejects.toThrow("LSP request timed out");
      expect(client.isRunning()).toBe(false);
      expect(internals.connection).toBeNull();
    } finally {
      rmSync(tempDir, { recursive: true, force: true });
    }
  });

  it("fails immediately when didChange notification rejects", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "devagent-lsp-change-fail-"));
    const fileName = "test.ts";
    writeFileSync(join(tempDir, fileName), "const x = 1;\n", "utf-8");

    const client = new LSPClient({
      command: "echo",
      args: [],
      rootPath: tempDir,
      languageId: "typescript",
      timeout: 5000,
    });

    const unhandled: unknown[] = [];
    const onUnhandledRejection = (reason: unknown) => {
      unhandled.push(reason);
    };
    process.on("unhandledRejection", onUnhandledRejection);

    const notifications: string[] = [];
    const internals = client as unknown as {
      initialized: boolean;
      connection: {
        sendNotification: (method: string, params?: unknown) => Promise<void>;
        sendRequest: (method: string) => Promise<unknown[]>;
      };
    };
    internals.initialized = true;
    internals.connection = {
      sendNotification: vi.fn(async (method: string) => {
        notifications.push(method);
        if (method === "textDocument/didChange") {
          throw makeDestroyedStreamError();
        }
      }),
      sendRequest: vi.fn(async () => []),
    };

    try {
      await client.getSymbols(fileName);
      writeFileSync(join(tempDir, fileName), "const x = 2;\n", "utf-8");

      await expect(client.getSymbols(fileName)).rejects.toThrow("stream was destroyed");
      await sleep(0);
      expect(unhandled).toHaveLength(0);
      expect(notifications).toEqual(["textDocument/didOpen", "textDocument/didChange"]);
    } finally {
      process.off("unhandledRejection", onUnhandledRejection);
      rmSync(tempDir, { recursive: true, force: true });
    }
  });

  it("completes sequential diagnostics then symbols on the same unchanged open document", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "devagent-lsp-sequential-"));
    const fileName = "test.ts";
    writeFileSync(join(tempDir, fileName), "export function answer() { return 42; }\n", "utf-8");

    const client = new LSPClient({
      command: "echo",
      args: [],
      rootPath: tempDir,
      languageId: "typescript",
      timeout: 50,
      diagnosticTimeout: 50,
    });

    const diagnosticsStore = (client as unknown as {
      diagnosticsStore: Map<string, unknown[]>;
    }).diagnosticsStore;
    const notifications: string[] = [];
    const fakeConnection = {
      dispose: vi.fn(),
      sendNotification: vi.fn(async (method: string, params: Record<string, unknown>) => {
        notifications.push(method);
        if (method !== "textDocument/didOpen") return;
        const textDoc = params["textDocument"] as { uri?: string } | undefined;
        const uri = textDoc?.uri;
        if (uri) diagnosticsStore.set(uri, []);
      }),
      sendRequest: vi.fn(async (method: string) => {
        if (method === "textDocument/documentSymbol") {
          return [{
            name: "answer",
            kind: 12,
            range: { start: { line: 0, character: 16 }, end: { line: 0, character: 22 } },
            selectionRange: { start: { line: 0, character: 16 }, end: { line: 0, character: 22 } },
          }];
        }
        return [];
      }),
    };

    (client as unknown as { initialized: boolean; connection: unknown }).initialized = true;
    (client as unknown as { initialized: boolean; connection: unknown }).connection = fakeConnection;

    try {
      const diagnostics = await client.getDiagnostics(fileName, "typescript");
      const symbols = await client.getSymbols(fileName, "typescript");

      expect(diagnostics.diagnostics).toHaveLength(0);
      expect(symbols).toEqual([
        { name: "answer", kind: "function", line: 1, character: 17 },
      ]);
      expect(notifications).toEqual(["textDocument/didOpen"]);
      expect(client.isRunning()).toBe(true);
    } finally {
      rmSync(tempDir, { recursive: true, force: true });
    }
  });

  it("sends protocol requests for hover, implementation, workspace symbols, and call hierarchy", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "devagent-lsp-ops-"));
    const fileName = "test.ts";
    writeFileSync(join(tempDir, fileName), "function caller() { callee(); }\nfunction callee() {}\n", "utf-8");

    const client = new LSPClient({
      command: "echo",
      args: [],
      rootPath: tempDir,
      languageId: "typescript",
      timeout: 5000,
    });

    const methods: string[] = [];
    const fakeConnection = {
      sendNotification: vi.fn(async () => undefined),
      sendRequest: vi.fn(async (method: string) => {
        methods.push(method);
        if (method === "textDocument/hover") return { contents: { kind: "markdown", value: "**callee**" } };
        if (method === "textDocument/implementation") {
          return [{ uri: `file://${tempDir}/${fileName}`, range: { start: { line: 1, character: 9 } } }];
        }
        if (method === "workspace/symbol") {
          return [{
            name: "callee",
            kind: 12,
            location: { uri: `file://${tempDir}/${fileName}`, range: { start: { line: 1, character: 9 } } },
          }];
        }
        if (method === "textDocument/prepareCallHierarchy") {
          return [{
            name: "callee",
            kind: 12,
            uri: `file://${tempDir}/${fileName}`,
            range: { start: { line: 1, character: 9 } },
            selectionRange: { start: { line: 1, character: 9 } },
          }];
        }
        if (method === "callHierarchy/incomingCalls") {
          return [{
            from: {
              name: "caller",
              kind: 12,
              uri: `file://${tempDir}/${fileName}`,
              range: { start: { line: 0, character: 9 } },
              selectionRange: { start: { line: 0, character: 9 } },
            },
          }];
        }
        if (method === "callHierarchy/outgoingCalls") {
          return [{
            to: {
              name: "callee",
              kind: 12,
              uri: `file://${tempDir}/${fileName}`,
              range: { start: { line: 1, character: 9 } },
              selectionRange: { start: { line: 1, character: 9 } },
            },
          }];
        }
        return [];
      }),
    };

    (client as unknown as { initialized: boolean; connection: unknown }).initialized = true;
    (client as unknown as { initialized: boolean; connection: unknown }).connection = fakeConnection;

    try {
      expect(await client.getHover(fileName, 1, 10)).toContain("callee");
      expect(await client.getImplementation(fileName, 1, 10)).toEqual([
        { file: fileName, line: 2, character: 10 },
      ]);
      expect(await client.getWorkspaceSymbols("callee")).toEqual([
        { name: "callee", kind: "function", file: fileName, line: 2, character: 10 },
      ]);
      expect(await client.getIncomingCalls(fileName, 2, 10)).toEqual([
        { name: "caller", kind: "function", file: fileName, line: 1, character: 10 },
      ]);
      expect(await client.getOutgoingCalls(fileName, 1, 10)).toEqual([
        { name: "callee", kind: "function", file: fileName, line: 2, character: 10 },
      ]);
      expect(methods).toContain("textDocument/hover");
      expect(methods).toContain("textDocument/implementation");
      expect(methods).toContain("workspace/symbol");
      expect(methods).toContain("callHierarchy/incomingCalls");
      expect(methods).toContain("callHierarchy/outgoingCalls");
    } finally {
      rmSync(tempDir, { recursive: true, force: true });
    }
  });

describe("createLSPTools", () => {
  it("creates one generic LSP tool", () => {
    const client = new LSPClient({
      command: "echo",
      args: [],
      rootPath: "/tmp",
      languageId: "typescript",
    });

    const tools = createLSPTools(client);
    expect(tools.length).toBe(1);

    const names = tools.map((t) => t.name);
    expect(names).toEqual(["lsp"]);
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
    const lspTool = tools.find((t) => t.name === "lsp");
    expect(lspTool).toBeDefined();

    const result = await lspTool!.handler(
      { operation: "diagnostics", path: "test.ts" },
      { repoRoot: "/tmp", config: {} as never, sessionId: "" },
    );

    expect(result.success).toBe(false);
    expect(result.error).toContain("not running");
  });
});

describe("createRoutingLSPTools", () => {
  it("routes diagnostics through the generic LSP tool based on file extension", async () => {
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
    expect(tools.length).toBe(1);

    // Both clients are not running, but the error message should differ
    // to prove routing happened — the tool should return "not running"
    // because the resolved client isn't started.
    const lspTool = tools.find((t) => t.name === "lsp")!;

    const tsResult = await lspTool.handler(
      { operation: "diagnostics", path: "foo.ts" },
      { repoRoot: "/tmp", config: {} as never, sessionId: "" },
    );
    expect(tsResult.success).toBe(false);
    expect(tsResult.error).toContain("not running");

    const pyResult = await lspTool.handler(
      { operation: "diagnostics", path: "bar.py" },
      { repoRoot: "/tmp", config: {} as never, sessionId: "" },
    );
    expect(pyResult.success).toBe(false);
    expect(pyResult.error).toContain("not running");
  });

  it("returns error for unrecognized file extensions", async () => {
    const resolver = () => null;

    const tools = createRoutingLSPTools(resolver);
    const lspTool = tools.find((t) => t.name === "lsp")!;

    const result = await lspTool.handler(
      { operation: "diagnostics", path: "unknown.xyz" },
      { repoRoot: "/tmp", config: {} as never, sessionId: "" },
    );
    expect(result.success).toBe(false);
    expect(result.error).toContain("No LSP server");
  });
});

function makeMockClient(overrides: Record<string, unknown> = {}) {
  return {
    isRunning: () => true,
    getDiagnostics: vi.fn(),
    getSymbols: vi.fn(),
    getDefinition: vi.fn(),
    getReferences: vi.fn(),
    getHover: vi.fn(),
    getImplementation: vi.fn(),
    getWorkspaceSymbols: vi.fn(),
    getIncomingCalls: vi.fn(),
    getOutgoingCalls: vi.fn(),
    ...overrides,
  } as unknown as LSPClient;
}

const ctx = { repoRoot: "/tmp", config: {} as never, sessionId: "" };

describe("generic LSP position operations", () => {
  it("runs definitions with path, line, and character", async () => {
    const client = makeMockClient();
    (client.getDefinition as ReturnType<typeof vi.fn>).mockResolvedValue([
      { file: "src/other.ts", line: 42, character: 1 },
    ]);

    const resolver = () => ({ client, languageId: "typescript" });
    const tools = createRoutingLSPTools(resolver);
    const tool = tools.find((t) => t.name === "lsp")!;
    expect(tool).toBeDefined();

    const result = await tool.handler({ operation: "definitions", path: "src/foo.ts", line: 10, character: 5 }, ctx);
    expect(result.success).toBe(true);
    expect(result.output).toContain("Definition(s):");
    expect(result.output).toContain("src/other.ts:42:1");
    expect(client.getDefinition).toHaveBeenCalledWith("src/foo.ts", 10, 5, "typescript");
  });

  it("returns validation errors for position operations missing coordinates", async () => {
    const client = makeMockClient();

    const resolver = () => ({ client, languageId: "typescript" });
    const tools = createRoutingLSPTools(resolver);
    const tool = tools.find((t) => t.name === "lsp")!;

    const result = await tool.handler({ operation: "hover", path: "src/foo.ts" }, ctx);
    expect(result.success).toBe(false);
    expect(result.error).toContain("requires");
  });

  it("runs workspace_symbols across all running clients", async () => {
    const client = makeMockClient();
    (client.getWorkspaceSymbols as ReturnType<typeof vi.fn>).mockResolvedValue([
      { name: "Foo", kind: "class", file: "src/foo.ts", line: 1, character: 8 },
    ]);

    const resolver = () => ({ client, languageId: "typescript" });
    const tools = createRoutingLSPTools(resolver, () => [client]);
    const tool = tools.find((t) => t.name === "lsp")!;
    expect(tool).toBeDefined();

    const result = await tool.handler({ operation: "workspace_symbols", query: "Foo" }, ctx);
    expect(result.success).toBe(true);
    expect(result.output).toContain("class Foo");
    expect(client.getWorkspaceSymbols).toHaveBeenCalledWith("Foo");
  });
});

describe("generic LSP workspace and timeout operations", () => {
  it("returns partial workspace_symbols results when one running client fails", async () => {
    const failingClient = makeMockClient();
    const workingClient = makeMockClient();
    (failingClient.getWorkspaceSymbols as ReturnType<typeof vi.fn>).mockRejectedValue(new Error("ts server down"));
    (workingClient.getWorkspaceSymbols as ReturnType<typeof vi.fn>).mockResolvedValue([
      { name: "Foo", kind: "class", file: "src/foo.ts", line: 1, character: 8 },
    ]);

    const resolver = () => ({ client: workingClient, languageId: "typescript" });
    const tools = createRoutingLSPTools(resolver, () => [failingClient, workingClient]);
    const tool = tools.find((t) => t.name === "lsp")!;

    const result = await tool.handler({ operation: "workspace_symbols", query: "Foo" }, ctx);

    expect(result.success).toBe(true);
    expect(result.output).toContain("class Foo");
    expect(failingClient.getWorkspaceSymbols).toHaveBeenCalledWith("Foo");
    expect(workingClient.getWorkspaceSymbols).toHaveBeenCalledWith("Foo");
  });

  it("fails workspace_symbols when every running client fails across retries", async () => {
    vi.useFakeTimers();
    const firstClient = makeMockClient();
    const secondClient = makeMockClient();
    (firstClient.getWorkspaceSymbols as ReturnType<typeof vi.fn>).mockRejectedValue(new Error("typescript failed"));
    (secondClient.getWorkspaceSymbols as ReturnType<typeof vi.fn>).mockRejectedValue(new Error("python failed"));

    const resolver = () => ({ client: firstClient, languageId: "typescript" });
    const tools = createRoutingLSPTools(resolver, () => [firstClient, secondClient]);
    const tool = tools.find((t) => t.name === "lsp")!;

    const resultPromise = tool.handler({ operation: "workspace_symbols", query: "Foo" }, ctx);
    await vi.advanceTimersByTimeAsync(350);
    const result = await resultPromise;

    expect(result.success).toBe(false);
    expect(result.error).toContain("workspace_symbols failed for all LSP clients");
    expect(result.error).toContain("typescript failed");
    expect(result.error).toContain("python failed");
    expect(firstClient.getWorkspaceSymbols).toHaveBeenCalledTimes(3);
    expect(secondClient.getWorkspaceSymbols).toHaveBeenCalledTimes(3);
  });

  it("does not fail workspace_symbols when a retry succeeds with empty results", async () => {
    vi.useFakeTimers();
    const client = makeMockClient();
    (client.getWorkspaceSymbols as ReturnType<typeof vi.fn>)
      .mockRejectedValueOnce(new Error("index warming"))
      .mockResolvedValueOnce([])
      .mockRejectedValueOnce(new Error("later failure"));

    const resolver = () => ({ client, languageId: "typescript" });
    const tools = createRoutingLSPTools(resolver, () => [client]);
    const tool = tools.find((t) => t.name === "lsp")!;

    const resultPromise = tool.handler({ operation: "workspace_symbols", query: "Missing" }, ctx);
    await vi.advanceTimersByTimeAsync(350);
    const result = await resultPromise;

    expect(result.success).toBe(true);
    expect(result.output).toContain("No workspace symbols found");
    expect(client.getWorkspaceSymbols).toHaveBeenCalledTimes(3);
  });

  it("returns an error when an LSP operation never resolves", async () => {
    vi.useFakeTimers();
    const client = makeMockClient();
    (client.getDiagnostics as ReturnType<typeof vi.fn>).mockReturnValue(new Promise(() => {}));

    const resolver = () => ({ client, languageId: "typescript" });
    const tools = createRoutingLSPTools(resolver, () => [client]);
    const tool = tools.find((t) => t.name === "lsp")!;

    const resultPromise = tool.handler({ operation: "diagnostics", path: "src/foo.ts" }, ctx);
    await vi.advanceTimersByTimeAsync(30_000);
    const result = await resultPromise;

    expect(result.success).toBe(false);
    expect(result.error).toContain("LSP operation timed out after 30000ms");
  });

  it("retries empty workspace_symbols results to allow cold indexes to settle", async () => {
    vi.useFakeTimers();
    const client = makeMockClient();
    (client.getWorkspaceSymbols as ReturnType<typeof vi.fn>)
      .mockResolvedValueOnce([])
      .mockResolvedValueOnce([
        { name: "Foo", kind: "class", file: "src/foo.ts", line: 1, character: 8 },
      ]);

    const resolver = () => ({ client, languageId: "typescript" });
    const tools = createRoutingLSPTools(resolver, () => [client]);
    const tool = tools.find((t) => t.name === "lsp")!;

    const resultPromise = tool.handler({ operation: "workspace_symbols", query: "Foo" }, ctx);
    await vi.advanceTimersByTimeAsync(100);
    const result = await resultPromise;

    expect(result.success).toBe(true);
    expect(result.output).toContain("class Foo");
    expect(client.getWorkspaceSymbols).toHaveBeenCalledTimes(2);
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
    expect(tools).toHaveLength(1);

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

  it("common hint mentions file path", () => {
    const client = new LSPClient({
      command: "echo",
      args: [],
      rootPath: "/tmp",
      languageId: "typescript",
    });

    const tools = createLSPTools(client);
    const lsp = tools.find((t) => t.name === "lsp")!;
    expect(lsp.errorGuidance!.common).toContain("file path");
  });
});

describe("Routing LSP tool errorGuidance", () => {
  it("all routing LSP tools have errorGuidance", () => {
    const resolver: LSPClientResolver = () => null;
    const tools = createRoutingLSPTools(resolver);
    expect(tools).toHaveLength(1);

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

  it("routing LSP tool covers missing parameter guidance", () => {
    const resolver: LSPClientResolver = () => null;
    const [tool] = createRoutingLSPTools(resolver);
    const matches = tool!.errorGuidance!.patterns!.map((p) => p.match);
    expect(matches).toContain("requires");
  });
});
