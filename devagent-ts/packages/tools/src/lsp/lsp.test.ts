import { describe, it, expect } from "vitest";
import { LSPClient } from "./client.js";
import { createLSPTools } from "./tools.js";

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
