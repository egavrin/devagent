import { describe, it, expect, vi } from "vitest";
import {
  createLSPDiagnosticProvider,
  createRoutingDiagnosticProvider,
  createShellTestRunner,
  detectLanguage,
  getLanguageEntry,
  LSPRouter,
  LANGUAGE_MAP,
} from "./double-check-wiring.js";
import type { LSPClient } from "@devagent/tools";

// ─── Language Detection ──────────────────────────────────────

describe("detectLanguage", () => {
  it("maps .ts to typescript", () => {
    expect(detectLanguage("src/foo.ts")).toBe("typescript");
  });

  it("maps .tsx to typescript", () => {
    expect(detectLanguage("components/App.tsx")).toBe("typescript");
  });

  it("maps .js to javascript", () => {
    expect(detectLanguage("lib/util.js")).toBe("javascript");
  });

  it("maps .py to python", () => {
    expect(detectLanguage("scripts/main.py")).toBe("python");
  });

  it("maps .c to c", () => {
    expect(detectLanguage("src/main.c")).toBe("c");
  });

  it("maps .cpp to cpp", () => {
    expect(detectLanguage("src/engine.cpp")).toBe("cpp");
  });

  it("maps .rs to rust", () => {
    expect(detectLanguage("src/lib.rs")).toBe("rust");
  });

  it("maps .sh to shellscript", () => {
    expect(detectLanguage("scripts/build.sh")).toBe("shellscript");
  });

  it("maps .ets to arkts", () => {
    expect(detectLanguage("entry/src/Index.ets")).toBe("arkts");
  });

  it("returns null for unknown extensions", () => {
    expect(detectLanguage("data/config.toml")).toBeNull();
    expect(detectLanguage("README.md")).toBeNull();
    expect(detectLanguage("image.png")).toBeNull();
  });

  it("handles case-insensitive extensions", () => {
    expect(detectLanguage("src/Main.TS")).toBe("typescript");
    expect(detectLanguage("lib/App.JSX")).toBe("javascript");
  });
});

describe("getLanguageEntry", () => {
  it("returns entry for known language", () => {
    const entry = getLanguageEntry("typescript");
    expect(entry).toBeDefined();
    expect(entry!.languageId).toBe("typescript");
    expect(entry!.extensions).toContain(".ts");
    expect(entry!.defaultCommand).toBe("typescript-language-server");
  });

  it("returns undefined for unknown language", () => {
    expect(getLanguageEntry("cobol")).toBeUndefined();
  });
});

describe("LANGUAGE_MAP", () => {
  it("contains 8 language entries", () => {
    expect(LANGUAGE_MAP).toHaveLength(8);
  });

  it("covers all target languages", () => {
    const ids = LANGUAGE_MAP.map((e) => e.languageId);
    expect(ids).toContain("typescript");
    expect(ids).toContain("javascript");
    expect(ids).toContain("python");
    expect(ids).toContain("c");
    expect(ids).toContain("cpp");
    expect(ids).toContain("rust");
    expect(ids).toContain("shellscript");
    expect(ids).toContain("arkts");
  });
});

// ─── LSP Router ──────────────────────────────────────────────

describe("LSPRouter", () => {
  function createMockClient(languageId: string): LSPClient {
    return {
      start: vi.fn().mockResolvedValue(undefined),
      stop: vi.fn().mockResolvedValue(undefined),
      getDiagnostics: vi.fn().mockResolvedValue({
        file: "test.ts",
        diagnostics: [
          { line: 1, character: 0, message: `Error from ${languageId}`, severity: "error" },
        ],
      }),
      getDefinition: vi.fn(),
      getReferences: vi.fn(),
      getSymbols: vi.fn(),
    } as unknown as LSPClient;
  }

  it("routes .ts files to the typescript client", async () => {
    const router = new LSPRouter("/tmp");

    // Manually inject a mock client (bypass addServer which calls client.start())
    const mockClient = createMockClient("typescript");
    (router as unknown as { clients: Map<string, LSPClient> }).clients.set(
      "typescript",
      mockClient,
    );
    (router as unknown as { extensionMap: Map<string, string> }).extensionMap.set(
      ".ts",
      "typescript",
    );

    const match = router.getClientForFile("src/foo.ts");
    expect(match).not.toBeNull();
    expect(match!.languageId).toBe("typescript");
    expect(match!.client).toBe(mockClient);
  });

  it("routes .rs files to the rust client", async () => {
    const router = new LSPRouter("/tmp");

    const mockClient = createMockClient("rust");
    (router as unknown as { clients: Map<string, LSPClient> }).clients.set(
      "rust",
      mockClient,
    );
    (router as unknown as { extensionMap: Map<string, string> }).extensionMap.set(
      ".rs",
      "rust",
    );

    const match = router.getClientForFile("src/lib.rs");
    expect(match).not.toBeNull();
    expect(match!.languageId).toBe("rust");
  });

  it("returns null for unregistered extensions", () => {
    const router = new LSPRouter("/tmp");
    expect(router.getClientForFile("config.toml")).toBeNull();
    expect(router.getClientForFile("README.md")).toBeNull();
  });

  it("returns unique clients from getClients()", () => {
    const router = new LSPRouter("/tmp");

    // Same client handles both TS and JS
    const sharedClient = createMockClient("typescript");
    const clients = (router as unknown as { clients: Map<string, LSPClient> })
      .clients;
    clients.set("typescript", sharedClient);
    clients.set("javascript", sharedClient);

    const unique = router.getClients();
    expect(unique).toHaveLength(1);
    expect(unique[0]).toBe(sharedClient);
  });

  it("returns all registered languages", () => {
    const router = new LSPRouter("/tmp");
    const clients = (router as unknown as { clients: Map<string, LSPClient> })
      .clients;
    clients.set("typescript", createMockClient("typescript"));
    clients.set("python", createMockClient("python"));

    const languages = router.getLanguages();
    expect(languages).toContain("typescript");
    expect(languages).toContain("python");
    expect(languages).toHaveLength(2);
  });

  it("clears state on stopAll()", async () => {
    const router = new LSPRouter("/tmp");
    const mockClient = createMockClient("typescript");
    (router as unknown as { clients: Map<string, LSPClient> }).clients.set(
      "typescript",
      mockClient,
    );
    (router as unknown as { extensionMap: Map<string, string> }).extensionMap.set(
      ".ts",
      "typescript",
    );

    await router.stopAll();

    expect(router.getClientForFile("src/foo.ts")).toBeNull();
    expect(router.getClients()).toHaveLength(0);
    expect(router.getLanguages()).toHaveLength(0);
    expect(mockClient.stop).toHaveBeenCalled();
  });
});

// ─── Routing Diagnostic Provider ─────────────────────────────

describe("createRoutingDiagnosticProvider", () => {
  it("skips files with no matching LSP server", async () => {
    const router = new LSPRouter("/tmp");
    // No servers registered
    const provider = createRoutingDiagnosticProvider(router);
    const result = await provider("config.toml");
    expect(result).toHaveLength(0);
  });

  it("routes diagnostics to the correct client", async () => {
    const router = new LSPRouter("/tmp");
    const mockClient = {
      getDiagnostics: vi.fn().mockResolvedValue({
        file: "src/app.ts",
        diagnostics: [
          { line: 5, character: 10, message: "Missing return", severity: "error" },
        ],
      }),
    } as unknown as LSPClient;

    (router as unknown as { clients: Map<string, LSPClient> }).clients.set(
      "typescript",
      mockClient,
    );
    (router as unknown as { extensionMap: Map<string, string> }).extensionMap.set(
      ".ts",
      "typescript",
    );

    const provider = createRoutingDiagnosticProvider(router);
    const result = await provider("src/app.ts");

    expect(result).toHaveLength(1);
    expect(result[0]!.message).toBe("src/app.ts:5:10: Missing return");
    expect(result[0]!.severity).toBe("error");
    // Verify the correct languageId was passed
    expect(mockClient.getDiagnostics).toHaveBeenCalledWith(
      "src/app.ts",
      "typescript",
    );
  });

  it("routes different file types to different clients", async () => {
    const router = new LSPRouter("/tmp");

    const tsClient = {
      getDiagnostics: vi.fn().mockResolvedValue({
        file: "src/app.ts",
        diagnostics: [{ line: 1, character: 0, message: "TS error", severity: "error" }],
      }),
    } as unknown as LSPClient;

    const pyClient = {
      getDiagnostics: vi.fn().mockResolvedValue({
        file: "scripts/main.py",
        diagnostics: [{ line: 2, character: 0, message: "PY error", severity: "warning" }],
      }),
    } as unknown as LSPClient;

    const clients = (router as unknown as { clients: Map<string, LSPClient> }).clients;
    const extMap = (router as unknown as { extensionMap: Map<string, string> }).extensionMap;

    clients.set("typescript", tsClient);
    clients.set("python", pyClient);
    extMap.set(".ts", "typescript");
    extMap.set(".py", "python");

    const provider = createRoutingDiagnosticProvider(router);

    const tsResult = await provider("src/app.ts");
    expect(tsResult[0]!.message).toContain("TS error");
    expect(tsClient.getDiagnostics).toHaveBeenCalledWith("src/app.ts", "typescript");

    const pyResult = await provider("scripts/main.py");
    expect(pyResult[0]!.message).toContain("PY error");
    expect(pyClient.getDiagnostics).toHaveBeenCalledWith("scripts/main.py", "python");
  });
});

// ─── LSP Diagnostic Provider (single-client, backward compat) ─

describe("createLSPDiagnosticProvider", () => {
  it("adapts LSPClient diagnostic results", async () => {
    const mockClient = {
      getDiagnostics: vi.fn().mockResolvedValue({
        file: "src/foo.ts",
        diagnostics: [
          { line: 10, character: 5, message: "Type error", severity: "error" },
          { line: 20, character: 1, message: "Unused var", severity: "warning" },
        ],
      }),
    } as unknown as LSPClient;

    const provider = createLSPDiagnosticProvider(mockClient);
    const result = await provider("src/foo.ts");

    expect(result).toHaveLength(2);
    expect(result[0]!.message).toBe("src/foo.ts:10:5: Type error");
    expect(result[0]!.severity).toBe("error");
    expect(result[1]!.message).toBe("src/foo.ts:20:1: Unused var");
    expect(result[1]!.severity).toBe("warning");
  });

  it("returns empty array for clean files", async () => {
    const mockClient = {
      getDiagnostics: vi.fn().mockResolvedValue({
        file: "src/clean.ts",
        diagnostics: [],
      }),
    } as unknown as LSPClient;

    const provider = createLSPDiagnosticProvider(mockClient);
    const result = await provider("src/clean.ts");

    expect(result).toHaveLength(0);
  });

  it("propagates LSP errors", async () => {
    const mockClient = {
      getDiagnostics: vi.fn().mockRejectedValue(new Error("LSP crashed")),
    } as unknown as LSPClient;

    const provider = createLSPDiagnosticProvider(mockClient);
    await expect(provider("src/foo.ts")).rejects.toThrow("LSP crashed");
  });
});

// ─── Shell Test Runner ─────────────────────────────────────

describe("createShellTestRunner", () => {
  it("runs command and returns output on success", async () => {
    const runner = createShellTestRunner("/tmp");
    const result = await runner("echo 'all tests passed'");

    expect(result.success).toBe(true);
    expect(result.output).toContain("all tests passed");
  });

  it("returns failure for non-zero exit code", async () => {
    const runner = createShellTestRunner("/tmp");
    const result = await runner("exit 1");

    expect(result.success).toBe(false);
  });

  it("handles spawn errors for invalid commands", async () => {
    const runner = createShellTestRunner("/tmp");
    // A command that fails at the shell level
    const result = await runner("/nonexistent/binary/that/does/not/exist 2>/dev/null");

    expect(result.success).toBe(false);
  });

  it("captures stderr in output", async () => {
    const runner = createShellTestRunner("/tmp");
    const result = await runner("echo 'out' && echo 'err' >&2");

    expect(result.output).toContain("out");
    expect(result.output).toContain("err");
    expect(result.output).toContain("--- stderr ---");
  });
});
