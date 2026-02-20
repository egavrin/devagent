import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { createMemoryTools } from "./memory-tools.js";
import { MemoryStore } from "@devagent/core";
import type { ToolSpec, DevAgentConfig } from "@devagent/core";
import { ApprovalMode } from "@devagent/core";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { mkdirSync, rmSync } from "node:fs";
import { randomUUID } from "node:crypto";

// ─── Helpers ────────────────────────────────────────────────

function makeTmpDir(): string {
  const dir = join(tmpdir(), `devagent-memory-tools-test-${randomUUID()}`);
  mkdirSync(dir, { recursive: true });
  return dir;
}

function makeToolContext() {
  return {
    repoRoot: "/tmp/test",
    config: {
      provider: "mock",
      model: "mock-model",
      providers: {},
      approval: {
        mode: ApprovalMode.FULL_AUTO,
        autoApprovePlan: false,
        autoApproveCode: false,
        autoApproveShell: false,
        auditLog: false,
        toolOverrides: {},
        pathRules: [],
      },
      budget: {
        maxIterations: 10,
        maxContextTokens: 100_000,
        responseHeadroom: 2_000,
        costWarningThreshold: 1.0,
        enableCostTracking: true,
      },
      context: {
        pruningStrategy: "hybrid" as const,
        triggerRatio: 0.8,
        keepRecentMessages: 10,
      },
      arkts: {
        enabled: false,
        strictMode: false,
        targetVersion: "5.0",
      },
    } as DevAgentConfig,
    sessionId: "test-session",
  };
}

describe("memory-tools", () => {
  let memoryStore: MemoryStore;
  let tools: ToolSpec[];
  let tmpDir: string;

  beforeEach(() => {
    tmpDir = makeTmpDir();
    memoryStore = new MemoryStore({ dbPath: join(tmpDir, "memory.db") });
    tools = createMemoryTools(memoryStore);
  });

  afterEach(() => {
    memoryStore.close();
    rmSync(tmpDir, { recursive: true, force: true });
  });

  it("creates two tools: memory_store and memory_recall", () => {
    expect(tools).toHaveLength(2);
    expect(tools[0]!.name).toBe("memory_store");
    expect(tools[1]!.name).toBe("memory_recall");
  });

  it("memory_store has correct category and schema", () => {
    const storeTool = tools[0]!;
    expect(storeTool.category).toBe("workflow");
    expect(storeTool.paramSchema).toBeDefined();
    expect((storeTool.paramSchema as any).required).toContain("category");
    expect((storeTool.paramSchema as any).required).toContain("key");
    expect((storeTool.paramSchema as any).required).toContain("content");
  });

  it("memory_recall has readonly category", () => {
    const recallTool = tools[1]!;
    expect(recallTool.category).toBe("readonly");
  });

  it("stores a memory and recalls it", async () => {
    const ctx = makeToolContext();

    // Store
    const storeResult = await tools[0]!.handler(
      {
        category: "pattern",
        key: "test-framework",
        content: "This project uses vitest for testing",
      },
      ctx,
    );
    expect(storeResult.success).toBe(true);
    expect(storeResult.output).toContain("Memory stored");
    expect(storeResult.output).toContain("pattern");
    expect(storeResult.output).toContain("test-framework");

    // Recall
    const recallResult = await tools[1]!.handler(
      { query: "test" },
      ctx,
    );
    expect(recallResult.success).toBe(true);
    expect(recallResult.output).toContain("test-framework");
    expect(recallResult.output).toContain("vitest");
  });

  it("recall returns empty message when no matches", async () => {
    const ctx = makeToolContext();

    const result = await tools[1]!.handler(
      { query: "nonexistent-thing" },
      ctx,
    );
    expect(result.success).toBe(true);
    expect(result.output).toBe("No relevant memories found.");
  });

  it("recall filters by category", async () => {
    const ctx = makeToolContext();

    // Store two memories in different categories
    await tools[0]!.handler(
      { category: "pattern", key: "import-style", content: "Use named imports" },
      ctx,
    );
    await tools[0]!.handler(
      { category: "mistake", key: "missing-await", content: "Always await async calls" },
      ctx,
    );

    // Recall only mistakes
    const result = await tools[1]!.handler(
      { category: "mistake" },
      ctx,
    );
    expect(result.success).toBe(true);
    expect(result.output).toContain("missing-await");
    expect(result.output).not.toContain("import-style");
  });

  it("store overwrites existing memory with same category+key", async () => {
    const ctx = makeToolContext();

    await tools[0]!.handler(
      { category: "decision", key: "db-choice", content: "Using SQLite" },
      ctx,
    );
    await tools[0]!.handler(
      { category: "decision", key: "db-choice", content: "Switched to PostgreSQL" },
      ctx,
    );

    const result = await tools[1]!.handler(
      { query: "db-choice" },
      ctx,
    );
    expect(result.success).toBe(true);
    expect(result.output).toContain("PostgreSQL");
    expect(result.output).not.toContain("SQLite");
  });
});
