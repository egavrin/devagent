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

  it("creates four tools: memory_store, memory_recall, memory_list, memory_delete", () => {
    expect(tools).toHaveLength(4);
    expect(tools[0]!.name).toBe("memory_store");
    expect(tools[1]!.name).toBe("memory_recall");
    expect(tools[2]!.name).toBe("memory_list");
    expect(tools[3]!.name).toBe("memory_delete");
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

  // ─── memory_list ──────────────────────────────────────────

  it("memory_list has readonly category", () => {
    expect(tools[2]!.category).toBe("readonly");
  });

  it("memory_list returns all memories with summary header", async () => {
    const ctx = makeToolContext();

    await tools[0]!.handler(
      { category: "pattern", key: "import-style", content: "Use named imports" },
      ctx,
    );
    await tools[0]!.handler(
      { category: "mistake", key: "missing-await", content: "Always await async calls" },
      ctx,
    );
    await tools[0]!.handler(
      { category: "pattern", key: "test-framework", content: "Use vitest" },
      ctx,
    );

    const listTool = tools[2]!;
    const result = await listTool.handler({}, ctx);
    expect(result.success).toBe(true);

    // Should include summary header
    expect(result.output).toContain("Total: 3 memories");
    expect(result.output).toContain("pattern: 2");
    expect(result.output).toContain("mistake: 1");

    // Should include individual entries with IDs and metadata
    expect(result.output).toContain("import-style");
    expect(result.output).toContain("missing-await");
    expect(result.output).toContain("test-framework");
    expect(result.output).toContain("accessed:");
    expect(result.output).toContain("relevance:");
  });

  it("memory_list filters by category", async () => {
    const ctx = makeToolContext();

    await tools[0]!.handler(
      { category: "pattern", key: "p1", content: "Pattern 1" },
      ctx,
    );
    await tools[0]!.handler(
      { category: "decision", key: "decision_key", content: "Decision 1" },
      ctx,
    );

    const listTool = tools[2]!;
    const result = await listTool.handler({ category: "pattern" }, ctx);
    expect(result.success).toBe(true);
    expect(result.output).toContain("p1");
    expect(result.output).not.toContain("decision_key");
  });

  it("memory_list returns empty message when no memories", async () => {
    const ctx = makeToolContext();

    const listTool = tools[2]!;
    const result = await listTool.handler({}, ctx);
    expect(result.success).toBe(true);
    expect(result.output).toBe("No memories stored.");
  });

  // ─── memory_delete ────────────────────────────────────────

  it("memory_delete has workflow category", () => {
    expect(tools[3]!.category).toBe("workflow");
    expect((tools[3]!.paramSchema as any).required).toContain("id");
  });

  it("memory_delete removes a memory by ID", async () => {
    const ctx = makeToolContext();

    // Store a memory
    const storeResult = await tools[0]!.handler(
      { category: "pattern", key: "to-delete", content: "Will be removed" },
      ctx,
    );

    // Extract the ID from the output (format: "Memory stored: [pattern] to-delete (id: <uuid>)")
    const idMatch = storeResult.output.match(/id: ([a-f0-9-]+)/);
    expect(idMatch).not.toBeNull();
    const memoryId = idMatch![1]!;

    // Delete it
    const deleteTool = tools[3]!;
    const deleteResult = await deleteTool.handler({ id: memoryId }, ctx);
    expect(deleteResult.success).toBe(true);
    expect(deleteResult.output).toContain("deleted");

    // Verify it's gone
    const recallResult = await tools[1]!.handler(
      { query: "to-delete" },
      ctx,
    );
    expect(recallResult.output).toBe("No relevant memories found.");
  });

  it("memory_delete returns error for nonexistent ID", async () => {
    const ctx = makeToolContext();

    const deleteTool = tools[3]!;
    const result = await deleteTool.handler(
      { id: "nonexistent-id-12345" },
      ctx,
    );
    expect(result.success).toBe(false);
    expect(result.output).toContain("not found");
    expect(result.error).toContain("No memory with ID");
  });

  // ─── Configurable recall limits ───────────────────────────

  it("memory_recall respects configurable minRelevance", async () => {
    // Create tools with high minimum relevance
    const strictTools = createMemoryTools(memoryStore, {
      recallMinRelevance: 0.9,
      recallLimit: 10,
    });
    const ctx = makeToolContext();

    // Store a memory with default relevance (1.0)
    await strictTools[0]!.handler(
      { category: "pattern", key: "high-rel", content: "High relevance memory" },
      ctx,
    );
    // Store a memory, then manually lower its relevance
    await strictTools[0]!.handler(
      { category: "pattern", key: "low-rel", content: "Low relevance memory" },
      ctx,
    );
    // Lower the relevance of the second one directly
    const storeAny = memoryStore as unknown as {
      db: { prepare: (sql: string) => { run: (...args: unknown[]) => void } };
    };
    storeAny.db
      .prepare("UPDATE memories SET relevance = 0.5 WHERE key = ?")
      .run("low-rel");

    // Recall should only return the high-relevance memory
    const recallTool = strictTools[1]!;
    const result = await recallTool.handler({}, ctx);
    expect(result.success).toBe(true);
    expect(result.output).toContain("high-rel");
    expect(result.output).not.toContain("low-rel");
  });

  it("memory_recall respects configurable limit", async () => {
    // Create tools with limit of 2
    const limitedTools = createMemoryTools(memoryStore, {
      recallMinRelevance: 0.1,
      recallLimit: 2,
    });
    const ctx = makeToolContext();

    // Store 5 memories
    for (let i = 0; i < 5; i++) {
      await limitedTools[0]!.handler(
        { category: "context", key: `ctx-${i}`, content: `Context ${i}` },
        ctx,
      );
    }

    // Recall should return at most 2
    const recallTool = limitedTools[1]!;
    const result = await recallTool.handler({ category: "context" }, ctx);
    expect(result.success).toBe(true);
    const lines = result.output.split("\n").filter((l: string) => l.trim().length > 0);
    expect(lines).toHaveLength(2);
  });
});
