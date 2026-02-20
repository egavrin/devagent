import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { MemoryStore } from "./memory.js";
import { mkdirSync, rmSync } from "node:fs";
import { join } from "node:path";

const TEST_DIR = "/tmp/devagent-memory-test";
const TEST_DB = join(TEST_DIR, "memory.db");

describe("MemoryStore", () => {
  let store: MemoryStore;

  beforeEach(() => {
    mkdirSync(TEST_DIR, { recursive: true });
    store = new MemoryStore({ dbPath: TEST_DB });
  });

  afterEach(() => {
    store.close();
    rmSync(TEST_DIR, { recursive: true, force: true });
  });

  // ─── Store & Recall ──────────────────────────────────────────

  it("stores and recalls a memory", () => {
    store.store("pattern", "error-handling", "Always use try/catch with specific error types");
    const memory = store.recall("pattern", "error-handling");

    expect(memory).not.toBeNull();
    expect(memory!.category).toBe("pattern");
    expect(memory!.key).toBe("error-handling");
    expect(memory!.content).toBe("Always use try/catch with specific error types");
    expect(memory!.relevance).toBeGreaterThanOrEqual(1.0);
  });

  it("returns null for unknown memory", () => {
    const memory = store.recall("pattern", "nonexistent");
    expect(memory).toBeNull();
  });

  it("updates existing memory with same category+key", () => {
    store.store("decision", "db-choice", "Use SQLite for persistence");
    store.store("decision", "db-choice", "Use SQLite with WAL mode for persistence");

    const memory = store.recall("decision", "db-choice");
    expect(memory).not.toBeNull();
    expect(memory!.content).toBe("Use SQLite with WAL mode for persistence");
    expect(memory!.accessCount).toBeGreaterThanOrEqual(1);
  });

  it("stores with tags and session ID", () => {
    const id = store.store("mistake", "missing-null-check", "Forgot to check null returns", {
      tags: ["typescript", "null-safety"],
      sessionId: "session-123",
    });

    expect(id).toBeTruthy();
    const memory = store.recall("mistake", "missing-null-check");
    expect(memory).not.toBeNull();
    expect(memory!.tags).toEqual(["typescript", "null-safety"]);
    expect(memory!.sessionId).toBe("session-123");
  });

  // ─── Search ──────────────────────────────────────────────────

  it("searches by category", () => {
    store.store("pattern", "p1", "Pattern 1");
    store.store("pattern", "p2", "Pattern 2");
    store.store("decision", "d1", "Decision 1");

    const patterns = store.search({ category: "pattern" });
    expect(patterns).toHaveLength(2);
    expect(patterns.every((m) => m.category === "pattern")).toBe(true);
  });

  it("searches by query text", () => {
    store.store("pattern", "error-handling", "Use try/catch blocks");
    store.store("pattern", "logging", "Use structured logging");
    store.store("decision", "framework", "Use React for UI");

    const results = store.search({ query: "logging" });
    expect(results).toHaveLength(1);
    expect(results[0]!.key).toBe("logging");
  });

  it("searches by tags", () => {
    store.store("pattern", "p1", "Content 1", { tags: ["typescript", "react"] });
    store.store("pattern", "p2", "Content 2", { tags: ["rust", "systems"] });
    store.store("pattern", "p3", "Content 3", { tags: ["typescript", "node"] });

    const results = store.search({ tags: ["typescript"] });
    expect(results).toHaveLength(2);
  });

  it("searches with limit", () => {
    for (let i = 0; i < 10; i++) {
      store.store("context", `ctx-${i}`, `Context ${i}`);
    }

    const results = store.search({ category: "context", limit: 3 });
    expect(results).toHaveLength(3);
  });

  it("searches with minimum relevance", () => {
    store.store("pattern", "high", "High relevance", { relevance: 0.9 });
    store.store("pattern", "low", "Low relevance", { relevance: 0.2 });

    const results = store.search({ minRelevance: 0.5 });
    expect(results).toHaveLength(1);
    expect(results[0]!.key).toBe("high");
  });

  // ─── Recent ──────────────────────────────────────────────────

  it("returns recent memories with limit", () => {
    store.store("pattern", "p1", "First");
    store.store("pattern", "p2", "Second");
    store.store("decision", "d1", "Third");

    const recent = store.recent(2);
    expect(recent).toHaveLength(2);
    // Returns 2 out of 3 (limit applied)
    const allKeys = recent.map((m) => m.key);
    expect(allKeys).toHaveLength(2);
  });

  // ─── Decay & Maintenance ────────────────────────────────────

  it("applies decay to old memories", () => {
    // Store a memory with old timestamp (simulate)
    const id = store.store("pattern", "old-pattern", "Old content");

    // Manually set updated_at to 30 days ago
    const thirtyDaysAgo = Date.now() - 30 * 24 * 60 * 60 * 1000;
    // Access internal db via type assertion for testing
    const storeAny = store as unknown as { db: { prepare: (sql: string) => { run: (...args: unknown[]) => void } } };
    storeAny.db
      .prepare("UPDATE memories SET updated_at = ? WHERE id = ?")
      .run(thirtyDaysAgo, id);

    const decayed = store.applyDecay();
    expect(decayed).toBe(1);

    const memory = store.recall("pattern", "old-pattern");
    expect(memory).not.toBeNull();
    // Relevance should have decreased (30 days * 0.02 = 0.6 decay)
    // Original 1.0 - 0.6 = 0.4, but recall boosts by 0.1, and we read the pre-boost value
    expect(memory!.relevance).toBeLessThan(1.0);
  });

  it("prunes low-relevance memories", () => {
    store.store("pattern", "good", "Good pattern", { relevance: 0.8 });
    store.store("pattern", "bad", "Bad pattern", { relevance: 0.05 });

    const pruned = store.prune(0.1);
    expect(pruned).toBe(1);
    expect(store.size).toBe(1);
  });

  // ─── Delete ──────────────────────────────────────────────────

  it("deletes a memory", () => {
    const id = store.store("preference", "theme", "dark mode");
    expect(store.size).toBe(1);

    const deleted = store.delete(id);
    expect(deleted).toBe(true);
    expect(store.size).toBe(0);
  });

  it("returns false for deleting nonexistent memory", () => {
    const deleted = store.delete("nonexistent-id");
    expect(deleted).toBe(false);
  });

  // ─── Summary ─────────────────────────────────────────────────

  it("returns summary by category", () => {
    store.store("pattern", "p1", "P1");
    store.store("pattern", "p2", "P2");
    store.store("decision", "d1", "D1");
    store.store("mistake", "m1", "M1");

    const summary = store.summary();
    expect(summary.pattern).toBe(2);
    expect(summary.decision).toBe(1);
    expect(summary.mistake).toBe(1);
    expect(summary.preference).toBe(0);
    expect(summary.context).toBe(0);
  });

  // ─── Size ────────────────────────────────────────────────────

  it("tracks total memory count", () => {
    expect(store.size).toBe(0);
    store.store("pattern", "p1", "Content");
    expect(store.size).toBe(1);
    store.store("decision", "d1", "Content");
    expect(store.size).toBe(2);
  });

  // ─── Configurable Parameters ────────────────────────────────

  it("uses custom dailyDecay from constructor", () => {
    // Create store with aggressive decay (0.1/day)
    const customStore = new MemoryStore({ dbPath: join(TEST_DIR, "custom.db"), dailyDecay: 0.1 });
    try {
      const id = customStore.store("pattern", "test", "Test content");

      // Set updated_at to 5 days ago
      const fiveDaysAgo = Date.now() - 5 * 24 * 60 * 60 * 1000;
      const storeAny = customStore as unknown as { db: { prepare: (sql: string) => { run: (...args: unknown[]) => void } } };
      storeAny.db.prepare("UPDATE memories SET updated_at = ? WHERE id = ?").run(fiveDaysAgo, id);

      customStore.applyDecay();

      // Read directly to check without recall boost
      const rows = (customStore as unknown as { db: { prepare: (sql: string) => { get: (...args: unknown[]) => { relevance: number } | null } } })
        .db.prepare("SELECT relevance FROM memories WHERE id = ?").get(id);
      // 1.0 - (5 * 0.1) = 0.5
      expect(rows!.relevance).toBeCloseTo(0.5, 1);
    } finally {
      customStore.close();
    }
  });

  it("uses custom accessBoost on recall", () => {
    const customStore = new MemoryStore({ dbPath: join(TEST_DIR, "boost.db"), accessBoost: 0.2 });
    try {
      customStore.store("pattern", "test", "Content", { relevance: 0.5 });
      customStore.recall("pattern", "test");

      // Read directly to check boosted value
      const rows = (customStore as unknown as { db: { prepare: (sql: string) => { all: () => Array<{ relevance: number }> } } })
        .db.prepare("SELECT relevance FROM memories WHERE key = 'test'").all();
      // 0.5 + 0.2 = 0.7
      expect(rows[0]!.relevance).toBeCloseTo(0.7, 1);
    } finally {
      customStore.close();
    }
  });

  // ─── Deduplication ──────────────────────────────────────────

  it("deduplicates similar memories within same category", () => {
    store.store("mistake", "err-1", "Tool read_file was called repeatedly with identical failing arguments. Try different approaches.", { relevance: 0.9 });
    store.store("mistake", "err-2", "Tool read_file was called repeatedly with identical failing arguments. Try a different approach.", { relevance: 0.5 });

    const merged = store.deduplicate(0.7);
    expect(merged).toBe(1);
    expect(store.size).toBe(1);

    // Higher relevance memory survives
    const remaining = store.search({ category: "mistake" });
    expect(remaining).toHaveLength(1);
    expect(remaining[0]!.key).toBe("err-1");
    // Relevance should be boosted
    expect(remaining[0]!.relevance).toBeGreaterThan(0.9);
  });

  it("does not merge dissimilar memories", () => {
    store.store("pattern", "p1", "Use TypeScript strict mode for all projects");
    store.store("pattern", "p2", "Always prefer functional components in React");

    const merged = store.deduplicate(0.7);
    expect(merged).toBe(0);
    expect(store.size).toBe(2);
  });

  it("does not merge across categories", () => {
    store.store("pattern", "same-key", "Same exact content for testing deduplication");
    store.store("decision", "same-key", "Same exact content for testing deduplication");

    const merged = store.deduplicate(0.7);
    expect(merged).toBe(0);
    expect(store.size).toBe(2);
  });

  it("deduplication keeps newer content when merging", () => {
    const id1 = store.store("mistake", "dup-1", "Old version of the advice");
    const id2 = store.store("mistake", "dup-2", "Old version of the advice updated", { relevance: 0.5 });

    // Make the lower-relevance one newer
    const future = Date.now() + 100_000;
    const storeAny = store as unknown as { db: { prepare: (sql: string) => { run: (...args: unknown[]) => void } } };
    storeAny.db.prepare("UPDATE memories SET updated_at = ? WHERE id = ?").run(future, id2);

    store.deduplicate(0.7);

    const remaining = store.search({ category: "mistake" });
    expect(remaining).toHaveLength(1);
    // Survivor has the newer content
    expect(remaining[0]!.content).toBe("Old version of the advice updated");
  });

  // ─── runMaintenance ─────────────────────────────────────────

  it("runMaintenance runs decay, prune, and dedup together", () => {
    // Old memory (will decay)
    const id1 = store.store("pattern", "old", "Old pattern");
    const thirtyDaysAgo = Date.now() - 30 * 24 * 60 * 60 * 1000;
    const storeAny = store as unknown as { db: { prepare: (sql: string) => { run: (...args: unknown[]) => void } } };
    storeAny.db.prepare("UPDATE memories SET updated_at = ? WHERE id = ?").run(thirtyDaysAgo, id1);

    // Low-relevance memory (will be pruned)
    store.store("mistake", "stale", "Stale mistake", { relevance: 0.05 });

    // Duplicates (will be merged)
    store.store("decision", "dup-a", "Use SQLite for storage persistence and data management", { relevance: 0.8 });
    store.store("decision", "dup-b", "Use SQLite for storage persistence and data management always", { relevance: 0.4 });

    // Good memory (should survive)
    store.store("preference", "theme", "Dark mode preferred");

    expect(store.size).toBe(5);

    const { decayed, pruned, merged } = store.runMaintenance();

    expect(decayed).toBeGreaterThanOrEqual(1); // old pattern decayed
    expect(pruned).toBeGreaterThanOrEqual(1);  // stale mistake pruned
    expect(merged).toBeGreaterThanOrEqual(1);  // duplicates merged

    // Theme preference and surviving dup should remain (plus maybe the decayed one if still above threshold)
    expect(store.size).toBeLessThanOrEqual(3);
  });
});
