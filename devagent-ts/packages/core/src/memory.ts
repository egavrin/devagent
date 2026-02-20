/**
 * Cross-session memory system.
 * Persists learnings, decisions, and patterns across sessions.
 * Uses bun:sqlite — same DB infrastructure as SessionStore.
 *
 * Design:
 * - Memories are key-value pairs with metadata (category, relevance, timestamps)
 * - Categories: "pattern", "decision", "mistake", "preference", "context"
 * - Memories decay over time (relevance score) unless reinforced
 * - Search by category, keyword, or recency
 * - Agent can store & recall learnings across sessions
 */

import { Database } from "bun:sqlite";
import { randomUUID } from "node:crypto";
import { mkdirSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { homedir } from "node:os";

// ─── Types ──────────────────────────────────────────────────

export type MemoryCategory =
  | "pattern"
  | "decision"
  | "mistake"
  | "preference"
  | "context";

export interface Memory {
  readonly id: string;
  readonly category: MemoryCategory;
  readonly key: string;
  readonly content: string;
  readonly relevance: number; // 0.0 - 1.0
  readonly tags: ReadonlyArray<string>;
  readonly createdAt: number;
  readonly updatedAt: number;
  readonly accessCount: number;
  readonly sessionId: string | null;
}

export interface MemoryStoreOptions {
  readonly dbPath?: string;
  readonly dailyDecay?: number;
  readonly minRelevance?: number;
  readonly accessBoost?: number;
}

export interface MemorySearchOptions {
  readonly category?: MemoryCategory;
  readonly tags?: ReadonlyArray<string>;
  readonly minRelevance?: number;
  readonly limit?: number;
  readonly query?: string;
}

// ─── Schema ──────────────────────────────────────────────────

const MEMORY_SCHEMA = [
  `CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    key TEXT NOT NULL,
    content TEXT NOT NULL,
    relevance REAL NOT NULL DEFAULT 1.0,
    tags TEXT NOT NULL DEFAULT '[]',
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0,
    session_id TEXT
  )`,
  `CREATE INDEX IF NOT EXISTS idx_memories_category
    ON memories(category)`,
  `CREATE INDEX IF NOT EXISTS idx_memories_key
    ON memories(key)`,
  `CREATE INDEX IF NOT EXISTS idx_memories_relevance
    ON memories(relevance DESC)`,
  `CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_category_key
    ON memories(category, key)`,
  `CREATE TABLE IF NOT EXISTS memory_schema_version (
    version INTEGER PRIMARY KEY
  )`,
];

const MEMORY_SCHEMA_VERSION = 1;

// ─── Decay Config ────────────────────────────────────────────

/** How much relevance decays per day without access. */
const DAILY_DECAY = 0.02;

/** Minimum relevance before a memory is considered stale. */
const MIN_RELEVANCE = 0.1;

/** Relevance boost when a memory is accessed. */
const ACCESS_BOOST = 0.1;

// ─── Memory Store ────────────────────────────────────────────

export class MemoryStore {
  private readonly db: Database;
  private readonly dailyDecay: number;
  private readonly minRelevance: number;
  private readonly accessBoost: number;

  constructor(options?: MemoryStoreOptions) {
    const dbPath = options?.dbPath ?? getDefaultMemoryDbPath();

    this.dailyDecay = options?.dailyDecay ?? DAILY_DECAY;
    this.minRelevance = options?.minRelevance ?? MIN_RELEVANCE;
    this.accessBoost = options?.accessBoost ?? ACCESS_BOOST;

    const dir = dirname(dbPath);
    if (!existsSync(dir)) {
      mkdirSync(dir, { recursive: true });
    }

    this.db = new Database(dbPath);
    this.db.exec("PRAGMA journal_mode = WAL");
    this.db.exec("PRAGMA foreign_keys = ON");
    this.initSchema();
  }

  private initSchema(): void {
    for (const stmt of MEMORY_SCHEMA) {
      this.db.exec(stmt);
    }

    const row = this.db
      .prepare("SELECT version FROM memory_schema_version LIMIT 1")
      .get() as { version: number } | null;

    if (!row) {
      this.db
        .prepare("INSERT INTO memory_schema_version (version) VALUES (?)")
        .run(MEMORY_SCHEMA_VERSION);
    }
  }

  // ─── Store / Update ──────────────────────────────────────────

  /**
   * Store a new memory or update existing one with the same category+key.
   * Returns the memory ID.
   */
  store(
    category: MemoryCategory,
    key: string,
    content: string,
    options?: {
      tags?: ReadonlyArray<string>;
      relevance?: number;
      sessionId?: string;
    },
  ): string {
    const now = Date.now();
    const tags = options?.tags ?? [];
    const relevance = options?.relevance ?? 1.0;
    const sessionId = options?.sessionId ?? null;

    // Try to update existing memory with same category+key
    const existing = this.db
      .prepare("SELECT id, access_count FROM memories WHERE category = ? AND key = ?")
      .get(category, key) as { id: string; access_count: number } | null;

    if (existing) {
      this.db
        .prepare(
          `UPDATE memories SET content = ?, relevance = ?, tags = ?,
           updated_at = ?, access_count = ?, session_id = ?
           WHERE id = ?`,
        )
        .run(
          content,
          Math.min(1.0, relevance),
          JSON.stringify(tags),
          now,
          existing.access_count + 1,
          sessionId,
          existing.id,
        );
      return existing.id;
    }

    // Create new memory
    const id = randomUUID();
    this.db
      .prepare(
        `INSERT INTO memories (id, category, key, content, relevance, tags,
         created_at, updated_at, access_count, session_id)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?)`,
      )
      .run(id, category, key, content, Math.min(1.0, relevance), JSON.stringify(tags), now, now, sessionId);

    return id;
  }

  // ─── Recall / Search ─────────────────────────────────────────

  /**
   * Recall a specific memory by category and key.
   * Boosts relevance on access.
   */
  recall(category: MemoryCategory, key: string): Memory | null {
    const row = this.db
      .prepare("SELECT * FROM memories WHERE category = ? AND key = ?")
      .get(category, key) as MemoryRow | null;

    if (!row) return null;

    // Boost relevance on access
    this.db
      .prepare(
        `UPDATE memories SET access_count = access_count + 1,
         relevance = MIN(1.0, relevance + ?), updated_at = ?
         WHERE id = ?`,
      )
      .run(this.accessBoost, Date.now(), row.id);

    return rowToMemory(row);
  }

  /**
   * Search memories with filters.
   */
  search(options?: MemorySearchOptions): ReadonlyArray<Memory> {
    const conditions: string[] = [];
    const params: (string | number)[] = [];

    if (options?.category) {
      conditions.push("category = ?");
      params.push(options.category);
    }

    if (options?.minRelevance !== undefined) {
      conditions.push("relevance >= ?");
      params.push(options.minRelevance);
    }

    if (options?.query) {
      conditions.push("(key LIKE ? OR content LIKE ?)");
      const q = `%${options.query}%`;
      params.push(q, q);
    }

    if (options?.tags && options.tags.length > 0) {
      // Match any tag
      const tagConditions = options.tags.map(() => "tags LIKE ?");
      conditions.push(`(${tagConditions.join(" OR ")})`);
      for (const tag of options.tags) {
        params.push(`%"${tag}"%`);
      }
    }

    const where = conditions.length > 0 ? `WHERE ${conditions.join(" AND ")}` : "";
    const limit = options?.limit ?? 50;

    const rows = this.db
      .prepare(
        `SELECT * FROM memories ${where}
         ORDER BY relevance DESC, updated_at DESC
         LIMIT ?`,
      )
      .all(...params, limit) as MemoryRow[];

    return rows.map(rowToMemory);
  }

  /**
   * Get recent memories across all categories.
   */
  recent(limit: number = 20): ReadonlyArray<Memory> {
    const rows = this.db
      .prepare("SELECT * FROM memories ORDER BY updated_at DESC LIMIT ?")
      .all(limit) as MemoryRow[];

    return rows.map(rowToMemory);
  }

  // ─── Maintenance ─────────────────────────────────────────────

  /**
   * Apply time-based decay to all memories.
   * Call periodically (e.g., once per session start).
   */
  applyDecay(): number {
    const now = Date.now();

    // Calculate days since last update for each memory
    const rows = this.db
      .prepare("SELECT id, updated_at, relevance FROM memories WHERE relevance > ?")
      .all(this.minRelevance) as Array<{ id: string; updated_at: number; relevance: number }>;

    let decayed = 0;
    for (const row of rows) {
      const daysSinceUpdate = (now - row.updated_at) / (1000 * 60 * 60 * 24);
      const decay = daysSinceUpdate * this.dailyDecay;
      const newRelevance = Math.max(this.minRelevance, row.relevance - decay);

      if (newRelevance < row.relevance) {
        this.db
          .prepare("UPDATE memories SET relevance = ? WHERE id = ?")
          .run(newRelevance, row.id);
        decayed++;
      }
    }

    return decayed;
  }

  /**
   * Remove memories below minimum relevance threshold.
   */
  prune(threshold?: number): number {
    const minRelevance = threshold ?? this.minRelevance;
    const result = this.db
      .prepare("DELETE FROM memories WHERE relevance < ?")
      .run(minRelevance);
    return result.changes;
  }

  /**
   * Delete a specific memory.
   */
  delete(id: string): boolean {
    const result = this.db
      .prepare("DELETE FROM memories WHERE id = ?")
      .run(id);
    return result.changes > 0;
  }

  /**
   * Get total memory count.
   */
  get size(): number {
    const row = this.db
      .prepare("SELECT COUNT(*) as count FROM memories")
      .get() as { count: number };
    return row.count;
  }

  /**
   * Get a summary of memories by category.
   */
  summary(): Record<MemoryCategory, number> {
    const rows = this.db
      .prepare("SELECT category, COUNT(*) as count FROM memories GROUP BY category")
      .all() as Array<{ category: string; count: number }>;

    const result: Record<string, number> = {
      pattern: 0,
      decision: 0,
      mistake: 0,
      preference: 0,
      context: 0,
    };

    for (const row of rows) {
      result[row.category] = row.count;
    }

    return result as Record<MemoryCategory, number>;
  }

  // ─── Deduplication ─────────────────────────────────────────

  /**
   * Deduplicate memories within the same category based on content similarity.
   * Uses Jaccard similarity on word tokens. Merges duplicates: keeps the
   * higher-relevance memory, updates its content if the other is newer,
   * and deletes the lower-relevance duplicate.
   *
   * @param similarityThreshold - Jaccard coefficient threshold (0.0-1.0). Default 0.7.
   * @returns Number of memories merged/removed.
   */
  deduplicate(similarityThreshold: number = 0.7): number {
    const categories: MemoryCategory[] = [
      "pattern", "decision", "mistake", "preference", "context",
    ];
    let mergedCount = 0;

    for (const category of categories) {
      const rows = this.db
        .prepare(
          "SELECT id, key, content, relevance, updated_at FROM memories WHERE category = ? ORDER BY relevance DESC",
        )
        .all(category) as Array<{
          id: string; key: string; content: string;
          relevance: number; updated_at: number;
        }>;

      const removed = new Set<string>();

      for (let i = 0; i < rows.length; i++) {
        if (removed.has(rows[i]!.id)) continue;

        for (let j = i + 1; j < rows.length; j++) {
          if (removed.has(rows[j]!.id)) continue;

          const sim = jaccardSimilarity(rows[i]!.content, rows[j]!.content);
          if (sim >= similarityThreshold) {
            // Keep the higher-relevance one (rows[i], since sorted DESC).
            // If the lower one is newer, update the survivor's content.
            if (rows[j]!.updated_at > rows[i]!.updated_at) {
              this.db
                .prepare("UPDATE memories SET content = ?, updated_at = ? WHERE id = ?")
                .run(rows[j]!.content, rows[j]!.updated_at, rows[i]!.id);
            }
            // Boost the survivor's relevance slightly
            this.db
              .prepare("UPDATE memories SET relevance = MIN(1.0, relevance + 0.05) WHERE id = ?")
              .run(rows[i]!.id);
            // Delete the duplicate
            this.db.prepare("DELETE FROM memories WHERE id = ?").run(rows[j]!.id);
            removed.add(rows[j]!.id);
            mergedCount++;
          }
        }
      }
    }

    return mergedCount;
  }

  // ─── Maintenance ──────────────────────────────────────────

  /**
   * Run startup maintenance: apply decay, prune stale memories, deduplicate.
   * Call once at process start, not per-turn.
   */
  runMaintenance(): { decayed: number; pruned: number; merged: number } {
    const decayed = this.applyDecay();
    const pruned = this.prune();
    const merged = this.deduplicate();
    return { decayed, pruned, merged };
  }

  // ─── Lifecycle ───────────────────────────────────────────────

  close(): void {
    this.db.close();
  }
}

// ─── Row Types ──────────────────────────────────────────────

interface MemoryRow {
  id: string;
  category: string;
  key: string;
  content: string;
  relevance: number;
  tags: string;
  created_at: number;
  updated_at: number;
  access_count: number;
  session_id: string | null;
}

function rowToMemory(row: MemoryRow): Memory {
  return {
    id: row.id,
    category: row.category as MemoryCategory,
    key: row.key,
    content: row.content,
    relevance: row.relevance,
    tags: JSON.parse(row.tags) as string[],
    createdAt: row.created_at,
    updatedAt: row.updated_at,
    accessCount: row.access_count,
    sessionId: row.session_id,
  };
}

/**
 * Jaccard similarity on whitespace-separated word tokens.
 * Returns 0.0 (no overlap) to 1.0 (identical token sets).
 */
function jaccardSimilarity(a: string, b: string): number {
  const tokensA = new Set(a.toLowerCase().split(/\s+/).filter(Boolean));
  const tokensB = new Set(b.toLowerCase().split(/\s+/).filter(Boolean));
  if (tokensA.size === 0 && tokensB.size === 0) return 1.0;
  if (tokensA.size === 0 || tokensB.size === 0) return 0.0;

  let intersection = 0;
  for (const t of tokensA) {
    if (tokensB.has(t)) intersection++;
  }
  const union = tokensA.size + tokensB.size - intersection;
  return union === 0 ? 0 : intersection / union;
}

function getDefaultMemoryDbPath(): string {
  return join(homedir(), ".config", "devagent", "memory.db");
}
