/**
 * Session management with SQLite persistence.
 * Stores conversations, messages, and cost records.
 * Uses bun:sqlite (Bun's built-in SQLite).
 * Fail fast: throws on DB errors, never silently drops data.
 */

import { Database } from "./bun-sqlite.js";
import { randomUUID } from "node:crypto";
import { mkdirSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { homedir } from "node:os";
import type { Session, Message, CostRecord } from "./types.js";
import { MessageRole } from "./types.js";
import { SessionError , extractErrorMessage } from "./errors.js";

// ─── Schema ──────────────────────────────────────────────────

const SCHEMA_VERSION = 3;

// DDL constants — referenced in both CREATE_TABLES_STATEMENTS and migrate()
const SESSION_STATE_DDL = `CREATE TABLE IF NOT EXISTS session_state (
    session_id TEXT PRIMARY KEY,
    state_json TEXT NOT NULL,
    updated_at INTEGER NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
  )`;

const COMPACTION_LOG_DDL = `CREATE TABLE IF NOT EXISTS compaction_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    tokens_before INTEGER NOT NULL,
    tokens_after INTEGER NOT NULL,
    removed_count INTEGER NOT NULL,
    tier TEXT NOT NULL DEFAULT 'unknown',
    created_at INTEGER NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
  )`;

const COMPACTION_LOG_INDEX_DDL = `CREATE INDEX IF NOT EXISTS idx_compaction_session
    ON compaction_log(session_id)`;

const CREATE_TABLES_STATEMENTS = [
  `CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}'
  )`,
  `CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT,
    tool_call_id TEXT,
    tool_calls TEXT,
    created_at INTEGER NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
  )`,
  `CREATE INDEX IF NOT EXISTS idx_messages_session
    ON messages(session_id, id)`,
  `CREATE TABLE IF NOT EXISTS cost_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens INTEGER NOT NULL DEFAULT 0,
    cache_write_tokens INTEGER NOT NULL DEFAULT 0,
    total_cost REAL NOT NULL DEFAULT 0.0,
    created_at INTEGER NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
  )`,
  `CREATE INDEX IF NOT EXISTS idx_cost_session
    ON cost_records(session_id)`,
  SESSION_STATE_DDL,
  COMPACTION_LOG_DDL,
  COMPACTION_LOG_INDEX_DDL,
  `CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
  )`,
];

// ─── Session Store ───────────────────────────────────────────

export interface SessionStoreOptions {
  readonly dbPath?: string;
}

export class SessionStore {
  private readonly db: Database;

  constructor(options?: SessionStoreOptions) {
    const dbPath = options?.dbPath ?? getDefaultDbPath();

    // Ensure directory exists
    const dir = dirname(dbPath);
    if (!existsSync(dir)) {
      mkdirSync(dir, { recursive: true });
    }

    try {
      this.db = new Database(dbPath);
    } catch (err) {
      const msg = extractErrorMessage(err);
      throw new SessionError(`Failed to open session database: ${msg}`);
    }

    // Enable WAL mode for better concurrent read performance
    this.db.exec("PRAGMA journal_mode = WAL");
    this.db.exec("PRAGMA foreign_keys = ON");

    this.initSchema();
  }

  private initSchema(): void {
    for (const stmt of CREATE_TABLES_STATEMENTS) {
      this.db.exec(stmt);
    }

    // Check/set schema version
    const row = this.db
      .prepare("SELECT version FROM schema_version LIMIT 1")
      .get() as { version: number } | null;

    if (!row) {
      this.db
        .prepare("INSERT INTO schema_version (version) VALUES (?)")
        .run(SCHEMA_VERSION);
    } else if (row.version < SCHEMA_VERSION) {
      this.migrate(row.version);
    }
  }

  private migrate(fromVersion: number): void {
    // v1 → v2: add session_state table
    if (fromVersion < 2) {
      this.db.exec(SESSION_STATE_DDL);
    }
    // v2 → v3: add compaction_log table
    if (fromVersion < 3) {
      this.db.exec(COMPACTION_LOG_DDL);
      this.db.exec(COMPACTION_LOG_INDEX_DDL);
    }
    // Update stored version
    this.db
      .prepare("UPDATE schema_version SET version = ?")
      .run(SCHEMA_VERSION);
  }

  // ─── Session CRUD ────────────────────────────────────────────

  createSession(metadata?: Record<string, unknown>): Session {
    const id = randomUUID();
    const now = Date.now();
    const metaJson = JSON.stringify(metadata ?? {});

    this.db
      .prepare(
        "INSERT INTO sessions (id, created_at, updated_at, metadata) VALUES (?, ?, ?, ?)",
      )
      .run(id, now, now, metaJson);

    return {
      id,
      createdAt: now,
      updatedAt: now,
      messages: [],
      metadata: metadata ?? {},
    };
  }

  getSession(id: string): Session | null {
    const row = this.db
      .prepare("SELECT * FROM sessions WHERE id = ?")
      .get(id) as SessionRow | null;

    if (!row) return null;

    const messages = this.getMessages(id);
    return rowToSession(row, messages);
  }

  listSessions(limit: number = 50, offset: number = 0): ReadonlyArray<Session> {
    const rows = this.db
      .prepare(
        "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ? OFFSET ?",
      )
      .all(limit, offset) as SessionRow[];

    return rows.map((row) => {
      const messages = this.getMessages(row.id);
      return rowToSession(row, messages);
    });
  }

  deleteSession(id: string): boolean {
    const result = this.db
      .prepare("DELETE FROM sessions WHERE id = ?")
      .run(id);
    return result.changes > 0;
  }

  // ─── Messages ────────────────────────────────────────────────

  addMessage(sessionId: string, message: Message): void {
    const now = Date.now();
    const toolCallsJson = message.toolCalls
      ? JSON.stringify(message.toolCalls)
      : null;

    this.db
      .prepare(
        `INSERT INTO messages (session_id, role, content, tool_call_id, tool_calls, created_at)
         VALUES (?, ?, ?, ?, ?, ?)`,
      )
      .run(
        sessionId,
        message.role,
        message.content,
        message.toolCallId ?? null,
        toolCallsJson,
        now,
      );

    // Update session timestamp
    this.db
      .prepare("UPDATE sessions SET updated_at = ? WHERE id = ?")
      .run(now, sessionId);
  }

  getMessages(sessionId: string): ReadonlyArray<Message> {
    const rows = this.db
      .prepare(
        "SELECT * FROM messages WHERE session_id = ? ORDER BY id ASC",
      )
      .all(sessionId) as MessageRow[];

    return rows.map(rowToMessage);
  }

  getMessageCount(sessionId: string): number {
    const row = this.db
      .prepare("SELECT COUNT(*) as count FROM messages WHERE session_id = ?")
      .get(sessionId) as { count: number };
    return row.count;
  }

  /**
   * Get recent messages for context window management.
   * Always preserves the first message (original task) per design doc Section 2.2.
   */
  getRecentMessages(
    sessionId: string,
    keepRecent: number,
  ): ReadonlyArray<Message> {
    const allMessages = this.getMessages(sessionId);
    if (allMessages.length <= keepRecent) return allMessages;

    // Always keep the first message (original task prompt)
    const first = allMessages[0]!; // safe: length > keepRecent > 0
    const recent = allMessages.slice(-keepRecent);

    // Avoid duplicating first if it's already in recent
    if (first === recent[0]) return recent;
    return [first, ...recent];
  }

  // ─── Cost Tracking ────────────────────────────────────────────

  addCostRecord(sessionId: string, cost: CostRecord): void {
    const now = Date.now();
    this.db
      .prepare(
        `INSERT INTO cost_records
         (session_id, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, total_cost, created_at)
         VALUES (?, ?, ?, ?, ?, ?, ?)`,
      )
      .run(
        sessionId,
        cost.inputTokens,
        cost.outputTokens,
        cost.cacheReadTokens,
        cost.cacheWriteTokens,
        cost.totalCost,
        now,
      );
  }

  getSessionCost(sessionId: string): CostRecord {
    const row = this.db
      .prepare(
        `SELECT
           COALESCE(SUM(input_tokens), 0) as input_tokens,
           COALESCE(SUM(output_tokens), 0) as output_tokens,
           COALESCE(SUM(cache_read_tokens), 0) as cache_read_tokens,
           COALESCE(SUM(cache_write_tokens), 0) as cache_write_tokens,
           COALESCE(SUM(total_cost), 0.0) as total_cost
         FROM cost_records WHERE session_id = ?`,
      )
      .get(sessionId) as CostRow;

    return {
      inputTokens: row.input_tokens,
      outputTokens: row.output_tokens,
      cacheReadTokens: row.cache_read_tokens,
      cacheWriteTokens: row.cache_write_tokens,
      totalCost: row.total_cost,
    };
  }

  // ─── Session State Persistence ──────────────────────────────

  /**
   * Save serialized session state (upsert — creates or overwrites).
   * Accepts any JSON-serializable object; the caller is responsible for
   * type-checking against SessionStateJSON from @devagent/engine.
   */
  saveSessionState(sessionId: string, state: object): void {
    const now = Date.now();
    const json = JSON.stringify(state);
    this.db
      .prepare(
        `INSERT INTO session_state (session_id, state_json, updated_at)
         VALUES (?, ?, ?)
         ON CONFLICT(session_id) DO UPDATE SET state_json = excluded.state_json, updated_at = excluded.updated_at`,
      )
      .run(sessionId, json, now);
  }

  /**
   * Load serialized session state for a session.
   * Returns null if no state has been saved for this session.
   */
  loadSessionState(sessionId: string): Record<string, unknown> | null {
    const row = this.db
      .prepare("SELECT state_json FROM session_state WHERE session_id = ?")
      .get(sessionId) as { state_json: string } | null;
    if (!row) return null;
    return JSON.parse(row.state_json) as Record<string, unknown>;
  }

  // ─── Compaction Log ──────────────────────────────────────────

  /**
   * Persist a compaction event for forensic analysis.
   */
  saveCompactionEvent(
    sessionId: string,
    event: {
      tokensBefore: number;
      tokensAfter: number;
      removedCount: number;
      tier?: string;
    },
  ): void {
    const now = Date.now();
    this.db
      .prepare(
        `INSERT INTO compaction_log
         (session_id, tokens_before, tokens_after, removed_count, tier, created_at)
         VALUES (?, ?, ?, ?, ?, ?)`,
      )
      .run(
        sessionId,
        event.tokensBefore,
        event.tokensAfter,
        event.removedCount,
        event.tier ?? "unknown",
        now,
      );
  }

  /**
   * Retrieve compaction events for a session, ordered chronologically.
   */
  getCompactionLog(
    sessionId: string,
  ): ReadonlyArray<{
    tokensBefore: number;
    tokensAfter: number;
    removedCount: number;
    tier: string;
    createdAt: number;
  }> {
    const rows = this.db
      .prepare(
        "SELECT tokens_before, tokens_after, removed_count, tier, created_at FROM compaction_log WHERE session_id = ? ORDER BY id ASC",
      )
      .all(sessionId) as Array<{
        tokens_before: number;
        tokens_after: number;
        removed_count: number;
        tier: string;
        created_at: number;
      }>;

    return rows.map((r) => ({
      tokensBefore: r.tokens_before,
      tokensAfter: r.tokens_after,
      removedCount: r.removed_count,
      tier: r.tier,
      createdAt: r.created_at,
    }));
  }

  // ─── Lifecycle ────────────────────────────────────────────────

  close(): void {
    this.db.close();
  }
}

// ─── Row Types ──────────────────────────────────────────────

interface SessionRow {
  id: string;
  created_at: number;
  updated_at: number;
  metadata: string;
}

interface MessageRow {
  id: number;
  session_id: string;
  role: string;
  content: string | null;
  tool_call_id: string | null;
  tool_calls: string | null;
  created_at: number;
}

interface CostRow {
  input_tokens: number;
  output_tokens: number;
  cache_read_tokens: number;
  cache_write_tokens: number;
  total_cost: number;
}

// ─── Conversion helpers ──────────────────────────────────────

function rowToSession(
  row: SessionRow,
  messages: ReadonlyArray<Message>,
): Session {
  return {
    id: row.id,
    createdAt: row.created_at,
    updatedAt: row.updated_at,
    messages,
    metadata: JSON.parse(row.metadata) as Record<string, unknown>,
  };
}

function rowToMessage(row: MessageRow): Message {
  const message: Message = {
    role: row.role as MessageRole,
    content: row.content,
  };

  if (row.tool_call_id) {
    return { ...message, toolCallId: row.tool_call_id };
  }

  if (row.tool_calls) {
    return {
      ...message,
      toolCalls: JSON.parse(row.tool_calls) as Message["toolCalls"],
    };
  }

  return message;
}

// ─── Helpers ────────────────────────────────────────────────

function getDefaultDbPath(): string {
  return join(homedir(), ".config", "devagent", "sessions.db");
}
