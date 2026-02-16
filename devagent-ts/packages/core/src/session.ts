/**
 * Session management with SQLite persistence.
 * Stores conversations, messages, and cost records.
 * Uses bun:sqlite (Bun's built-in SQLite).
 * Fail fast: throws on DB errors, never silently drops data.
 */

import { Database } from "bun:sqlite";
import { randomUUID } from "node:crypto";
import { mkdirSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { homedir } from "node:os";
import type { Session, Message, CostRecord } from "./types.js";
import { MessageRole } from "./types.js";
import { SessionError } from "./errors.js";

// ─── Schema ──────────────────────────────────────────────────

const SCHEMA_VERSION = 1;

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
      const msg = err instanceof Error ? err.message : String(err);
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
    }
    // Future: handle migrations when SCHEMA_VERSION > row.version
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
