import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, it, expect, beforeEach, afterEach } from "vitest";

import { BUN_SQLITE_AVAILABLE } from "./bun-sqlite.js";
import { Database } from "./bun-sqlite.js";
import { SessionStore } from "./session.js";
import { MessageRole } from "./types.js";
import type { Message, CostRecord } from "./types.js";
let store: SessionStore;
let tmpDir: string;

beforeEach(() => {
  tmpDir = mkdtempSync(join(tmpdir(), "devagent-test-"));
  store = new SessionStore({ dbPath: join(tmpDir, "test.db") });
});

afterEach(() => {
  store.close();
  rmSync(tmpDir, { recursive: true, force: true });
});

describe.skipIf(!BUN_SQLITE_AVAILABLE)("createSession", () => {
  it("creates a session with a unique id", () => {
    const session = store.createSession();
    expect(session.id).toBeDefined();
    expect(session.id.length).toBeGreaterThan(0);
    expect(session.createdAt).toBeGreaterThan(0);
    expect(session.updatedAt).toBe(session.createdAt);
    expect(session.messages).toEqual([]);
    expect(session.metadata).toEqual({});
  });

  it("stores metadata", () => {
    const meta = { project: "devagent", task: "review" };
    const session = store.createSession(meta);
    expect(session.metadata).toEqual(meta);

    const loaded = store.getSession(session.id);
    expect(loaded).not.toBeNull();
    expect(loaded!.metadata).toEqual(meta);
  });
});

describe.skipIf(!BUN_SQLITE_AVAILABLE)("getSession", () => {
  it("returns null for non-existent session", () => {
    const result = store.getSession("non-existent-id");
    expect(result).toBeNull();
  });

  it("returns session with messages", () => {
    const session = store.createSession();
    const msg: Message = {
      role: MessageRole.USER,
      content: "Hello",
    };
    store.addMessage(session.id, msg);

    const loaded = store.getSession(session.id);
    expect(loaded).not.toBeNull();
    expect(loaded!.messages).toHaveLength(1);
    expect(loaded!.messages[0]!.content).toBe("Hello");
    expect(loaded!.messages[0]!.role).toBe(MessageRole.USER);
  });

  it("updates session metadata without replacing existing fields", () => {
    const session = store.createSession({ query: "test", provider: "chatgpt" });
    const updated = store.updateSessionMetadata(session.id, {
      delegatedWork: {
        childCount: 2,
        lanes: ["docs", "runtime"],
      },
    });

    expect(updated).not.toBeNull();
    expect(updated!.metadata).toMatchObject({
      query: "test",
      provider: "chatgpt",
      delegatedWork: {
        childCount: 2,
        lanes: ["docs", "runtime"],
      },
    });
  });

});

describe.skipIf(!BUN_SQLITE_AVAILABLE)("listSessions", () => {
  it("lists sessions ordered by updated_at desc", () => {
    const s1 = store.createSession({ name: "first" });
    store.createSession({ name: "second" });

    // Add a message to s1 to make it most recent
    store.addMessage(s1.id, {
      role: MessageRole.USER,
      content: "update",
    });

    const sessions = store.listSessions();
    expect(sessions.length).toBe(2);
    expect(sessions[0]!.id).toBe(s1.id); // s1 was updated more recently
  });

  it("respects limit and offset", () => {
    store.createSession({ name: "a" });
    store.createSession({ name: "b" });
    store.createSession({ name: "c" });

    const page1 = store.listSessions(2, 0);
    expect(page1.length).toBe(2);

    const page2 = store.listSessions(2, 2);
    expect(page2.length).toBe(1);
  });
});

describe.skipIf(!BUN_SQLITE_AVAILABLE)("deleteSession", () => {
  it("deletes session and returns true", () => {
    const session = store.createSession();
    const result = store.deleteSession(session.id);
    expect(result).toBe(true);
    expect(store.getSession(session.id)).toBeNull();
  });

  it("returns false for non-existent session", () => {
    const result = store.deleteSession("non-existent");
    expect(result).toBe(false);
  });
});

describe.skipIf(!BUN_SQLITE_AVAILABLE)("addMessage", () => {
  it("adds messages in order", () => {
    const session = store.createSession();

    store.addMessage(session.id, {
      role: MessageRole.USER,
      content: "Question",
    });
    store.addMessage(session.id, {
      role: MessageRole.ASSISTANT,
      content: "Answer",
    });

    const messages = store.getMessages(session.id);
    expect(messages).toHaveLength(2);
    expect(messages[0]!.role).toBe(MessageRole.USER);
    expect(messages[0]!.content).toBe("Question");
    expect(messages[1]!.role).toBe(MessageRole.ASSISTANT);
    expect(messages[1]!.content).toBe("Answer");
  });

  it("stores tool calls", () => {
    const session = store.createSession();
    const msg: Message = {
      role: MessageRole.ASSISTANT,
      content: null,
      toolCalls: [
        {
          name: "read_file",
          arguments: { path: "/src/index.ts" },
          callId: "call_1",
        },
      ],
    };
    store.addMessage(session.id, msg);

    const messages = store.getMessages(session.id);
    expect(messages).toHaveLength(1);
    expect(messages[0]!.toolCalls).toHaveLength(1);
    expect(messages[0]!.toolCalls![0]!.name).toBe("read_file");
  });

  it("stores assistant thinking content", () => {
    const session = store.createSession();
    const msg: Message = {
      role: MessageRole.ASSISTANT,
      content: "I'll check.",
      thinking: "Need a shell command.",
      toolCalls: [
        {
          name: "run_command",
          arguments: { cmd: "pwd" },
          callId: "call_1",
        },
      ],
    };
    store.addMessage(session.id, msg);

    const messages = store.getMessages(session.id);
    expect(messages).toHaveLength(1);
    expect(messages[0]!.thinking).toBe("Need a shell command.");
    expect(messages[0]!.toolCalls).toHaveLength(1);
  });

  it("stores tool result messages", () => {
    const session = store.createSession();
    const msg: Message = {
      role: MessageRole.TOOL,
      content: "file contents here",
      toolCallId: "call_1",
    };
    store.addMessage(session.id, msg);

    const messages = store.getMessages(session.id);
    expect(messages).toHaveLength(1);
    expect(messages[0]!.toolCallId).toBe("call_1");
  });

  it("updates session timestamp", () => {
    const session = store.createSession();
    const originalUpdated = session.updatedAt;

    // Small delay to ensure different timestamp
    const msg: Message = { role: MessageRole.USER, content: "hi" };
    store.addMessage(session.id, msg);

    const loaded = store.getSession(session.id);
    expect(loaded!.updatedAt).toBeGreaterThanOrEqual(originalUpdated);
  });
});

describe.skipIf(!BUN_SQLITE_AVAILABLE)("schema migration", () => {
  it("adds thinking column to existing v3 session databases", () => {
    store.close();
    const dbPath = join(tmpDir, "legacy-v3.db");
    const db = new Database(dbPath);
    db.exec(`
      CREATE TABLE sessions (
        id TEXT PRIMARY KEY,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        metadata TEXT NOT NULL DEFAULT '{}'
      );
      CREATE TABLE messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT,
        tool_call_id TEXT,
        tool_calls TEXT,
        created_at INTEGER NOT NULL,
        FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
      );
      CREATE TABLE cost_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        input_tokens INTEGER NOT NULL DEFAULT 0,
        output_tokens INTEGER NOT NULL DEFAULT 0,
        cache_read_tokens INTEGER NOT NULL DEFAULT 0,
        cache_write_tokens INTEGER NOT NULL DEFAULT 0,
        total_cost REAL NOT NULL DEFAULT 0.0,
        created_at INTEGER NOT NULL,
        FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
      );
      CREATE TABLE session_state (
        session_id TEXT PRIMARY KEY,
        state_json TEXT NOT NULL,
        updated_at INTEGER NOT NULL,
        FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
      );
      CREATE TABLE compaction_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        tokens_before INTEGER NOT NULL,
        tokens_after INTEGER NOT NULL,
        removed_count INTEGER NOT NULL,
        tier TEXT NOT NULL DEFAULT 'unknown',
        created_at INTEGER NOT NULL,
        FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
      );
      CREATE TABLE schema_version (version INTEGER PRIMARY KEY);
      INSERT INTO schema_version (version) VALUES (3);
    `);
    db.close();

    const migrated = new SessionStore({ dbPath });
    store = migrated;
    const session = migrated.createSession();
    migrated.addMessage(session.id, {
      role: MessageRole.ASSISTANT,
      content: "done",
      thinking: "private reasoning",
    });

    expect(migrated.getMessages(session.id)[0]!.thinking).toBe("private reasoning");
  });
});

describe.skipIf(!BUN_SQLITE_AVAILABLE)("getMessageCount", () => {
  it("returns 0 for empty session", () => {
    const session = store.createSession();
    expect(store.getMessageCount(session.id)).toBe(0);
  });

  it("returns correct count", () => {
    const session = store.createSession();
    store.addMessage(session.id, {
      role: MessageRole.USER,
      content: "one",
    });
    store.addMessage(session.id, {
      role: MessageRole.ASSISTANT,
      content: "two",
    });
    expect(store.getMessageCount(session.id)).toBe(2);
  });
});

describe.skipIf(!BUN_SQLITE_AVAILABLE)("getRecentMessages", () => {
  it("returns all messages when count <= keepRecent", () => {
    const session = store.createSession();
    store.addMessage(session.id, {
      role: MessageRole.USER,
      content: "first",
    });
    store.addMessage(session.id, {
      role: MessageRole.ASSISTANT,
      content: "second",
    });

    const recent = store.getRecentMessages(session.id, 5);
    expect(recent).toHaveLength(2);
  });

  it("preserves first message and returns recent ones", () => {
    const session = store.createSession();

    // Add 5 messages
    for (let i = 0; i < 5; i++) {
      store.addMessage(session.id, {
        role: MessageRole.USER,
        content: `msg-${i}`,
      });
    }

    const recent = store.getRecentMessages(session.id, 2);
    // Should have first message + 2 recent = 3
    expect(recent).toHaveLength(3);
    expect(recent[0]!.content).toBe("msg-0"); // original task preserved
    expect(recent[1]!.content).toBe("msg-3");
    expect(recent[2]!.content).toBe("msg-4");
  });
});

describe.skipIf(!BUN_SQLITE_AVAILABLE)("cost tracking", () => {
  it("adds and aggregates cost records", () => {
    const session = store.createSession();

    const cost1: CostRecord = {
      inputTokens: 100,
      outputTokens: 50,
      cacheReadTokens: 10,
      cacheWriteTokens: 5,
      totalCost: 0.01,
    };
    const cost2: CostRecord = {
      inputTokens: 200,
      outputTokens: 100,
      cacheReadTokens: 20,
      cacheWriteTokens: 10,
      totalCost: 0.02,
    };

    store.addCostRecord(session.id, cost1);
    store.addCostRecord(session.id, cost2);

    const total = store.getSessionCost(session.id);
    expect(total.inputTokens).toBe(300);
    expect(total.outputTokens).toBe(150);
    expect(total.cacheReadTokens).toBe(30);
    expect(total.cacheWriteTokens).toBe(15);
    expect(total.totalCost).toBeCloseTo(0.03);
  });

  it("returns zeroes for session with no cost records", () => {
    const session = store.createSession();
    const cost = store.getSessionCost(session.id);
    expect(cost.inputTokens).toBe(0);
    expect(cost.outputTokens).toBe(0);
    expect(cost.totalCost).toBe(0);
  });
});

describe.skipIf(!BUN_SQLITE_AVAILABLE)("cascade delete", () => {
  it("deletes messages and cost records when session is deleted", () => {
    const session = store.createSession();
    store.addMessage(session.id, {
      role: MessageRole.USER,
      content: "test",
    });
    store.addCostRecord(session.id, {
      inputTokens: 100,
      outputTokens: 50,
      cacheReadTokens: 0,
      cacheWriteTokens: 0,
      totalCost: 0.01,
    });

    store.deleteSession(session.id);

    // Messages and costs should be gone
    expect(store.getMessages(session.id)).toHaveLength(0);
    expect(store.getSessionCost(session.id).inputTokens).toBe(0);
  });
});

// ─── Compaction Log ──────────────────────────────────────────

describe.skipIf(!BUN_SQLITE_AVAILABLE)("compaction log", () => {
  it("saves and retrieves compaction events", () => {
    const session = store.createSession();

    store.saveCompactionEvent(session.id, {
      tokensBefore: 90000,
      tokensAfter: 45000,
      removedCount: 12,
      tier: "full",
    });

    store.saveCompactionEvent(session.id, {
      tokensBefore: 80000,
      tokensAfter: 40000,
      removedCount: 8,
    });

    const log = store.getCompactionLog(session.id);
    expect(log).toHaveLength(2);
    expect(log[0]!.tokensBefore).toBe(90000);
    expect(log[0]!.tokensAfter).toBe(45000);
    expect(log[0]!.removedCount).toBe(12);
    expect(log[0]!.tier).toBe("full");
    expect(log[1]!.tier).toBe("unknown");
    expect(log[1]!.createdAt).toBeGreaterThan(0);
  });

  it("returns empty array for session with no compaction events", () => {
    const session = store.createSession();
    expect(store.getCompactionLog(session.id)).toEqual([]);
  });

  it("cascade deletes compaction log when session is deleted", () => {
    const session = store.createSession();
    store.saveCompactionEvent(session.id, {
      tokensBefore: 90000,
      tokensAfter: 45000,
      removedCount: 12,
    });
    store.deleteSession(session.id);
    expect(store.getCompactionLog(session.id)).toEqual([]);
  });
});

// ─── Session State Persistence ──────────────────────────────

describe.skipIf(!BUN_SQLITE_AVAILABLE)("session state persistence", () => {
  it("saveSessionState + loadSessionState round-trips", () => {
    const session = store.createSession({ query: "test" });

    const stateData = {
      version: 1,
      plan: [{ description: "Step 1", status: "completed" }],
      modifiedFiles: ["/src/a.ts"],
      envFacts: [{ key: "cmd-not-found:rg", value: "rg not installed" }],
      toolSummaries: [],
    };

    store.saveSessionState(session.id, stateData);
    const loaded = store.loadSessionState(session.id);
    expect(loaded).toEqual(stateData);
  });

  it("loadSessionState returns null for nonexistent session", () => {
    expect(store.loadSessionState("nonexistent")).toBeNull();
  });

  it("saveSessionState overwrites previous state (upsert)", () => {
    const session = store.createSession({ query: "test" });

    store.saveSessionState(session.id, {
      version: 1,
      plan: null,
      modifiedFiles: [],
      envFacts: [],
      toolSummaries: [],
    });
    store.saveSessionState(session.id, {
      version: 1,
      plan: [{ description: "Updated", status: "in_progress" }],
      modifiedFiles: ["/b.ts"],
      envFacts: [],
      toolSummaries: [],
    });

    const loaded = store.loadSessionState(session.id);
    expect(loaded).not.toBeNull();
    expect((loaded as Record<string, unknown>).plan).toEqual([
      { description: "Updated", status: "in_progress" },
    ]);
  });

  it("saveSessionState updates the parent session timestamp", async () => {
    const session = store.createSession({ query: "test" });
    const before = store.getSession(session.id);

    await new Promise((resolve) => setTimeout(resolve, 5));
    store.saveSessionState(session.id, {
      version: 1,
      plan: null,
      modifiedFiles: [],
      envFacts: [],
      toolSummaries: [],
    });

    const after = store.getSession(session.id);
    expect(before).not.toBeNull();
    expect(after).not.toBeNull();
    expect(after!.updatedAt).toBeGreaterThan(before!.updatedAt);
  });

  it("listSessions reflects recency changes from state-only updates", async () => {
    const first = store.createSession({ query: "first" });
    await new Promise((resolve) => setTimeout(resolve, 5));
    const second = store.createSession({ query: "second" });

    expect(store.listSessions()[0]!.id).toBe(second.id);

    await new Promise((resolve) => setTimeout(resolve, 5));
    store.saveSessionState(first.id, {
      version: 1,
      plan: [{ description: "Keep going", status: "in_progress" }],
      modifiedFiles: [],
      envFacts: [],
      toolSummaries: [],
    });

    expect(store.listSessions()[0]!.id).toBe(first.id);
  });

  it("deleteSession cascades to session_state", () => {
    const session = store.createSession({ query: "test" });
    store.saveSessionState(session.id, {
      version: 1,
      plan: null,
      modifiedFiles: [],
      envFacts: [],
      toolSummaries: [],
    });
    store.deleteSession(session.id);
    expect(store.loadSessionState(session.id)).toBeNull();
  });
});
