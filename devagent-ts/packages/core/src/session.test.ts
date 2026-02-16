import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { join } from "node:path";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { SessionStore } from "./session.js";
import { MessageRole } from "./types.js";
import type { Message, CostRecord } from "./types.js";

describe("SessionStore", () => {
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

  describe("createSession", () => {
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

  describe("getSession", () => {
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
  });

  describe("listSessions", () => {
    it("lists sessions ordered by updated_at desc", () => {
      const s1 = store.createSession({ name: "first" });
      const s2 = store.createSession({ name: "second" });

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

  describe("deleteSession", () => {
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

  describe("addMessage", () => {
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

  describe("getMessageCount", () => {
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

  describe("getRecentMessages", () => {
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

  describe("cost tracking", () => {
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

  describe("cascade delete", () => {
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
});
