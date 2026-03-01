/**
 * Tests for EventLogger — JSONL event persistence.
 */

import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { mkdtempSync, rmSync, readFileSync, existsSync, writeFileSync, mkdirSync, utimesSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { EventLogger } from "./event-logger.js";
import type { LogEntry } from "./event-logger.js";
import { EventBus } from "./events.js";

let testDir: string;

beforeEach(() => {
  testDir = mkdtempSync(join(tmpdir(), "event-logger-test-"));
});

afterEach(() => {
  rmSync(testDir, { recursive: true, force: true });
});

describe("EventLogger", () => {
  it("creates log directory and file on construction", () => {
    const logger = new EventLogger("sess_001", testDir);
    logger.close();
    expect(existsSync(testDir)).toBe(true);
    expect(existsSync(join(testDir, "sess_001.jsonl"))).toBe(true);
  });

  it("write() produces valid JSONL lines", () => {
    const logger = new EventLogger("sess_002", testDir);
    const entry: LogEntry = {
      ts: 1700000000000,
      event: "tool:before",
      sessionId: "sess_002",
      data: { name: "read_file", params: { path: "foo.ts" } },
    };
    logger.write(entry);
    logger.write({ ...entry, ts: 1700000001000, event: "tool:after" });
    logger.close();

    const content = readFileSync(join(testDir, "sess_002.jsonl"), "utf-8");
    const lines = content.trim().split("\n");
    expect(lines.length).toBe(2);

    const parsed0 = JSON.parse(lines[0]!) as LogEntry;
    expect(parsed0.event).toBe("tool:before");
    expect(parsed0.ts).toBe(1700000000000);

    const parsed1 = JSON.parse(lines[1]!) as LogEntry;
    expect(parsed1.event).toBe("tool:after");
  });

  it("attach() captures events from real EventBus", () => {
    const bus = new EventBus();
    const logger = new EventLogger("sess_003", testDir);
    logger.attach(bus);

    bus.emit("session:start", { sessionId: "sess_003" });
    bus.emit("tool:before", { name: "read_file", params: { path: "bar.ts" }, callId: "c1" });
    bus.emit("tool:after", {
      name: "read_file",
      result: { success: true, output: "contents", error: null, artifacts: [] },
      callId: "c1",
      durationMs: 50,
    });

    logger.close();

    const entries = EventLogger.readLog("sess_003", testDir);
    expect(entries.length).toBe(3);
    expect(entries[0]!.event).toBe("session:start");
    expect(entries[1]!.event).toBe("tool:before");
    expect(entries[2]!.event).toBe("tool:after");
  });

  it("readLog() round-trips entries", () => {
    const logger = new EventLogger("sess_004", testDir);
    const entry: LogEntry = {
      ts: Date.now(),
      event: "cost:update",
      sessionId: "sess_004",
      data: { inputTokens: 100, outputTokens: 50, totalCost: 0.01, model: "test" },
    };
    logger.write(entry);
    logger.close();

    const entries = EventLogger.readLog("sess_004", testDir);
    expect(entries.length).toBe(1);
    expect(entries[0]!.event).toBe("cost:update");
    expect((entries[0]!.data as Record<string, unknown>)["inputTokens"]).toBe(100);
  });

  it("readLog() returns empty array for missing log", () => {
    const entries = EventLogger.readLog("nonexistent", testDir);
    expect(entries).toEqual([]);
  });

  it("listLogs() discovers log files", () => {
    // Create two log files
    const logger1 = new EventLogger("sess_a", testDir);
    logger1.write({ ts: 1000, event: "session:start", sessionId: "sess_a", data: {} });
    logger1.close();

    const logger2 = new EventLogger("sess_b", testDir);
    logger2.write({ ts: 2000, event: "session:start", sessionId: "sess_b", data: {} });
    logger2.close();

    const logs = EventLogger.listLogs(testDir);
    expect(logs.length).toBe(2);

    const ids = logs.map((l) => l.sessionId);
    expect(ids).toContain("sess_a");
    expect(ids).toContain("sess_b");

    for (const log of logs) {
      expect(log.sizeBytes).toBeGreaterThan(0);
      expect(log.createdAt).toBeGreaterThan(0);
    }
  });

  it("listLogs() returns empty for missing directory", () => {
    const logs = EventLogger.listLogs(join(testDir, "nonexistent"));
    expect(logs).toEqual([]);
  });

  it("close() stops writing", () => {
    const bus = new EventBus();
    const logger = new EventLogger("sess_close", testDir);
    logger.attach(bus);

    bus.emit("session:start", { sessionId: "sess_close" });
    logger.close();

    // Events after close should not be written
    bus.emit("session:end", { sessionId: "sess_close", reason: "completed" });

    const entries = EventLogger.readLog("sess_close", testDir);
    expect(entries.length).toBe(1);
    expect(entries[0]!.event).toBe("session:start");
  });

  // ─── Log Rotation ──────────────────────────────────────────

  it("rotate() deletes old files and preserves recent", () => {
    // Create two log files
    const logger1 = new EventLogger("old_sess", testDir);
    logger1.write({ ts: 1000, event: "session:start", sessionId: "old_sess", data: {} });
    logger1.close();

    const logger2 = new EventLogger("new_sess", testDir);
    logger2.write({ ts: 2000, event: "session:start", sessionId: "new_sess", data: {} });
    logger2.close();

    // Make old_sess file appear old by modifying its mtime
    const oldPath = join(testDir, "old_sess.jsonl");
    const pastDate = new Date(Date.now() - 60 * 24 * 60 * 60 * 1000); // 60 days ago
    utimesSync(oldPath, pastDate, pastDate);

    const deleted = EventLogger.rotate(30, testDir);
    expect(deleted).toBe(1);

    // old_sess should be gone
    expect(existsSync(join(testDir, "old_sess.jsonl"))).toBe(false);
    // new_sess should still be there
    expect(existsSync(join(testDir, "new_sess.jsonl"))).toBe(true);
  });

  it("rotate() handles empty/missing directory", () => {
    const deleted = EventLogger.rotate(30, join(testDir, "nonexistent"));
    expect(deleted).toBe(0);
  });

  it("getTotalSize() returns correct total", () => {
    const logger1 = new EventLogger("size_a", testDir);
    logger1.write({ ts: 1000, event: "test", sessionId: "size_a", data: { x: 1 } });
    logger1.close();

    const logger2 = new EventLogger("size_b", testDir);
    logger2.write({ ts: 2000, event: "test", sessionId: "size_b", data: { y: 2 } });
    logger2.close();

    const total = EventLogger.getTotalSize(testDir);
    expect(total).toBeGreaterThan(0);
    // Should be roughly the size of two JSONL lines
    expect(total).toBeLessThan(10000);
  });

  it("getTotalSize() returns 0 for missing directory", () => {
    expect(EventLogger.getTotalSize(join(testDir, "nonexistent"))).toBe(0);
  });
});
