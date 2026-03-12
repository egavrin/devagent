import { describe, it, expect, afterEach } from "vitest";
import { mkdtempSync, rmSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { EventLogger } from "./event-logger.js";
import { EventBus } from "./events.js";

describe("EventLogger", () => {
  const tempDirs: string[] = [];

  function makeTempDir(): string {
    const dir = mkdtempSync(join(tmpdir(), "event-logger-test-"));
    tempDirs.push(dir);
    return dir;
  }

  afterEach(() => {
    for (const dir of tempDirs) {
      rmSync(dir, { recursive: true, force: true });
    }
    tempDirs.length = 0;
  });

  it("creates a JSONL file named after the session ID", () => {
    const dir = makeTempDir();
    const logger = new EventLogger("session-abc", dir);
    logger.close();

    const filePath = join(dir, "session-abc.jsonl");
    const content = readFileSync(filePath, "utf-8");
    expect(content).toBeDefined();
  });

  it("writes events as JSONL lines", () => {
    const dir = makeTempDir();
    const logger = new EventLogger("sess-1", dir);

    logger.write({
      ts: 1000,
      event: "test:event",
      sessionId: "sess-1",
      data: { key: "value1" },
    });

    logger.write({
      ts: 2000,
      event: "test:event",
      sessionId: "sess-1",
      data: { key: "value2" },
    });

    logger.close();

    const content = readFileSync(join(dir, "sess-1.jsonl"), "utf-8").trim();
    const lines = content.split("\n");
    expect(lines).toHaveLength(2);

    const entry1 = JSON.parse(lines[0]);
    expect(entry1.ts).toBe(1000);
    expect(entry1.event).toBe("test:event");
    expect(entry1.data).toEqual({ key: "value1" });

    const entry2 = JSON.parse(lines[1]);
    expect(entry2.ts).toBe(2000);
    expect(entry2.data).toEqual({ key: "value2" });
  });

  it("multiple events produce multiple lines", () => {
    const dir = makeTempDir();
    const logger = new EventLogger("sess-multi", dir);

    for (let i = 0; i < 5; i++) {
      logger.write({
        ts: i * 1000,
        event: "iteration",
        sessionId: "sess-multi",
        data: { i },
      });
    }

    logger.close();

    const content = readFileSync(join(dir, "sess-multi.jsonl"), "utf-8").trim();
    const lines = content.split("\n");
    expect(lines).toHaveLength(5);
  });

  it("stops writing after close()", () => {
    const dir = makeTempDir();
    const logger = new EventLogger("sess-close", dir);

    logger.write({
      ts: 1000,
      event: "before",
      sessionId: "sess-close",
      data: null,
    });

    logger.close();

    logger.write({
      ts: 2000,
      event: "after",
      sessionId: "sess-close",
      data: null,
    });

    const content = readFileSync(join(dir, "sess-close.jsonl"), "utf-8").trim();
    const lines = content.split("\n");
    expect(lines).toHaveLength(1);
    expect(JSON.parse(lines[0]).event).toBe("before");
  });

  it("readLog returns parsed entries", () => {
    const dir = makeTempDir();
    const logger = new EventLogger("sess-read", dir);

    logger.write({
      ts: 100,
      event: "test",
      sessionId: "sess-read",
      data: "hello",
    });
    logger.close();

    const entries = EventLogger.readLog("sess-read", dir);
    expect(entries).toHaveLength(1);
    expect(entries[0].data).toBe("hello");
  });

  it("attach() subscribes to EventBus events and writes them", () => {
    const dir = makeTempDir();
    const logger = new EventLogger("sess-bus", dir);
    const bus = new EventBus();

    logger.attach(bus);

    bus.emit("session:start", {
      sessionId: "sess-bus",
      resumedFrom: null,
    } as never);

    bus.emit("error", {
      error: "something failed",
      context: "test",
    } as never);

    logger.close();

    const entries = EventLogger.readLog("sess-bus", dir);
    expect(entries.length).toBeGreaterThanOrEqual(2);
    expect(entries[0].event).toBe("session:start");
    expect(entries[1].event).toBe("error");
  });
});
