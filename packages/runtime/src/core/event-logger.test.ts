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

  it("persists and summarizes subagent lifecycle events", () => {
    const dir = makeTempDir();
    const logger = new EventLogger("sess-subagents", dir);
    const bus = new EventBus();

    logger.attach(bus);

    bus.emit("subagent:start", {
      agentId: "root-sub-1",
      parentAgentId: "root",
      depth: 1,
      agentType: "explore",
      laneLabel: "frontend",
      objective: "Inspect frontend lane",
      model: "gpt-5.4-mini",
      reasoningEffort: "low",
      status: "running",
      batchId: "batch-1",
      batchSize: 2,
    } as never);
    bus.emit("tool:before", {
      name: "search_files",
      params: { pattern: "FixedArray" },
      callId: "call-1",
      agentId: "root-sub-1",
      parentAgentId: "root",
      depth: 1,
      agentType: "explore",
    } as never);
    bus.emit("subagent:end", {
      agentId: "root-sub-1",
      parentAgentId: "root",
      depth: 1,
      agentType: "explore",
      laneLabel: "frontend",
      objective: "Inspect frontend lane",
      model: "gpt-5.4-mini",
      reasoningEffort: "low",
      status: "completed",
      durationMs: 4200,
      iterations: 3,
      cost: {
        inputTokens: 100,
        outputTokens: 20,
        cacheReadTokens: 0,
        cacheWriteTokens: 0,
        totalCost: 0.01,
      },
      parsedOutputKeys: ["answer", "evidence"],
      batchId: "batch-1",
      batchSize: 2,
    } as never);

    logger.close();

    const entries = EventLogger.readLog("sess-subagents", dir);
    const summary = EventLogger.summarizeSubagents(entries);
    expect(entries.some((entry) => entry.event === "subagent:start")).toBe(true);
    expect(summary.childCount).toBe(1);
    expect(summary.children[0]).toMatchObject({
      agentId: "root-sub-1",
      agentType: "explore",
      laneLabel: "frontend",
      status: "completed",
      durationMs: 4200,
      toolCalls: 1,
    });
    expect(summary.children[0]?.quality).toBeUndefined();
    expect(summary.parallelBatchCount).toBe(1);
    expect(summary.maxParallelChildren).toBe(2);
  });

  it("suppresses streamed assistant chunks and sanitizes delegate tool results", () => {
    const dir = makeTempDir();
    const logger = new EventLogger("sess-sanitized", dir);
    const bus = new EventBus();

    logger.attach(bus);

    bus.emit("message:assistant", {
      content: "partial child chunk",
      partial: true,
      agentId: "root-sub-1",
      parentAgentId: "root",
      depth: 1,
      agentType: "explore",
    } as never);
    bus.emit("message:assistant", {
      content: "root final text",
      partial: false,
    } as never);
    bus.emit("message:assistant", {
      content: "root partial chunk",
      partial: true,
    } as never);
    bus.emit("tool:after", {
      name: "delegate",
      callId: "call-1",
      durationMs: 4500,
      result: {
        success: true,
        output: "Subagent full JSON payload",
        error: null,
        artifacts: [],
        metadata: {
          agentMeta: { agentId: "root-sub-1", parentId: "root", depth: 1, agentType: "explore" },
          parsedOutput: { answer: "too much detail" },
          childSessionState: { plan: [] },
          quality: { score: 0.81, completeness: "partial" },
          delegateSummary: {
            agentId: "root-sub-1",
            agentType: "explore",
            laneLabel: "docs/spec",
            durationMs: 4500,
            iterations: 6,
            quality: { score: 0.81, completeness: "partial" },
          },
        },
      },
    } as never);

    logger.close();

    const entries = EventLogger.readLog("sess-sanitized", dir);
    expect(entries.some((entry) => entry.event === "message:assistant" && String((entry.data as { content?: string }).content).includes("partial child chunk"))).toBe(false);
    expect(entries.some((entry) => entry.event === "message:assistant" && String((entry.data as { content?: string }).content).includes("root partial chunk"))).toBe(false);
    const delegateAfter = entries.find((entry) => entry.event === "tool:after");
    expect(delegateAfter).toBeTruthy();
    const data = delegateAfter!.data as { result: { output: string; metadata: Record<string, unknown> } };
    expect(data.result.output).toContain("root-sub-1");
    expect(data.result.output).toContain("docs/spec");
    expect(data.result.output).not.toContain("full JSON payload");
    expect(data.result.metadata).not.toHaveProperty("parsedOutput");
    expect(data.result.metadata).not.toHaveProperty("childSessionState");
    expect(entries.some((entry) => entry.event === "message:assistant" && String((entry.data as { content?: string }).content).includes("root final text"))).toBe(true);
  });

  it("strips before/after snapshots from persisted tool:after file edits", () => {
    const dir = makeTempDir();
    const logger = new EventLogger("sess-file-edits", dir);
    const bus = new EventBus();

    logger.attach(bus);

    bus.emit("tool:after", {
      name: "write_file",
      callId: "call-file-1",
      durationMs: 12,
      fileEdits: [{
        path: "src/new.ts",
        kind: "create",
        additions: 1,
        deletions: 0,
        unifiedDiff: "--- /dev/null\n+++ b/src/new.ts\n@@ -0,0 +1,1 @@\n+export const x = 1;",
        truncated: false,
        structuredDiff: {
          hunks: [{
            oldStart: 0,
            oldLines: 0,
            newStart: 1,
            newLines: 1,
            lines: [{
              type: "add",
              text: "export const x = 1;",
              oldLine: null,
              newLine: 1,
            }],
          }],
        },
        before: "",
        after: "export const x = 1;\n",
      }],
      result: {
        success: true,
        output: "Wrote file",
        error: null,
        artifacts: ["src/new.ts"],
        metadata: {
          fileEdits: [{
            path: "src/new.ts",
            kind: "create",
            additions: 1,
            deletions: 0,
            unifiedDiff: "--- /dev/null\n+++ b/src/new.ts\n@@ -0,0 +1,1 @@\n+export const x = 1;",
            truncated: false,
            structuredDiff: {
              hunks: [{
                oldStart: 0,
                oldLines: 0,
                newStart: 1,
                newLines: 1,
                lines: [{
                  type: "add",
                  text: "export const x = 1;",
                  oldLine: null,
                  newLine: 1,
                }],
              }],
            },
            before: "",
            after: "export const x = 1;\n",
          }],
        },
      },
    } as never);

    logger.close();

    const entries = EventLogger.readLog("sess-file-edits", dir);
    const toolAfter = entries.find((entry) => entry.event === "tool:after");
    expect(toolAfter).toBeTruthy();
    const data = toolAfter!.data as {
      fileEdits: Array<Record<string, unknown>>;
      result: { metadata?: Record<string, unknown> };
    };
    expect(data.fileEdits[0]).not.toHaveProperty("before");
    expect(data.fileEdits[0]).not.toHaveProperty("after");
    expect(data.fileEdits[0]).not.toHaveProperty("structuredDiff");
    const persistedMetadataEdits = data.result.metadata?.["fileEdits"] as Array<Record<string, unknown>>;
    expect(persistedMetadataEdits[0]).not.toHaveProperty("before");
    expect(persistedMetadataEdits[0]).not.toHaveProperty("after");
    expect(persistedMetadataEdits[0]).not.toHaveProperty("structuredDiff");
  });

  it("strips bulky command and validation previews from persisted tool:after metadata", () => {
    const dir = makeTempDir();
    const logger = new EventLogger("sess-tool-metadata", dir);
    const bus = new EventBus();

    logger.attach(bus);

    bus.emit("tool:after", {
      name: "run_command",
      callId: "call-cmd-1",
      durationMs: 25,
      result: {
        success: false,
        output: "Exit code: 1",
        error: "Command exited with code 1",
        artifacts: [],
        metadata: {
          commandResult: {
            command: "npm test",
            cwd: ".",
            exitCode: 1,
            timedOut: false,
            warningOnly: false,
            stdoutPreview: "lots of stdout",
            stderrPreview: "lots of stderr",
            stdoutTruncated: true,
            stderrTruncated: true,
          },
          validationResult: {
            passed: false,
            diagnosticErrors: ["src/a.ts: boom", "src/b.ts: bang"],
            testPassed: false,
            testOutputPreview: "full stack trace",
            testSummary: {
              framework: "vitest",
              passed: 10,
              failed: 2,
              failureMessages: ["first failure"],
            },
          },
        },
      },
    } as never);

    logger.close();

    const entries = EventLogger.readLog("sess-tool-metadata", dir);
    const toolAfter = entries.find((entry) => entry.event === "tool:after");
    expect(toolAfter).toBeTruthy();
    const metadata = (toolAfter!.data as { result: { metadata?: Record<string, unknown> } }).result.metadata!;
    expect(metadata["commandResult"]).toEqual({
      command: "npm test",
      cwd: ".",
      exitCode: 1,
      timedOut: false,
      warningOnly: false,
      stdoutTruncated: true,
      stderrTruncated: true,
    });
    expect(metadata["validationResult"]).toEqual({
      passed: false,
      diagnosticCount: 2,
      testPassed: false,
      testSummary: {
        framework: "vitest",
        passed: 10,
        failed: 2,
      },
    });
  });
});
