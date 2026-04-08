import { describe, it, expect, vi } from "vitest";
import { EventBus } from "./events.js";

describe("EventBus", () => {
  it("emits events to subscribed handlers", () => {
    const bus = new EventBus();
    const handler = vi.fn();

    bus.on("session:start", handler);
    bus.emit("session:start", { sessionId: "test-123" });

    expect(handler).toHaveBeenCalledOnce();
    expect(handler).toHaveBeenCalledWith({ sessionId: "test-123" });
  });

  it("supports multiple handlers for the same event", () => {
    const bus = new EventBus();
    const handler1 = vi.fn();
    const handler2 = vi.fn();

    bus.on("session:start", handler1);
    bus.on("session:start", handler2);
    bus.emit("session:start", { sessionId: "test-123" });

    expect(handler1).toHaveBeenCalledOnce();
    expect(handler2).toHaveBeenCalledOnce();
  });

  it("returns unsubscribe function from on()", () => {
    const bus = new EventBus();
    const handler = vi.fn();

    const unsubscribe = bus.on("session:start", handler);
    unsubscribe();
    bus.emit("session:start", { sessionId: "test-123" });

    expect(handler).not.toHaveBeenCalled();
  });

  it("once() fires handler only once", () => {
    const bus = new EventBus();
    const handler = vi.fn();

    bus.once("session:start", handler);
    bus.emit("session:start", { sessionId: "1" });
    bus.emit("session:start", { sessionId: "2" });

    expect(handler).toHaveBeenCalledOnce();
    expect(handler).toHaveBeenCalledWith({ sessionId: "1" });
  });

  it("does not throw when emitting with no listeners", () => {
    const bus = new EventBus();
    expect(() =>
      bus.emit("session:start", { sessionId: "test" }),
    ).not.toThrow();
  });

  it("logs errors from handlers but does not stop other handlers", () => {
    const bus = new EventBus();
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    const badHandler = vi.fn(() => {
      throw new Error("handler broke");
    });
    const goodHandler = vi.fn();

    bus.on("session:start", badHandler);
    bus.on("session:start", goodHandler);
    bus.emit("session:start", { sessionId: "test" });

    expect(badHandler).toHaveBeenCalled();
    expect(goodHandler).toHaveBeenCalled();
    expect(errorSpy).toHaveBeenCalledWith(
      expect.stringContaining("handler broke"),
    );
    errorSpy.mockRestore();
  });

  it("removeAllListeners clears specific event", () => {
    const bus = new EventBus();
    const handler = vi.fn();

    bus.on("session:start", handler);
    bus.removeAllListeners("session:start");
    bus.emit("session:start", { sessionId: "test" });

    expect(handler).not.toHaveBeenCalled();
  });

  it("removeAllListeners with no args clears everything", () => {
    const bus = new EventBus();
    const h1 = vi.fn();
    const h2 = vi.fn();

    bus.on("session:start", h1);
    bus.on("session:end", h2);
    bus.removeAllListeners();
    bus.emit("session:start", { sessionId: "test" });
    bus.emit("session:end", {
      sessionId: "test",
      reason: "completed",
    });

    expect(h1).not.toHaveBeenCalled();
    expect(h2).not.toHaveBeenCalled();
  });

  it("listenerCount returns correct count", () => {
    const bus = new EventBus();

    expect(bus.listenerCount("session:start")).toBe(0);

    bus.on("session:start", () => {});
    bus.on("session:start", () => {});
    expect(bus.listenerCount("session:start")).toBe(2);
  });

  it("handles tool events with correct types", () => {
    const bus = new EventBus();
    const handler = vi.fn();

    bus.on("tool:after", handler);
    bus.emit("tool:after", {
      name: "read_file",
      callId: "call-1",
      durationMs: 42,
      result: {
        success: true,
        output: "file contents",
        error: null,
        artifacts: [],
      },
    });

    expect(handler).toHaveBeenCalledWith(
      expect.objectContaining({
        name: "read_file",
        durationMs: 42,
        result: expect.objectContaining({ success: true }),
      }),
    );
  });

  it("preserves typed file edit previews on tool:after events", () => {
    const bus = new EventBus();
    const handler = vi.fn();

    bus.on("tool:after", handler);
    bus.emit("tool:after", {
      name: "write_file",
      callId: "call-edit-1",
      durationMs: 15,
      fileEditHiddenCount: 1,
      fileEdits: [{
        path: "src/new-file.ts",
        kind: "create",
        additions: 3,
        deletions: 0,
        unifiedDiff: "--- /dev/null\n+++ b/src/new-file.ts\n@@ -0,0 +1,3 @@\n+export const x = 1;",
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
        output: "Wrote 18 bytes to src/new-file.ts",
        error: null,
        artifacts: ["src/new-file.ts"],
      },
    });

    expect(handler).toHaveBeenCalledWith(
      expect.objectContaining({
        name: "write_file",
        fileEdits: [
          expect.objectContaining({
            path: "src/new-file.ts",
            kind: "create",
            additions: 3,
            before: "",
            after: "export const x = 1;\n",
            structuredDiff: expect.objectContaining({
              hunks: expect.any(Array),
            }),
          }),
        ],
        fileEditHiddenCount: 1,
      }),
    );
  });
});
