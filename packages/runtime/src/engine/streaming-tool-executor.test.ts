import { describe, it, expect, vi } from "vitest";
import { StreamingToolExecutor } from "./streaming-tool-executor.js";
import type { ToolCategory, ToolResult } from "../core/index.js";

// ─── Helpers ────────────────────────────────────────────────

function okResult(output = "ok"): ToolResult {
  return { success: true, output, error: null, artifacts: [] };
}

function failResult(error = "boom"): ToolResult {
  return { success: false, output: "", error, artifacts: [] };
}

function makeCall(id: string, name = "read_file") {
  return { name, arguments: {}, callId: id };
}

/** Deferred promise — lets tests control when an executor resolves. */
function deferred<T>() {
  let resolve!: (v: T) => void;
  let reject!: (e: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function delay(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}

// ─── Tests ──────────────────────────────────────────────────

describe("StreamingToolExecutor", () => {
  // ── submit ────────────────────────────────────────────────

  describe("submit()", () => {
    it("queues tools correctly", () => {
      const executor = new StreamingToolExecutor(() => "readonly");
      const exec = vi.fn(async () => okResult());

      executor.submit(makeCall("c1"), exec);
      executor.submit(makeCall("c2"), exec);

      expect(executor.total).toBe(2);
    });

    it("starts readonly tools immediately", () => {
      const exec = vi.fn(async () => okResult());
      const executor = new StreamingToolExecutor(() => "readonly");

      executor.submit(makeCall("c1"), exec);

      // readonly tool should already be executing
      expect(exec).toHaveBeenCalledTimes(1);
    });

    it("does not start non-readonly tools until results() is called", async () => {
      const exec = vi.fn(async () => okResult());
      const executor = new StreamingToolExecutor(() => "mutating");

      executor.submit(makeCall("c1", "write_file"), exec);

      // mutating tool stays queued
      expect(exec).not.toHaveBeenCalled();
      expect(executor.pending).toBe(1);

      // drain results to start execution
      const results: ToolResult[] = [];
      for await (const r of executor.results()) {
        results.push(r.result);
      }
      expect(exec).toHaveBeenCalledTimes(1);
      expect(results).toHaveLength(1);
      expect(results[0]!.success).toBe(true);
    });
  });

  // ── results() ordering ────────────────────────────────────

  describe("results()", () => {
    it("yields results in submission order", async () => {
      const d1 = deferred<ToolResult>();
      const d2 = deferred<ToolResult>();

      const executor = new StreamingToolExecutor(() => "readonly");
      executor.submit(makeCall("c1"), () => d1.promise);
      executor.submit(makeCall("c2"), () => d2.promise);

      // Resolve second before first
      d2.resolve(okResult("second"));
      d1.resolve(okResult("first"));

      const order: string[] = [];
      for await (const r of executor.results()) {
        order.push(r.call.callId);
      }
      expect(order).toEqual(["c1", "c2"]);
    });

    it("handles mixed readonly + mutating batch ordering", async () => {
      const resolver = (name: string): ToolCategory | null => {
        return name === "read_file" ? "readonly" : "mutating";
      };
      const executor = new StreamingToolExecutor(resolver);

      const callOrder: string[] = [];
      const makeExec = (id: string) => async () => {
        callOrder.push(id);
        return okResult(id);
      };

      executor.submit(makeCall("c1", "read_file"), makeExec("c1"));
      executor.submit(makeCall("c2", "write_file"), makeExec("c2"));
      executor.submit(makeCall("c3", "read_file"), makeExec("c3"));

      const results: string[] = [];
      for await (const r of executor.results()) {
        results.push(r.call.callId);
      }

      // Results are yielded in submission order
      expect(results).toEqual(["c1", "c2", "c3"]);
      // c1 (readonly) was started immediately, c2 (mutating) waited
      expect(callOrder[0]).toBe("c1");
    });
  });

  // ── concurrency ───────────────────────────────────────────

  describe("concurrency", () => {
    it("respects maxConcurrency for readonly tools", () => {
      const deferreds = Array.from({ length: 5 }, () => deferred<ToolResult>());
      let callIndex = 0;

      const executor = new StreamingToolExecutor(() => "readonly", 2);

      for (let i = 0; i < 5; i++) {
        const d = deferreds[i]!;
        executor.submit(makeCall(`c${i}`), () => {
          callIndex++;
          return d.promise;
        });
      }

      // Only 2 should have started (maxConcurrency = 2)
      expect(callIndex).toBe(2);
      expect(executor.pending).toBe(5);
    });
  });

  // ── abort ─────────────────────────────────────────────────

  describe("abort()", () => {
    it("stops pending executions and returns abort results", async () => {
      const executor = new StreamingToolExecutor(() => "mutating");
      executor.submit(makeCall("c1", "write_file"), async () => okResult());
      executor.submit(makeCall("c2", "write_file"), async () => okResult());

      executor.abort();

      const results: ToolResult[] = [];
      for await (const r of executor.results()) {
        results.push(r.result);
      }

      expect(results).toHaveLength(2);
      expect(results[0]!.success).toBe(false);
      expect(results[0]!.error).toBe("Tool execution aborted");
      expect(results[1]!.success).toBe(false);
      expect(results[1]!.error).toBe("Tool execution aborted");
    });
  });

  // ── error handling ────────────────────────────────────────

  describe("error handling", () => {
    it("returns error ToolResult when execution throws", async () => {
      const executor = new StreamingToolExecutor(() => "readonly");

      executor.submit(makeCall("c1"), async () => {
        throw new Error("kaboom");
      });

      const results: ToolResult[] = [];
      for await (const r of executor.results()) {
        results.push(r.result);
      }

      expect(results).toHaveLength(1);
      expect(results[0]!.success).toBe(false);
      expect(results[0]!.error).toBe("kaboom");
    });
  });

  // ── getters ───────────────────────────────────────────────

  describe("pending/completed getters", () => {
    it("tracks pending and completed correctly", async () => {
      const d = deferred<ToolResult>();
      const executor = new StreamingToolExecutor(() => "readonly");

      executor.submit(makeCall("c1"), () => d.promise);
      expect(executor.pending).toBe(1);
      expect(executor.completed).toBe(0);
      expect(executor.total).toBe(1);

      d.resolve(okResult());

      // Drain to mark as completed
      for await (const _ of executor.results()) {
        /* consume */
      }

      expect(executor.pending).toBe(0);
      expect(executor.completed).toBe(1);
    });
  });
});
