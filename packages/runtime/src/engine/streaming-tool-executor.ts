/**
 * StreamingToolExecutor — starts executing safe (readonly) tools
 * as they arrive from the LLM stream, before the stream finishes.
 *
 * Concurrency model:
 * - Readonly tools start immediately (up to maxConcurrency)
 * - Non-readonly tools queue until all prior tools complete, then run sequentially
 * - Results are always yielded in submission order
 *
 * Inspired by claude-code-src StreamingToolExecutor pattern.
 */

import type { ToolCategory, ToolResult } from "../core/index.js";
import { extractErrorMessage } from "../core/index.js";

// ─── Types ──────────────────────────────────────────────────

export interface StreamingToolCall {
  readonly name: string;
  readonly arguments: Record<string, unknown>;
  readonly callId: string;
}

type ToolCallState = "queued" | "executing" | "completed";

interface ToolEntry {
  readonly call: StreamingToolCall;
  readonly execute: (call: StreamingToolCall) => Promise<ToolResult>;
  readonly category: ToolCategory | null;
  state: ToolCallState;
  result: ToolResult | null;
  promise: Promise<ToolResult> | null;
}

// ─── StreamingToolExecutor ──────────────────────────────────

/**
 * Manages concurrent tool execution during LLM response streaming.
 * Safe (readonly) tools start executing immediately as they arrive;
 * unsafe tools act as barriers — no readonly tool submitted AFTER
 * an unsafe tool starts until the unsafe tool completes.
 */
export class StreamingToolExecutor {
  private readonly categoryResolver: (toolName: string) => ToolCategory | null;
  private readonly maxConcurrency: number;
  private readonly entries: ToolEntry[] = [];
  private readonly abortController = new AbortController();
  private runningCount = 0;
  private completedCount = 0;
  /** When true, an unsafe tool has been seen — no more early starts. */
  private barrierHit = false;

  constructor(
    categoryResolver: (toolName: string) => ToolCategory | null,
    maxConcurrency: number = 10,
  ) {
    this.categoryResolver = categoryResolver;
    this.maxConcurrency = maxConcurrency;
  }

  /**
   * Submit a tool call for execution. Safe tools may start immediately
   * if concurrency allows and no barrier (unsafe tool) has been hit.
   * Unsafe tools act as barriers: once seen, no further early starts.
   */
  submit(
    call: StreamingToolCall,
    execute: (call: StreamingToolCall) => Promise<ToolResult>,
  ): void {
    const category = this.categoryResolver(call.name);
    const entry: ToolEntry = {
      call,
      execute,
      category,
      state: "queued",
      result: null,
      promise: null,
    };

    this.entries.push(entry);

    if (category !== "readonly") {
      // Unsafe tool acts as barrier — stop early execution of subsequent tools
      this.barrierHit = true;
      return;
    }

    // Start readonly tools immediately if concurrency allows and no barrier
    if (!this.barrierHit && this.runningCount < this.maxConcurrency && !this.abortController.signal.aborted) {
      this.startExecution(entry);
    }
  }

  /**
   * Yield results in submission order. Call after the LLM stream ends.
   *
   * - Already-completed tools yield immediately
   * - Still-executing tools are awaited
   * - Queued unsafe tools execute sequentially
   */
  async *results(): AsyncGenerator<{ call: StreamingToolCall; result: ToolResult }> {
    for (const entry of this.entries) {
      if (this.abortController.signal.aborted) {
        yield { call: entry.call, result: abortedResult() };
        continue;
      }

      switch (entry.state) {
        case "completed":
          this.completedCount++;
          yield { call: entry.call, result: entry.result! };
          break;

        case "executing":
          entry.result = await entry.promise!;
          entry.state = "completed";
          this.completedCount++;
          yield { call: entry.call, result: entry.result };
          break;

        case "queued": {
          if (this.abortController.signal.aborted) {
            yield { call: entry.call, result: abortedResult() };
            break;
          }

          // For queued readonly tools that didn't start (concurrency was full),
          // start them now. For unsafe tools, wait for all running to finish first.
          if (entry.category !== "readonly") {
            await this.waitForRunning();
          }

          if (this.abortController.signal.aborted) {
            yield { call: entry.call, result: abortedResult() };
            break;
          }

          this.startExecution(entry);
          entry.result = await entry.promise!;
          entry.state = "completed";
          this.completedCount++;
          yield { call: entry.call, result: entry.result };
          break;
        }
      }
    }
  }

  /**
   * Abort all pending executions.
   */
  abort(): void {
    this.abortController.abort();
  }

  /**
   * Number of tools not yet completed.
   */
  get pending(): number {
    return this.entries.length - this.completedCount;
  }

  /**
   * Number of completed tools.
   */
  get completed(): number {
    return this.completedCount;
  }

  /**
   * Total number of submitted tools.
   */
  get total(): number {
    return this.entries.length;
  }

  // ─── Private ────────────────────────────────────────────────

  private startExecution(entry: ToolEntry): void {
    entry.state = "executing";
    this.runningCount++;

    entry.promise = entry.execute(entry.call)
      .catch((err): ToolResult => ({
        success: false,
        output: "",
        error: extractErrorMessage(err),
        artifacts: [],
      }))
      .finally(() => {
        this.runningCount--;
      });
  }

  private async waitForRunning(): Promise<void> {
    const running = this.entries.filter((e) => e.state === "executing" && e.promise);
    if (running.length === 0) return;
    await Promise.all(running.map((e) => e.promise));
  }
}

// ─── Helpers ────────────────────────────────────────────────

function abortedResult(): ToolResult {
  return {
    success: false,
    output: "",
    error: "Tool execution aborted",
    artifacts: [],
  };
}
