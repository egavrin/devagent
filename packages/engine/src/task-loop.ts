/**
 * ReAct task loop — the core execution engine.
 * Streams LLM responses, parses tool calls, checks approval,
 * executes tools, feeds results back to LLM.
 * Fail fast: tool errors surface to LLM immediately.
 *
 * Auto-compaction: monitors estimated token usage and triggers
 * context truncation (sliding window or LLM-based summarization)
 * when approaching the budget limit, following the Codex pattern.
 */

import type {
  LLMProvider,
  Message,
  ToolSpec,
  StreamChunk,
  ToolResult,
  CostRecord,
  DevAgentConfig,
} from "@devagent/core";
import {
  MessageRole,
  EventBus,
  ApprovalGate,
  ProviderError,
  ContextManager,
  estimateMessageTokens,
} from "@devagent/core";
import type { MemoryStore } from "@devagent/core";
import type { ToolRegistry } from "@devagent/tools";
import type { CheckpointManager } from "./checkpoints.js";
import type { DoubleCheck } from "./double-check.js";

// ─── Types ──────────────────────────────────────────────────

export type TaskMode = "plan" | "act";

export type TaskCompletionStatus =
  | "success"
  | "empty_response"
  | "budget_exceeded"
  | "aborted";

/**
 * Callback invoked at midpoint briefing intervals during long-running turns.
 * Receives the current messages and iteration count. Returns replacement
 * messages (synthesized context) or null to skip midpoint briefing.
 */
export type MidpointCallback = (
  messages: ReadonlyArray<Message>,
  iteration: number,
) => Promise<{ continueMessages: ReadonlyArray<Message> } | null>;

export interface TaskLoopOptions {
  readonly provider: LLMProvider;
  readonly tools: ToolRegistry;
  readonly bus: EventBus;
  readonly approvalGate: ApprovalGate;
  readonly config: DevAgentConfig;
  readonly systemPrompt: string;
  readonly repoRoot: string;
  readonly mode?: TaskMode;
  readonly contextManager?: ContextManager;
  readonly memoryStore?: MemoryStore;
  readonly checkpointManager?: CheckpointManager;
  readonly doubleCheck?: DoubleCheck;
  readonly initialMessages?: ReadonlyArray<Message>;
  /** Callback for midpoint context re-synthesis during long-running turns. */
  readonly midpointCallback?: MidpointCallback;
}

export interface TaskLoopResult {
  readonly messages: ReadonlyArray<Message>;
  readonly iterations: number;
  readonly cost: CostRecord;
  readonly aborted: boolean;
  readonly status: TaskCompletionStatus;
  readonly lastText: string | null;
}

interface PendingToolCall {
  readonly name: string;
  readonly arguments: Record<string, unknown>;
  readonly callId: string;
}

interface ToolCallBatch {
  readonly parallel: boolean;
  readonly calls: ReadonlyArray<PendingToolCall>;
}

// ─── Task Loop ──────────────────────────────────────────────

export class TaskLoop {
  private readonly provider: LLMProvider;
  private readonly tools: ToolRegistry;
  private readonly bus: EventBus;
  private readonly approvalGate: ApprovalGate;
  private readonly config: DevAgentConfig;
  private readonly systemPrompt: string;
  private readonly repoRoot: string;
  private readonly contextManager: ContextManager | null;
  private readonly memoryStore: MemoryStore | null;
  private readonly checkpointManager: CheckpointManager | null;
  private readonly doubleCheck: DoubleCheck | null;
  private readonly midpointCallback: MidpointCallback | null;
  private readonly midpointInterval: number;
  private mode: TaskMode;
  private messages: Message[] = [];
  private iterations = 0;
  private totalCost: CostRecord = {
    inputTokens: 0,
    outputTokens: 0,
    cacheReadTokens: 0,
    cacheWriteTokens: 0,
    totalCost: 0,
  };
  private aborted = false;
  private recentToolCalls: Array<{ name: string; argsKey: string }> = [];
  private doomLoopWarned = false;
  // Per-tool consecutive failure counts (reset on any success of that tool)
  private toolFailureCounts: Map<string, number> = new Map();
  private toolFatigueWarned: Set<string> = new Set();
  private unresolvedDoubleCheckFailure = false;

  constructor(options: TaskLoopOptions) {
    this.provider = options.provider;
    this.tools = options.tools;
    this.bus = options.bus;
    this.approvalGate = options.approvalGate;
    this.config = options.config;
    this.systemPrompt = options.systemPrompt;
    this.repoRoot = options.repoRoot;
    this.mode = options.mode ?? "act";
    this.contextManager = options.contextManager ?? null;
    this.memoryStore = options.memoryStore ?? null;
    this.checkpointManager = options.checkpointManager ?? null;
    this.doubleCheck = options.doubleCheck ?? null;
    this.midpointCallback = options.midpointCallback ?? null;
    this.midpointInterval =
      options.config.context.midpointBriefingInterval ?? DEFAULT_MIDPOINT_INTERVAL;

    // Initialize messages: from previous session or fresh system prompt
    if (options.initialMessages && options.initialMessages.length > 0) {
      this.messages = [...options.initialMessages];
    } else {
      this.messages.push({
        role: MessageRole.SYSTEM,
        content: this.systemPrompt,
      });
    }
  }

  /**
   * Run the task loop with a user query.
   * Returns when the LLM produces a final text response (no more tool calls)
   * or when the budget is exceeded.
   */
  async run(userQuery: string): Promise<TaskLoopResult> {
    this.unresolvedDoubleCheckFailure = false;

    // Add user message
    this.messages.push({
      role: MessageRole.USER,
      content: userQuery,
    });
    this.bus.emit("message:user", { content: userQuery });

    let hadToolCalls = false;
    let summaryRequested = false;
    let budgetGraceUsed = false;
    let lastNonEmptyText: string | null = null;
    let status: TaskCompletionStatus = "success";

    while (!this.aborted) {
      // Check budget (0 = unlimited)
      if (this.config.budget.maxIterations > 0 && this.iterations >= this.config.budget.maxIterations) {
        if (!budgetGraceUsed) {
          // Grace iteration: ask the model to summarize before stopping
          budgetGraceUsed = true;
          this.messages.push({
            role: MessageRole.SYSTEM,
            content: "You have reached the iteration limit. Please provide a concise summary of your progress and findings so far. Do not use any tools — respond with text only.",
          });
          // Fall through to allow one more LLM call
        } else {
          status = "budget_exceeded";
          break;
        }
      }

      this.iterations++;

      // Get available tools based on mode (no tools during grace iteration)
      const availableTools = budgetGraceUsed ? [] : this.getAvailableTools();

      // Preflight compaction: ensure context fits before calling the provider.
      await this.maybeCompactContext();

      // Stream LLM response with retry on transient provider errors
      let textContent = "";
      let toolCalls: PendingToolCall[] = [];
      let overflowCompactionUsed = false;
      for (let attempt = 0; attempt < MAX_RETRY_ATTEMPTS; attempt++) {
        try {
          const result = await this.streamLLMResponse(availableTools);
          textContent = result.textContent;
          toolCalls = result.toolCalls;
          break;
        } catch (err) {
          if (!(err instanceof ProviderError)) throw err;

          // If provider reports a context overflow, force one compaction pass and retry immediately.
          if (
            !overflowCompactionUsed &&
            this.contextManager &&
            this.isContextOverflowError(err.message)
          ) {
            overflowCompactionUsed = true;
            await this.maybeCompactContext({ force: true });
            this.bus.emit("error", {
              message: "Provider rejected prompt for context size. Forced compaction and retrying immediately.",
              code: "CONTEXT_OVERFLOW_RETRY",
              fatal: false,
            });
            continue;
          }

          if (attempt >= RETRY_DELAYS.length) throw err; // Exhausted retries
          this.bus.emit("error", {
            message: `Provider error (attempt ${attempt + 1}/${MAX_RETRY_ATTEMPTS}): ${(err as Error).message}. Retrying in ${RETRY_DELAYS[attempt]!}ms…`,
            code: "PROVIDER_RETRY",
            fatal: false,
          });
          await sleep(RETRY_DELAYS[attempt]!);
        }
      }

      // Track last non-empty text from the LLM (even if tool calls follow)
      if (textContent.trim()) {
        lastNonEmptyText = textContent;
      }

      if (toolCalls.length > 0) {
        hadToolCalls = true;

        // Add single assistant message with both text and tool calls
        this.messages.push({
          role: MessageRole.ASSISTANT,
          content: textContent,
          toolCalls: toolCalls.map((tc) => ({
            name: tc.name,
            arguments: tc.arguments,
            callId: tc.callId,
          })),
        });

        // Coalesce replace-all tool calls (e.g., multiple update_plan in one batch)
        const { toExecute, skipped } =
          this.coalesceReplaceAllCalls(toolCalls);
        for (const tc of skipped) {
          this.messages.push({
            role: MessageRole.TOOL,
            content:
              "Skipped: superseded by a later update_plan call in this batch.",
            toolCallId: tc.callId,
          });
        }

        // Execute tool calls — parallel for independent readonly, sequential for mutating.
        // Partition into batches: consecutive readonly calls form a parallel batch,
        // a mutating/workflow call is its own sequential batch.
        const batches = this.partitionToolCalls(toExecute, availableTools);

        for (const batch of batches) {
          if (this.aborted) break;

          if (batch.parallel) {
            // Run all calls in the batch concurrently
            const promises = batch.calls.map((tc) =>
              this.executeToolCall(tc, availableTools).then((result) => ({
                callId: tc.callId,
                result,
              })),
            );
            const settled = await Promise.all(promises);

            // Append results in original order (API requires matching order)
            for (const { callId, result } of settled) {
              this.messages.push({
                role: MessageRole.TOOL,
                content: result.success
                  ? result.output
                  : `Error: ${result.error}`,
                toolCallId: callId,
              });
            }
          } else {
            // Sequential execution (mutating tools, or single call)
            for (const tc of batch.calls) {
              if (this.aborted) break;
              const result = await this.executeToolCall(tc, availableTools);
              this.messages.push({
                role: MessageRole.TOOL,
                content: result.success
                  ? result.output
                  : `Error: ${result.error}`,
                toolCallId: tc.callId,
              });
            }
          }
        }

        // Doom loop detection: warn the LLM if it's repeating identical failing calls
        const doomLoopWarning = this.checkDoomLoop(toolCalls);
        if (doomLoopWarning) {
          this.messages.push({
            role: MessageRole.SYSTEM,
            content: doomLoopWarning,
          });
        }

        // Tool fatigue detection: same tool failing repeatedly with different args
        const fatigueWarning = this.checkToolFatigue(toolCalls);
        if (fatigueWarning) {
          this.messages.push({
            role: MessageRole.SYSTEM,
            content: fatigueWarning,
          });
        }

        // Auto-compact: check if context is approaching token budget
        await this.maybeCompactContext();

        // Proactive midpoint briefing: re-synthesize context every N iterations
        // to prevent accumulated history from degrading accuracy (paper finding)
        await this.maybeMidpointBriefing();

        // Continue loop — feed tool results back to LLM
        continue;
      }

      // No tool calls — LLM produced a final text response
      if (textContent) {
        if (this.unresolvedDoubleCheckFailure) {
          this.messages.push({
            role: MessageRole.ASSISTANT,
            content: textContent,
          });
          this.messages.push({
            role: MessageRole.SYSTEM,
            content: "Double-check still failing from prior edits. You must fix validation errors before finalizing.",
          });
          continue;
        }

        this.messages.push({
          role: MessageRole.ASSISTANT,
          content: textContent,
        });
        this.bus.emit("message:assistant", {
          content: textContent,
          partial: false,
        });
        status = "success";
        break;
      }

      // Empty response — try to get a summary if work was done
      if (hadToolCalls && !summaryRequested) {
        summaryRequested = true;
        this.messages.push({
          role: MessageRole.SYSTEM,
          content: "Please provide a summary of your findings and conclusions based on the work done so far.",
        });
        continue;
      }

      // Still empty after summary request — give up gracefully
      status = hadToolCalls ? "empty_response" : "success";
      break;
    }

    if (this.aborted && status === "success") {
      status = "aborted";
    }

    // Extract lessons from this session into persistent memory
    this.extractLessons();

    return {
      messages: this.messages,
      iterations: this.iterations,
      cost: this.totalCost,
      aborted: this.aborted,
      status,
      lastText: lastNonEmptyText,
    };
  }

  abort(): void {
    this.aborted = true;
    this.provider.abort();
  }

  setMode(mode: TaskMode): void {
    this.mode = mode;
  }

  getMode(): TaskMode {
    return this.mode;
  }

  /**
   * Get the current message history.
   */
  getMessages(): ReadonlyArray<Message> {
    return this.messages;
  }

  /**
   * Get the current iteration count.
   */
  getIterations(): number {
    return this.iterations;
  }

  /**
   * Reset iteration counter for a new turn in multi-turn conversations.
   * Does NOT clear message history — messages persist for context.
   */
  resetIterations(): void {
    this.iterations = 0;
    this.aborted = false;
    this.recentToolCalls = [];
    this.doomLoopWarned = false;
    this.toolFailureCounts.clear();
    this.toolFatigueWarned.clear();
    this.unresolvedDoubleCheckFailure = false;
  }

  // ─── Private ────────────────────────────────────────────────

  /**
   * Auto-compact context when approaching the token budget.
   * Uses the ContextManager's truncateAsync (sliding window or hybrid
   * summarization) to compress older messages while preserving the
   * system prompt and original user task.
   */
  private async maybeCompactContext(
    options?: { force?: boolean },
  ): Promise<void> {
    if (!this.contextManager) return;

    const maxTokens = this.getEffectiveContextBudget();
    if (maxTokens <= 0) return;

    const estimatedTokens = estimateMessageTokens(this.messages);
    const threshold = maxTokens * this.config.context.triggerRatio;

    if (!options?.force && estimatedTokens <= threshold) return;

    this.bus.emit("context:compacting", { estimatedTokens, maxTokens });

    try {
      const result = await this.contextManager.truncateAsync(
        this.messages,
        maxTokens,
      );

      if (result.truncated) {
        this.messages = [...result.messages];
        this.bus.emit("context:compacted", {
          removedCount: result.removedCount,
          estimatedTokens: result.estimatedTokens,
        });
      }
      if (result.estimatedTokens > maxTokens) {
        throw new Error(
          `Compaction did not fit budget: ${result.estimatedTokens} > ${maxTokens}`,
        );
      }
    } catch (err) {
      this.bus.emit("error", {
        message: `Context compaction failed: ${(err as Error).message}`,
        code: "COMPACTION_FAILED",
        fatal: true,
      });
      throw err;
    }
  }

  private getEffectiveContextBudget(): number {
    const maxContextTokens = this.config.budget.maxContextTokens;
    if (maxContextTokens <= 0) return 0;
    const headroom = Math.max(0, this.config.budget.responseHeadroom);
    const effective = maxContextTokens - headroom;
    if (effective <= 0) {
      throw new Error(
        `Invalid context budget: budget.maxContextTokens (${maxContextTokens}) must be greater than budget.responseHeadroom (${headroom})`,
      );
    }
    return effective;
  }

  private isContextOverflowError(message: string): boolean {
    const normalized = message.toLowerCase();
    return (
      normalized.includes("context length") ||
      normalized.includes("maximum context") ||
      normalized.includes("max context") ||
      normalized.includes("token limit") ||
      normalized.includes("too many tokens") ||
      normalized.includes("prompt is too long")
    );
  }

  /**
   * Proactive midpoint briefing: at regular intervals, re-synthesize
   * context to prevent accumulated history from degrading LLM accuracy.
   * Unlike reactive compaction (which triggers at token budget limit),
   * this fires proactively every N iterations regardless of token count.
   */
  private async maybeMidpointBriefing(): Promise<void> {
    if (!this.midpointCallback) return;
    if (this.midpointInterval <= 0) return;
    if (this.iterations <= 0 || this.iterations % this.midpointInterval !== 0) return;

    const result = await this.midpointCallback(this.messages, this.iterations);
    if (result) {
      this.messages = [...result.continueMessages];
      this.bus.emit("context:compacted", {
        removedCount: 0,
        estimatedTokens: estimateMessageTokens(this.messages),
      });
    }
  }

  private getAvailableTools(): ReadonlyArray<ToolSpec> {
    if (this.mode === "plan") {
      return this.tools.getReadOnly();
    }
    return this.tools.getAll();
  }

  /**
   * Coalesce replace-all tool calls: when multiple calls to the same tool
   * with replace-all semantics (like update_plan) appear in a single LLM
   * response batch, only execute the last one. Earlier calls are skipped
   * with synthetic results.
   */
  private coalesceReplaceAllCalls(
    toolCalls: ReadonlyArray<PendingToolCall>,
  ): { toExecute: PendingToolCall[]; skipped: PendingToolCall[] } {
    // Tools where only the last call in a batch matters
    const replaceAllTools = new Set(["update_plan"]);

    // Find the last index of each replace-all tool
    const lastIndex = new Map<string, number>();
    for (let i = toolCalls.length - 1; i >= 0; i--) {
      const tc = toolCalls[i]!;
      if (replaceAllTools.has(tc.name) && !lastIndex.has(tc.name)) {
        lastIndex.set(tc.name, i);
      }
    }

    const toExecute: PendingToolCall[] = [];
    const skipped: PendingToolCall[] = [];

    for (let i = 0; i < toolCalls.length; i++) {
      const tc = toolCalls[i]!;
      if (replaceAllTools.has(tc.name) && i !== lastIndex.get(tc.name)) {
        skipped.push(tc);
      } else {
        toExecute.push(tc);
      }
    }

    return { toExecute, skipped };
  }

  /**
   * Partition tool calls into batches for parallel/sequential execution.
   *
   * Rules:
   * - Consecutive readonly calls → single parallel batch (safe to run concurrently)
   * - A mutating/workflow/external call → its own sequential batch (must run alone)
   * - Unknown tools → sequential batch (fail gracefully)
   *
   * A single readonly call is still "parallel" (batch of 1) — no overhead.
   */
  private partitionToolCalls(
    toolCalls: ReadonlyArray<PendingToolCall>,
    availableTools: ReadonlyArray<ToolSpec>,
  ): ToolCallBatch[] {
    const batches: ToolCallBatch[] = [];
    let currentReadonly: PendingToolCall[] = [];

    const flushReadonly = (): void => {
      if (currentReadonly.length > 0) {
        batches.push({ parallel: true, calls: currentReadonly });
        currentReadonly = [];
      }
    };

    for (const tc of toolCalls) {
      const isAvailable = availableTools.some((t) => t.name === tc.name);
      if (!isAvailable) {
        // Unknown tool — flush readonly batch, add as sequential
        flushReadonly();
        batches.push({ parallel: false, calls: [tc] });
        continue;
      }

      const tool = this.tools.get(tc.name);
      if (tool.category === "readonly") {
        currentReadonly.push(tc);
      } else {
        // Mutating/workflow/external — flush readonly, add as sequential
        flushReadonly();
        batches.push({ parallel: false, calls: [tc] });
      }
    }

    flushReadonly();
    return batches;
  }

  private async streamLLMResponse(
    tools: ReadonlyArray<ToolSpec>,
  ): Promise<{ textContent: string; toolCalls: PendingToolCall[] }> {
    let textContent = "";
    const toolCalls: PendingToolCall[] = [];
    const pendingToolArgs = new Map<string, { name: string; chunks: string[] }>();

    const stream = this.provider.chat(this.messages, tools);

    for await (const chunk of stream) {
      switch (chunk.type) {
        case "text":
          textContent += chunk.content;
          this.bus.emit("message:assistant", {
            content: chunk.content,
            partial: true,
            chunk,
          });
          break;

        case "tool_call": {
          // Tool call comes as a single chunk with full args
          const args = parseToolArgs(chunk.content);
          toolCalls.push({
            name: chunk.toolName ?? "",
            arguments: args,
            callId: chunk.toolCallId ?? `call_${toolCalls.length}`,
          });
          break;
        }

        case "done":
          break;
      }
    }

    return { textContent, toolCalls };
  }

  private async executeToolCall(
    toolCall: PendingToolCall,
    availableTools: ReadonlyArray<ToolSpec>,
  ): Promise<ToolResult> {
    const callId = toolCall.callId;

    // Check tool exists and is available in current mode
    const isAvailable = availableTools.some((t) => t.name === toolCall.name);
    if (!isAvailable) {
      return {
        success: false,
        output: "",
        error: `Unknown tool: ${toolCall.name}`,
        artifacts: [],
      };
    }

    const tool = this.tools.get(toolCall.name);

    // Fire tool:before event
    this.bus.emit("tool:before", {
      name: toolCall.name,
      params: toolCall.arguments,
      callId,
    });

    // Check approval
    const approvalResult = await this.approvalGate.check({
      toolName: toolCall.name,
      toolCategory: tool.category,
      filePath: (toolCall.arguments["path"] as string) ?? null,
      description: `${toolCall.name}: ${JSON.stringify(toolCall.arguments).substring(0, 200)}`,
    });

    if (!approvalResult.approved) {
      const result: ToolResult = {
        success: false,
        output: "",
        error: `Tool execution denied: ${approvalResult.reason}`,
        artifacts: [],
      };
      this.bus.emit("tool:after", {
        name: toolCall.name,
        result,
        callId,
        durationMs: 0,
      });
      return result;
    }

    // Execute tool — fail fast
    const startTime = Date.now();
    let result: ToolResult;
    try {
      result = await tool.handler(toolCall.arguments, {
        repoRoot: this.repoRoot,
        config: this.config,
        sessionId: "", // Filled by engine wrapper
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      result = {
        success: false,
        output: "",
        error: message,
        artifacts: [],
      };
    }

    const durationMs = Date.now() - startTime;

    // Track recent tool calls for doom loop + tool fatigue detection
    if (result.success) {
      // Success resets doom loop tracking — the LLM found a working approach
      this.recentToolCalls = [];
      this.doomLoopWarned = false;
      // Reset fatigue counter for this specific tool
      this.toolFailureCounts.delete(toolCall.name);
      this.toolFatigueWarned.delete(toolCall.name);
    } else {
      const argsKey = JSON.stringify(toolCall.arguments);
      this.recentToolCalls.push({ name: toolCall.name, argsKey });
      if (this.recentToolCalls.length > DOOM_LOOP_THRESHOLD) {
        this.recentToolCalls.shift();
      }
      // Increment per-tool failure count (regardless of args)
      const prevCount = this.toolFailureCounts.get(toolCall.name) ?? 0;
      this.toolFailureCounts.set(toolCall.name, prevCount + 1);
    }

    // Fire tool:after event
    this.bus.emit("tool:after", {
      name: toolCall.name,
      result,
      callId,
      durationMs,
    });

    // Checkpoint + double-check for successful mutating tools
    if (result.success && tool.category === "mutating") {
      // Create checkpoint snapshot
      this.checkpointManager?.create(
        `${toolCall.name}: ${(toolCall.arguments["path"] as string) ?? ""}`,
        toolCall.name,
      );

      // Validate the edit with diagnostics/tests — inline with tool result
      if (this.doubleCheck?.isEnabled()) {
        const modifiedFiles = result.artifacts
          .filter((a): a is string => typeof a === "string");
        if (modifiedFiles.length > 0) {
          const checkResult = await this.doubleCheck.check(modifiedFiles);
          if (!checkResult.passed) {
            this.unresolvedDoubleCheckFailure = true;
            const feedback = this.doubleCheck.formatResults(checkResult);
            // Append validation errors inline with tool output (OpenCode pattern)
            result = {
              ...result,
              output: `${result.output}\n\nVALIDATION ERRORS:\n${feedback}\nFix these errors before continuing.`,
            };
          } else {
            this.unresolvedDoubleCheckFailure = false;
          }
        }
      }
    }

    return result;
  }

  /**
   * Doom loop detection: check if the LLM keeps calling the same tool
   * with identical arguments and it keeps failing.
   * Returns a warning message to inject, or null if no doom loop detected.
   *
   * Following the OpenCode pattern (DOOM_LOOP_THRESHOLD = 3):
   * - Does NOT kill the loop — the LLM gets to try a different approach
   * - Warning is injected once per doom loop pattern
   * - Resets when any tool call succeeds (see executeToolCall)
   */
  private checkDoomLoop(toolCalls: ReadonlyArray<PendingToolCall>): string | null {
    if (this.recentToolCalls.length < DOOM_LOOP_THRESHOLD) return null;

    // Check if all recent calls are identical (same tool + same args)
    const first = this.recentToolCalls[0]!;
    const isDoomLoop = this.recentToolCalls.every(
      (tc) => tc.name === first.name && tc.argsKey === first.argsKey,
    );

    if (!isDoomLoop) return null;
    if (this.doomLoopWarned) return null; // Only warn once per pattern

    this.doomLoopWarned = true;
    const toolName = first.name;

    this.bus.emit("error", {
      message: `Doom loop detected: "${toolName}" called ${DOOM_LOOP_THRESHOLD} times with identical arguments and keeps failing.`,
      code: "DOOM_LOOP",
      fatal: false,
    });

    return `WARNING: You have called "${toolName}" ${DOOM_LOOP_THRESHOLD} times with the exact same arguments, and it keeps failing. This approach is not working. Try a completely different strategy — change the command arguments, use a different tool, or modify your approach. Do NOT repeat the same failing call.`;
  }

  /**
   * Tool fatigue detection: same tool keeps failing with different arguments.
   * Complementary to doom loop (which catches identical args).
   * Returns an escalated warning, or null.
   */
  private checkToolFatigue(
    toolCalls: ReadonlyArray<PendingToolCall>,
  ): string | null {
    for (const tc of toolCalls) {
      const count = this.toolFailureCounts.get(tc.name) ?? 0;
      if (
        count >= TOOL_FATIGUE_THRESHOLD &&
        !this.toolFatigueWarned.has(tc.name)
      ) {
        this.toolFatigueWarned.add(tc.name);

        this.bus.emit("error", {
          message: `Tool fatigue: "${tc.name}" has failed ${count} times consecutively with different arguments.`,
          code: "TOOL_FATIGUE",
          fatal: false,
        });

        return `ESCALATED WARNING: The tool "${tc.name}" has failed ${count} consecutive times, even with different arguments. This tool is not working for the current task. You MUST try a fundamentally different approach: use a different tool, break the problem into smaller steps, or ask the user for guidance. Do NOT call "${tc.name}" again unless you have resolved the underlying issue.`;
      }
    }
    return null;
  }

  /**
   * Extract lessons from this session into persistent memory.
   * Heuristic-based — no extra LLM call. Called once at the end of run().
   */
  private extractLessons(): void {
    if (!this.memoryStore) return;

    // NOTE: Decay + prune now run at startup via runMaintenance(), not per-turn.

    // Extract lessons from doom loop warnings
    const doomMessages = this.messages.filter(
      (m) =>
        m.role === MessageRole.SYSTEM &&
        m.content?.includes("same arguments"),
    );
    for (const msg of doomMessages) {
      const match = msg.content?.match(/called "([^"]+)"/);
      if (match?.[1]) {
        this.memoryStore.store(
          "mistake",
          `doom-loop-${match[1]}`,
          `Tool "${match[1]}" was called repeatedly with identical failing arguments. Try different approaches or verify preconditions first.`,
          { tags: ["doom-loop", match[1]] },
        );
      }
    }

    // Extract lessons from budget exhaustion
    if (this.messages.some(
      (m) => m.role === MessageRole.SYSTEM && m.content?.includes("iteration limit"),
    )) {
      this.memoryStore.store(
        "mistake",
        "budget-exhausted",
        "Session hit the iteration limit. Break complex tasks into smaller steps and verify progress frequently.",
        { tags: ["budget-exhausted"] },
      );
    }
  }
}

// ─── Doom Loop Detection ─────────────────────────────────────

/** Number of identical failing tool calls (same name + same args) that triggers a doom loop warning. */
const DOOM_LOOP_THRESHOLD = 3;

/** Number of consecutive failures of the same tool (regardless of args) that triggers a "tool fatigue" warning. */
const TOOL_FATIGUE_THRESHOLD = 5;

// ─── Midpoint Briefing ──────────────────────────────────────

/** Default iterations between midpoint briefing checkpoints. */
const DEFAULT_MIDPOINT_INTERVAL = 15;

// ─── Retry Constants ─────────────────────────────────────────

/** Delay (ms) before each retry attempt. Length = max retries. */
const RETRY_DELAYS = [300, 900, 1800] as const;

/** Total attempts = 1 initial + retries. */
const MAX_RETRY_ATTEMPTS = RETRY_DELAYS.length + 1;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ─── Helpers ────────────────────────────────────────────────

function parseToolArgs(content: string): Record<string, unknown> {
  try {
    const parsed = JSON.parse(content);
    // JSON.parse can return non-objects (null, arrays, primitives) — wrap them
    if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
      return { raw: content };
    }
    return parsed as Record<string, unknown>;
  } catch {
    // Fail fast: surface the malformed JSON so the LLM can see the real error
    // instead of silently dropping structured arguments.
    return { _parseError: `Malformed tool arguments (invalid JSON): ${content.substring(0, 200)}` };
  }
}
