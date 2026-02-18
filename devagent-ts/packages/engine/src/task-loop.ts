/**
 * ReAct task loop — the core execution engine.
 * Streams LLM responses, parses tool calls, checks approval,
 * executes tools, feeds results back to LLM.
 * Fail fast: tool errors surface to LLM immediately.
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
  BudgetExceededError,
  ProviderError,
} from "@devagent/core";
import type { ToolRegistry } from "@devagent/tools";

// ─── Types ──────────────────────────────────────────────────

export type TaskMode = "plan" | "act";

export interface TaskLoopOptions {
  readonly provider: LLMProvider;
  readonly tools: ToolRegistry;
  readonly bus: EventBus;
  readonly approvalGate: ApprovalGate;
  readonly config: DevAgentConfig;
  readonly systemPrompt: string;
  readonly repoRoot: string;
  readonly mode?: TaskMode;
}

export interface TaskLoopResult {
  readonly messages: ReadonlyArray<Message>;
  readonly iterations: number;
  readonly cost: CostRecord;
  readonly aborted: boolean;
}

interface PendingToolCall {
  readonly name: string;
  readonly arguments: Record<string, unknown>;
  readonly callId: string;
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
  private consecutiveFailures = 0;

  constructor(options: TaskLoopOptions) {
    this.provider = options.provider;
    this.tools = options.tools;
    this.bus = options.bus;
    this.approvalGate = options.approvalGate;
    this.config = options.config;
    this.systemPrompt = options.systemPrompt;
    this.repoRoot = options.repoRoot;
    this.mode = options.mode ?? "act";

    // Add system prompt as first message
    this.messages.push({
      role: MessageRole.SYSTEM,
      content: this.systemPrompt,
    });
  }

  /**
   * Run the task loop with a user query.
   * Returns when the LLM produces a final text response (no more tool calls)
   * or when the budget is exceeded.
   */
  async run(userQuery: string): Promise<TaskLoopResult> {
    // Add user message
    this.messages.push({
      role: MessageRole.USER,
      content: userQuery,
    });
    this.bus.emit("message:user", { content: userQuery });

    while (!this.aborted) {
      // Check budget (0 = unlimited)
      if (this.config.budget.maxIterations > 0 && this.iterations >= this.config.budget.maxIterations) {
        throw new BudgetExceededError(
          `Max iterations (${this.config.budget.maxIterations}) exceeded`,
        );
      }

      this.iterations++;

      // Get available tools based on mode
      const availableTools = this.getAvailableTools();

      // Stream LLM response with retry on transient provider errors
      let textContent = "";
      let toolCalls: PendingToolCall[] = [];
      for (let attempt = 0; attempt < MAX_RETRY_ATTEMPTS; attempt++) {
        try {
          const result = await this.streamLLMResponse(availableTools);
          textContent = result.textContent;
          toolCalls = result.toolCalls;
          break;
        } catch (err) {
          if (!(err instanceof ProviderError)) throw err;
          if (attempt >= RETRY_DELAYS.length) throw err; // Exhausted retries
          this.bus.emit("error", {
            message: `Provider error (attempt ${attempt + 1}/${MAX_RETRY_ATTEMPTS}): ${(err as Error).message}. Retrying in ${RETRY_DELAYS[attempt]!}ms…`,
            code: "PROVIDER_RETRY",
            fatal: false,
          });
          await sleep(RETRY_DELAYS[attempt]!);
        }
      }

      if (toolCalls.length > 0) {
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

        // Execute each tool call
        for (const toolCall of toolCalls) {
          if (this.aborted) break;
          const result = await this.executeToolCall(toolCall, availableTools);

          // Add tool result message
          this.messages.push({
            role: MessageRole.TOOL,
            content: result.success
              ? result.output
              : `Error: ${result.error}`,
            toolCallId: toolCall.callId,
          });
        }

        // Inject failure warning if needed
        const failureWarning = this.getFailureWarning();
        if (failureWarning) {
          this.messages.push({
            role: MessageRole.SYSTEM,
            content: failureWarning,
          });
          if (this.consecutiveFailures >= 5) {
            break;
          }
        }

        // Continue loop — feed tool results back to LLM
        continue;
      }

      // No tool calls — LLM produced a final text response
      if (textContent) {
        this.messages.push({
          role: MessageRole.ASSISTANT,
          content: textContent,
        });
        this.bus.emit("message:assistant", {
          content: textContent,
          partial: false,
        });
      }

      break;
    }

    return {
      messages: this.messages,
      iterations: this.iterations,
      cost: this.totalCost,
      aborted: this.aborted,
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
    this.consecutiveFailures = 0;
  }

  // ─── Private ────────────────────────────────────────────────

  private getAvailableTools(): ReadonlyArray<ToolSpec> {
    if (this.mode === "plan") {
      return this.tools.getReadOnly();
    }
    return this.tools.getAll();
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

    // Track consecutive failures
    if (result.success) {
      this.consecutiveFailures = 0;
    } else {
      this.consecutiveFailures++;
    }

    // Fire tool:after event
    this.bus.emit("tool:after", {
      name: toolCall.name,
      result,
      callId,
      durationMs,
    });

    return result;
  }

  /**
   * Check if a failure warning should be injected after tool results.
   * Returns a warning message if 3+ consecutive failures, null otherwise.
   * At 5+ failures, the loop should break.
   */
  private getFailureWarning(): string | null {
    if (this.consecutiveFailures >= 5) {
      return "CRITICAL: 5 consecutive tool failures. Stopping execution. Report the issue to the user and suggest a different approach.";
    }
    if (this.consecutiveFailures >= 3) {
      return "Warning: 3 consecutive tool failures. Consider a different approach or ask the user for guidance.";
    }
    return null;
  }
}

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
    return JSON.parse(content) as Record<string, unknown>;
  } catch {
    return { raw: content };
  }
}
