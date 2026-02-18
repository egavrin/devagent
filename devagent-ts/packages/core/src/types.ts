/**
 * Core type definitions for DevAgent.
 * ArkTS-compatible: no `any`, explicit types, interfaces only for objects.
 */

// ─── Agent Types ──────────────────────────────────────────────

export enum AgentStatus {
  IDLE = "idle",
  RUNNING = "running",
  COMPLETED = "completed",
  FAILED = "failed",
  BLOCKED = "blocked",
}

export enum AgentType {
  GENERAL = "general",
  REVIEWER = "reviewer",
  ARCHITECT = "architect",
}

export interface AgentContext {
  readonly sessionId: string;
  readonly parentId: string | null;
  readonly workingDirectory: string;
  readonly environment: Record<string, string>;
  readonly metadata: Record<string, unknown>;
}

export interface AgentMessage {
  readonly agentName: string;
  readonly content: string;
  readonly messageType: "info" | "warning" | "error" | "success";
  readonly timestamp: number;
}

export interface AgentResult {
  readonly success: boolean;
  readonly output: string;
  readonly error: string | null;
  readonly toolCalls: ReadonlyArray<ToolCallRecord>;
  readonly messages: ReadonlyArray<AgentMessage>;
  readonly cost: CostRecord;
}

// ─── Tool Types ───────────────────────────────────────────────

export interface ToolSpec {
  readonly name: string;
  readonly description: string;
  readonly category: ToolCategory;
  readonly paramSchema: JsonSchema;
  readonly resultSchema: JsonSchema;
  readonly handler: ToolHandler;
}

export type ToolCategory = "readonly" | "mutating" | "workflow" | "external";

export type ToolHandler = (
  params: Record<string, unknown>,
  context: ToolContext,
) => Promise<ToolResult>;

export interface ToolContext {
  readonly repoRoot: string;
  readonly config: DevAgentConfig;
  readonly sessionId: string;
}

export interface ToolResult {
  readonly success: boolean;
  readonly output: string;
  readonly error: string | null;
  readonly artifacts: ReadonlyArray<string>;
}

export interface ToolCallRecord {
  readonly name: string;
  readonly arguments: Record<string, unknown>;
  readonly callId: string;
  readonly result: ToolResult | null;
}

// ─── LLM Message Types ───────────────────────────────────────

export enum MessageRole {
  SYSTEM = "system",
  USER = "user",
  ASSISTANT = "assistant",
  TOOL = "tool",
}

export interface Message {
  readonly role: MessageRole;
  readonly content: string | null;
  readonly toolCallId?: string;
  readonly toolCalls?: ReadonlyArray<ToolCallRequest>;
}

export interface ToolCallRequest {
  readonly name: string;
  readonly arguments: Record<string, unknown>;
  readonly callId: string;
}

export interface StreamChunk {
  readonly type: "text" | "tool_call" | "thinking" | "error" | "done";
  readonly content: string;
  readonly toolCallId?: string;
  readonly toolName?: string;
}

// ─── Provider Types ───────────────────────────────────────────

export interface LLMProvider {
  readonly id: string;
  chat(
    messages: ReadonlyArray<Message>,
    tools?: ReadonlyArray<ToolSpec>,
  ): AsyncIterable<StreamChunk>;
  abort(): void;
}

/**
 * Model capability flags — explicitly configured per-model via TOML or config.
 * No heuristics: if not set, safe defaults apply (Chat Completions, temperature
 * enabled, 4096 output tokens).
 *
 * Example TOML:
 *   [providers.openai]
 *   model = "gpt-5.2-codex"
 *   api_key = "env:OPENAI_API_KEY"
 *   use_responses_api = true        # Codex models require Responses API
 *   reasoning = true                # Reasoning model
 *   supports_temperature = false    # Reasoning models reject temperature
 *   default_max_tokens = 128000     # GPT-5.2 family supports 128K output
 */
export interface ModelCapabilities {
  /** Use OpenAI Responses API (v1/responses) instead of Chat Completions.
   *  Required for codex models; Chat Completions deprecated for codex (Feb 2026). */
  readonly useResponsesApi?: boolean;
  /** Model supports extended reasoning (codex, o-series, gpt-5 family). */
  readonly reasoning?: boolean;
  /** Model accepts temperature parameter.
   *  false for reasoning models (gpt-5, o3, o4-mini, codex).
   *  gpt-5.2 with reasoning_effort="none" supports temperature. */
  readonly supportsTemperature?: boolean;
  /** Default max output tokens when not explicitly set.
   *  GPT-5.2 family: 128000, o3/o4-mini: 16384, standard: 4096. */
  readonly defaultMaxTokens?: number;
}

export interface ProviderConfig {
  readonly apiKey?: string;
  /** OAuth Bearer token (from browser/device-code login). */
  readonly oauthToken?: string;
  /** OAuth account ID (e.g., ChatGPT org ID for subscription routing). */
  readonly oauthAccountId?: string;
  readonly baseUrl?: string;
  readonly model: string;
  readonly maxTokens?: number;
  readonly temperature?: number;
  /** Reasoning effort: none, low, medium, high, xhigh (model-dependent). */
  readonly reasoningEffort?: "low" | "medium" | "high";
  readonly capabilities?: ModelCapabilities;
  /** ChatGPT Codex-specific options (store, include, instructions). */
  readonly codexOptions?: {
    readonly store?: boolean;
    readonly include?: ReadonlyArray<string>;
    readonly instructions?: string;
  };
  /** Custom headers to send with every API request. Used by Copilot provider. */
  readonly customHeaders?: Record<string, string>;
  /** Fields to strip from request bodies (Copilot rejects store, metadata, etc.). */
  readonly stripFields?: ReadonlyArray<string>;
}

// ─── Approval Types ──────────────────────────────────────────

export enum ApprovalMode {
  SUGGEST = "suggest",
  AUTO_EDIT = "auto-edit",
  FULL_AUTO = "full-auto",
}

export interface ApprovalPolicy {
  readonly mode: ApprovalMode;
  readonly autoApprovePlan: boolean;
  readonly autoApproveCode: boolean;
  readonly autoApproveShell: boolean;
  readonly auditLog: boolean;
  readonly toolOverrides: Record<string, "allow" | "deny" | "ask">;
  readonly pathRules: ReadonlyArray<PathRule>;
}

export interface PathRule {
  readonly pattern: string;
  readonly action: "allow" | "deny" | "ask";
}

// ─── Configuration ───────────────────────────────────────────

export interface DevAgentConfig {
  readonly provider: string;
  readonly model: string;
  readonly providers: Record<string, ProviderConfig>;
  readonly approval: ApprovalPolicy;
  readonly budget: BudgetConfig;
  readonly context: ContextConfig;
  readonly arkts: ArkTSConfig;
}

export interface BudgetConfig {
  readonly maxIterations: number;
  readonly maxContextTokens: number;
  readonly responseHeadroom: number;
  readonly costWarningThreshold: number;
  readonly enableCostTracking: boolean;
}

export interface ContextConfig {
  readonly pruningStrategy: "sliding_window" | "summarize" | "hybrid";
  readonly triggerRatio: number;
  readonly keepRecentMessages: number;
}

export interface ArkTSConfig {
  readonly enabled: boolean;
  readonly strictMode: boolean;
  readonly targetVersion: string;
}

// ─── Session Types ───────────────────────────────────────────

export interface Session {
  readonly id: string;
  readonly createdAt: number;
  readonly updatedAt: number;
  readonly messages: ReadonlyArray<Message>;
  readonly metadata: Record<string, unknown>;
}

// ─── Cost Tracking ───────────────────────────────────────────

export interface CostRecord {
  readonly inputTokens: number;
  readonly outputTokens: number;
  readonly cacheReadTokens: number;
  readonly cacheWriteTokens: number;
  readonly totalCost: number;
}

// ─── Utility Types ───────────────────────────────────────────

export interface JsonSchema {
  readonly type: string;
  readonly properties?: Record<string, unknown>;
  readonly required?: ReadonlyArray<string>;
  readonly description?: string;
}

// ─── Task Types ──────────────────────────────────────────────

export enum TaskStatus {
  PENDING = "pending",
  RUNNING = "running",
  DONE = "done",
  FAILED = "failed",
}

export interface TaskStep {
  readonly id: string;
  readonly description: string;
  readonly agentType: AgentType;
  readonly status: TaskStatus;
  readonly result: AgentResult | null;
  readonly cost: CostRecord;
}
