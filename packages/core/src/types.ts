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

export type ToolCategory = "readonly" | "mutating" | "workflow" | "external" | "state";

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
  /** When true, this message survives context compaction (e.g., critical git diffs). */
  readonly pinned?: boolean;
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
  /** Token usage reported by the provider on "done" chunks. */
  readonly usage?: {
    readonly promptTokens: number;
    readonly completionTokens: number;
  };
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
  readonly memory: MemoryConfig;
  readonly arkts: ArkTSConfig;
  readonly logging?: LoggingConfig;
  readonly checkpoints?: CheckpointConfig;
  readonly doubleCheck?: DoubleCheckConfig;
  readonly lsp?: LSPConfig;
  readonly sessionState?: SessionStateConfigCore;
}

/** Config for session state persistence and tracking (core-level definition). */
export interface SessionStateConfigCore {
  readonly persist?: boolean;
  readonly trackPlan?: boolean;
  readonly trackFiles?: boolean;
  readonly trackEnv?: boolean;
  readonly trackToolResults?: boolean;
  readonly trackFindings?: boolean;
  readonly maxModifiedFiles?: number;
  readonly maxEnvFacts?: number;
  readonly maxToolSummaries?: number;
  readonly maxFindings?: number;
}

export interface CheckpointConfig {
  readonly enabled: boolean;
}

export interface DoubleCheckConfig {
  readonly enabled: boolean;
  readonly checkDiagnostics?: boolean;
  readonly runTests?: boolean;
  readonly testCommand?: string | null;
  readonly diagnosticTimeout?: number;
}

export interface LSPServerConfig {
  /** LSP server command (e.g., "typescript-language-server"). */
  readonly command: string;
  /** Command arguments (e.g., ["--stdio"]). */
  readonly args: ReadonlyArray<string>;
  /** Language IDs this server handles (e.g., ["typescript", "javascript"]). */
  readonly languages: ReadonlyArray<string>;
  /** File extensions this server handles (e.g., [".ts", ".tsx", ".js", ".jsx"]). */
  readonly extensions: ReadonlyArray<string>;
  /** Timeout for LSP requests in ms. Default: 10000. */
  readonly timeout: number;
  /** How long to wait for diagnostics (pushed async) in ms. Default: 3000.
   *  rust-analyzer needs 15-30s (delegates to `cargo check`). */
  readonly diagnosticTimeout?: number;
}

export interface LSPConfig {
  /** LSP server configurations. Each entry handles one or more languages. */
  readonly servers: ReadonlyArray<LSPServerConfig>;
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
  /** Enable turn isolation in interactive mode (fresh TaskLoop per turn). Default: true. */
  readonly turnIsolation?: boolean;
  /** Iterations between midpoint briefing checkpoints (0 = disabled). Default: 15. */
  readonly midpointBriefingInterval?: number;
  /** Strategy for turn briefing synthesis. Default: "auto". */
  readonly briefingStrategy?: "heuristic" | "llm" | "auto";
  /** Tokens of recent tool output protected from Phase-1 pruning. Default: 60000. */
  readonly pruneProtectTokens?: number;
}

export interface MemoryConfig {
  /** Enable cross-session memory system. Default: true. */
  readonly enabled: boolean;
  /** How much relevance decays per day without access. Default: 0.02. */
  readonly dailyDecay: number;
  /** Minimum relevance before a memory is prunable. Default: 0.1. */
  readonly minRelevance: number;
  /** Relevance boost when a memory is accessed. Default: 0.1. */
  readonly accessBoost: number;
  /** Minimum relevance threshold for recall search results. Default: 0.3. */
  readonly recallMinRelevance: number;
  /** Maximum results returned by recall search. Default: 10. */
  readonly recallLimit: number;
  /** Maximum memories injected into the system prompt. Default: 10. */
  readonly promptMaxMemories: number;
  /** Maximum character budget for memories in the system prompt. Default: 2000. */
  readonly promptMaxChars: number;
  /** Run maintenance (decay + prune + dedup) on startup. Default: true. */
  readonly maintenanceOnStartup: boolean;
}

export interface ArkTSConfig {
  readonly enabled: boolean;
  readonly strictMode: boolean;
  readonly targetVersion: string;
  /** Path to the ets2panda/linter directory (contains dist/tslinter.js after build). */
  readonly linterPath?: string;
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

// ─── Logging ─────────────────────────────────────────────────

export interface LoggingConfig {
  readonly enabled: boolean;
  readonly logDir?: string;
  readonly retentionDays?: number;
}

// ─── Verbosity ───────────────────────────────────────────────

export interface VerbosityConfig {
  readonly base: "quiet" | "normal" | "verbose";
  readonly categories: ReadonlySet<string>;
}

// ─── Utility Types ───────────────────────────────────────────

export interface JsonSchema {
  readonly type: string;
  readonly properties?: Record<string, unknown>;
  readonly required?: ReadonlyArray<string>;
  readonly description?: string;
  readonly additionalProperties?: boolean;
  readonly items?: Record<string, unknown>;
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
