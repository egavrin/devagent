/**
 * Core type definitions for DevAgent.
 * ArkTS-compatible: no `any`, explicit types, interfaces only for objects.
 */

// ─── Agent Types ──────────────────────────────────────────────

export enum AgentType {
  GENERAL = "general",
  REVIEWER = "reviewer",
  ARCHITECT = "architect",
  EXPLORE = "explore",
}

export type ReasoningEffort = "low" | "medium" | "high" | "xhigh";

export interface AgentToolPermissionOverride {
  readonly readonly?: "allow" | "deny";
  readonly mutating?: "allow" | "deny";
  readonly workflow?: "allow" | "deny";
  readonly external?: "allow" | "deny";
  readonly state?: "allow" | "deny";
}

// ─── Tool Types ───────────────────────────────────────────────

/** Recovery guidance appended to tool error messages on failure. */
export interface ToolErrorGuidance {
  /** Always appended on first failure of this tool. */
  readonly common: string;
  /** Pattern-matched hints — first substring match wins, overrides common. */
  readonly patterns?: ReadonlyArray<{
    readonly match: string;
    readonly hint: string;
  }>;
}

export interface ToolSpec {
  readonly name: string;
  readonly description: string;
  readonly category: ToolCategory;
  readonly paramSchema: JsonSchema;
  readonly resultSchema: JsonSchema;
  readonly handler: ToolHandler;
  /** Optional error recovery guidance appended to error messages on failure. */
  readonly errorGuidance?: ToolErrorGuidance;
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
  readonly callId?: string;
  readonly batchId?: string;
  readonly batchSize?: number;
}

export interface ToolResult {
  readonly success: boolean;
  readonly output: string;
  readonly error: string | null;
  readonly artifacts: ReadonlyArray<string>;
  readonly metadata?: Record<string, unknown>;
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
   *  Required for responses-api-only models such as codex. */
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
  readonly reasoningEffort?: ReasoningEffort;
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
  readonly arkts: ArkTSConfig;
  readonly logging?: LoggingConfig;
  readonly doubleCheck?: DoubleCheckConfig;
  readonly lsp?: LSPConfig;
  readonly sessionState?: SessionStateConfigCore;
  readonly agentModelOverrides?: Partial<Record<AgentType, string>>;
  readonly agentReasoningOverrides?: Partial<Record<AgentType, ReasoningEffort>>;
  readonly agentIterationCaps?: Partial<Record<AgentType, number>>;
  readonly agentPermissionOverrides?: Partial<Record<AgentType, AgentToolPermissionOverride>>;
  readonly allowedChildAgents?: Partial<Record<AgentType, ReadonlyArray<AgentType>>>;
  readonly subagentTimeoutMs?: number;
}

/** Config for session state persistence and tracking (core-level definition). */
export interface SessionStateConfigCore {
  readonly persist?: boolean;
  readonly trackPlan?: boolean;
  readonly trackFiles?: boolean;
  readonly trackEnv?: boolean;
  readonly trackToolResults?: boolean;
  readonly trackFindings?: boolean;
  readonly trackKnowledge?: boolean;
  readonly maxModifiedFiles?: number;
  readonly maxEnvFacts?: number;
  readonly maxToolSummaries?: number;
  readonly maxFindings?: number;
  readonly maxKnowledge?: number;
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
  /** Enable turn isolation when resuming persisted sessions. Default: true. */
  readonly turnIsolation?: boolean;
  /** Iterations between midpoint briefings (0 = disabled). Default: 15. */
  readonly midpointBriefingInterval?: number;
  /** Strategy for turn briefing synthesis. Default: "auto". */
  readonly briefingStrategy?: "heuristic" | "llm" | "auto";
  /** Tokens of recent tool output protected from Phase-1 pruning. Default: 60000. */
  readonly pruneProtectTokens?: number;
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
