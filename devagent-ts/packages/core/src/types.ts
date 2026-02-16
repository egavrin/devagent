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

export interface ProviderConfig {
  readonly apiKey?: string;
  readonly baseUrl?: string;
  readonly model: string;
  readonly maxTokens?: number;
  readonly temperature?: number;
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
