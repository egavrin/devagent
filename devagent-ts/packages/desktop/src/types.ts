/**
 * Desktop app types — shared across components.
 */

// ─── Chat ──────────────────────────────────────────────────

export interface ChatMessage {
  readonly id: string;
  readonly role: "user" | "assistant" | "system" | "tool";
  readonly content: string;
  readonly timestamp: number;
  readonly isStreaming?: boolean;
  readonly toolName?: string;
  readonly toolCallId?: string;
}

// ─── App State ─────────────────────────────────────────────

export type AppMode = "plan" | "act";

export type AppView = "chat" | "diff" | "settings" | "skills" | "mcp" | "memory";

// ─── Tools ─────────────────────────────────────────────────

export interface ToolExecution {
  readonly id: string;
  readonly name: string;
  readonly params: Record<string, unknown>;
  readonly status: "pending" | "running" | "done" | "error";
  readonly result?: string;
  readonly error?: string;
  readonly durationMs?: number;
  readonly timestamp: number;
}

// ─── Approval ──────────────────────────────────────────────

export interface ApprovalRequest {
  readonly id: string;
  readonly toolName: string;
  readonly details: string;
  readonly timestamp: number;
}

// ─── Diffs ─────────────────────────────────────────────────

export interface FileDiff {
  readonly id: string;
  readonly filePath: string;
  readonly hunks: ReadonlyArray<DiffHunk>;
  readonly status: "pending" | "accepted" | "rejected";
  readonly toolCallId: string;
  readonly timestamp: number;
}

export interface DiffHunk {
  readonly header: string;
  readonly oldStart: number;
  readonly newStart: number;
  readonly oldCount: number;
  readonly newCount: number;
  readonly lines: ReadonlyArray<DiffLine>;
}

export interface DiffLine {
  readonly type: "add" | "remove" | "context";
  readonly content: string;
  readonly oldLineNumber?: number;
  readonly newLineNumber?: number;
}

// ─── Skills ────────────────────────────────────────────────

export interface SkillInfo {
  readonly name: string;
  readonly description: string;
  readonly source: "project" | "global";
}

// ─── MCP ───────────────────────────────────────────────────

export interface McpServerInfo {
  readonly name: string;
  readonly status: "running" | "stopped" | "error";
  readonly toolCount: number;
  readonly tools: ReadonlyArray<McpToolInfo>;
  readonly error?: string;
}

export interface McpToolInfo {
  readonly name: string;
  readonly description: string;
}

// ─── Memory ────────────────────────────────────────────────

export type MemoryCategory = "pattern" | "decision" | "mistake" | "preference" | "context";

export interface MemoryEntry {
  readonly id: string;
  readonly category: MemoryCategory;
  readonly key: string;
  readonly content: string;
  readonly relevance: number;
  readonly tags: ReadonlyArray<string>;
  readonly updatedAt: number;
  readonly accessCount: number;
}

// ─── Commands ──────────────────────────────────────────────

export interface CommandInfo {
  readonly name: string;
  readonly description: string;
  readonly plugin: string;
  readonly usage?: string;
}

// ─── Cost ──────────────────────────────────────────────────

export interface CostState {
  readonly inputTokens: number;
  readonly outputTokens: number;
  readonly totalCost: number;
}

// ─── Provider / Settings ───────────────────────────────────

export interface ProviderInfo {
  readonly id: string;
  readonly name: string;
  readonly models: ReadonlyArray<string>;
}

export interface AppSettings {
  readonly provider: string;
  readonly model: string;
  readonly apiKey: string;
  readonly approvalMode: "suggest" | "auto-edit" | "full-auto";
  readonly maxIterations: number;
  readonly theme: "light" | "dark" | "system";
}
