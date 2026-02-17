/**
 * Desktop app types — shared across components.
 */

export interface ChatMessage {
  readonly id: string;
  readonly role: "user" | "assistant" | "system" | "tool";
  readonly content: string;
  readonly timestamp: number;
  readonly isStreaming?: boolean;
  readonly toolName?: string;
  readonly toolCallId?: string;
}

export type AppMode = "plan" | "act";

export type AppView = "chat" | "diff" | "settings";

export interface ToolExecution {
  readonly id: string;
  readonly name: string;
  readonly params: Record<string, unknown>;
  readonly status: "pending" | "running" | "done" | "error";
  readonly result?: string;
  readonly error?: string;
  readonly timestamp: number;
}

export interface ApprovalRequest {
  readonly id: string;
  readonly toolName: string;
  readonly details: string;
  readonly timestamp: number;
}

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
