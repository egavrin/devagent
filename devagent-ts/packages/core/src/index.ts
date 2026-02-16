/**
 * @devagent/core — types, config, event bus, session, approval, errors.
 */

// Types
export type {
  AgentContext,
  AgentMessage,
  AgentResult,
  ToolSpec,
  ToolContext,
  ToolResult,
  ToolHandler,
  ToolCallRecord,
  ToolCallRequest,
  Message,
  StreamChunk,
  LLMProvider,
  ProviderConfig,
  ApprovalPolicy,
  PathRule,
  DevAgentConfig,
  BudgetConfig,
  ContextConfig,
  ArkTSConfig,
  Session,
  CostRecord,
  JsonSchema,
  TaskStep,
} from "./types.js";

export {
  AgentStatus,
  AgentType,
  MessageRole,
  ApprovalMode,
  TaskStatus,
} from "./types.js";

export type { ToolCategory } from "./types.js";

// Event bus
export { EventBus } from "./events.js";
export type {
  EventMap,
  ToolBeforeEvent,
  ToolAfterEvent,
  AssistantMessageEvent,
  UserMessageEvent,
  ApprovalRequestEvent,
  ApprovalResponseEvent,
  CheckpointEvent,
  SessionStartEvent,
  SessionEndEvent,
  CostUpdateEvent,
  ErrorEvent,
} from "./events.js";

// Config
export { loadConfig, findProjectRoot } from "./config.js";

// Session
export { SessionStore } from "./session.js";
export type { SessionStoreOptions } from "./session.js";

// Approval
export { ApprovalGate } from "./approval.js";
export type {
  ApprovalDecision,
  ApprovalRequest,
  ApprovalResult,
} from "./approval.js";

// Errors
export {
  DevAgentError,
  ConfigError,
  ConfigNotFoundError,
  ProviderError,
  RateLimitError,
  ProviderTimeoutError,
  ProviderConnectionError,
  ToolError,
  ToolNotFoundError,
  ToolValidationError,
  ApprovalDeniedError,
  BudgetExceededError,
  SessionError,
} from "./errors.js";
