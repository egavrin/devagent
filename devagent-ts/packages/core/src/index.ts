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
  ModelCapabilities,
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
  PlanUpdatedEvent,
  ErrorEvent,
} from "./events.js";

// Config
export { loadConfig, findProjectRoot, resolveProviderCredentials } from "./config.js";

// Model registry
export {
  loadModelRegistry,
  lookupModelCapabilities,
  lookupModelEntry,
  getRegisteredModels,
} from "./model-registry.js";
export type { ModelRegistryEntry } from "./model-registry.js";

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

// Plugins
export { PluginManager } from "./plugins.js";
export type {
  Plugin,
  PluginContext,
  CommandHandler,
} from "./plugins.js";

// Skills
export { SkillRegistry } from "./skills.js";
export type {
  Skill,
  SkillMetadata,
} from "./skills.js";

// Context management
export { ContextManager, estimateTokens, estimateMessageTokens } from "./context.js";
export type {
  ContextTruncationResult,
  SummarizeCallback,
} from "./context.js";

// Memory (cross-session learning)
export { MemoryStore } from "./memory.js";
export type {
  Memory,
  MemoryCategory,
  MemoryStoreOptions,
  MemorySearchOptions,
} from "./memory.js";

// Credentials (persistent API key and OAuth token storage)
export { CredentialStore } from "./credentials.js";
export type {
  Credential,
  ApiCredential,
  OAuthCredential,
  CredentialInfo,
  CredentialStoreOptions,
} from "./credentials.js";

// OAuth primitives
export {
  generatePKCE,
  generateState,
  startCallbackServer,
  exchangeCodeForTokens,
  refreshAccessToken,
  requestDeviceCode,
  pollDeviceCodeToken,
  requestChatGPTDeviceCode,
  pollChatGPTDeviceAuth,
  exchangeChatGPTDeviceToken,
  exchangeCopilotSessionToken,
  extractAccountIdFromIdToken,
} from "./oauth.js";
export type {
  PKCEPair,
  TokenResponse,
  DeviceCodeResponse,
  ChatGPTDeviceCodeResponse,
  CopilotSessionToken,
  CallbackResult,
  CallbackServer,
} from "./oauth.js";

// OAuth provider configs
export { getOAuthProvider, OAUTH_PROVIDERS } from "./oauth-providers.js";
export type { OAuthProviderConfig } from "./oauth-providers.js";

// Browser URL opener
export { openUrl } from "./open-url.js";

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
  CredentialError,
  OAuthError,
} from "./errors.js";
