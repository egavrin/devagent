/**
 * @devagent/runtime — types, config, event bus, session, approval, errors.
 */

// Types
export type {
  ToolSpec,
  ToolErrorGuidance,
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
  DoubleCheckConfig,
  LSPConfig,
  LSPServerConfig,
  Session,
  CostRecord,
  JsonSchema,
  SessionStateConfigCore,
  LoggingConfig,
  VerbosityConfig,
} from "./types.js";

export {
  AgentType,
  MessageRole,
  ApprovalMode,
} from "./types.js";

export type { ToolCategory } from "./types.js";

// Event bus
export { EventBus } from "./events.js";
export type {
  EventMap,
  ToolBeforeEvent,
  ToolAfterEvent,
  AssistantMessageEvent,
  ToolMessageEvent,
  UserMessageEvent,
  ApprovalRequestEvent,
  ApprovalResponseEvent,
  SessionStartEvent,
  SessionEndEvent,
  CostUpdateEvent,
  PlanUpdatedEvent,
  ContextCompactingEvent,
  ContextCompactedEvent,
  IterationStartEvent,
  ErrorEvent,
} from "./events.js";

// Config
export { loadConfig, findProjectRoot, resolveProviderCredentials, DEFAULT_BUDGET, DEFAULT_CONTEXT } from "./config.js";

// Model registry
export {
  loadModelRegistry,
  lookupModelCapabilities,
  lookupModelEntry,
  lookupModelPricing,
  getRegisteredModels,
} from "./model-registry.js";
export type { ModelRegistryEntry, ModelPricing } from "./model-registry.js";

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

// Skills (Agent Skills standard)
export {
  SkillRegistry,
  SkillLoader,
  SkillResolver,
  isValidSkillName,
} from "./skills/index.js";
export type {
  SkillFrontmatter,
  SkillSource,
  SkillMetadata,
  Skill,
  ResolvedSkill,
  DiscoverOptions,
  ResolveContext,
  SkillResolverOptions,
} from "./skills/index.js";

// Context management
export {
  ContextManager,
  ContextFitError,
  estimateTokens,
  estimateMessageTokens,
} from "./context.js";
export type {
  ContextTruncationResult,
  SummarizeCallback,
} from "./context.js";

// bun:sqlite availability flag (for test skipping in non-Bun environments)
export { BUN_SQLITE_AVAILABLE } from "./bun-sqlite.js";

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

// Event logger
export { EventLogger } from "./event-logger.js";
export type { LogEntry } from "./event-logger.js";

// Language extensions
export { LANGUAGE_EXTENSIONS } from "./languages.js";

// Artifact store
export { ArtifactStore } from "./artifact-store.js";
export type { ArtifactMetadata } from "./artifact-store.js";

// Workflow phase result schemas
export { WORKFLOW_SCHEMA_VERSION } from "./workflow-types.js";
export type {
  PhaseResult,
  TriageReport,
  PlanDraft,
  PlanStep as WorkflowPlanStep,
  ExecutionReport,
  VerificationReport,
  VerificationCommand,
  ReviewReport,
  ReviewFinding,
  RepairReport,
} from "./workflow-types.js";

// Repository instruction loader
export { RepositoryInstructionLoader } from "./instruction-loader.js";
export type { RepoInstruction } from "./instruction-loader.js";

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
  extractErrorMessage,
} from "./errors.js";
