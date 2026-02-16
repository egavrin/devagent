/**
 * Error hierarchy for DevAgent.
 * Fail fast: all errors are explicit, typed, and surfaced immediately.
 */

export class DevAgentError extends Error {
  readonly code: string;

  constructor(message: string, code: string) {
    super(message);
    this.name = "DevAgentError";
    this.code = code;
  }
}

// ─── Config Errors ───────────────────────────────────────────

export class ConfigError extends DevAgentError {
  constructor(message: string) {
    super(message, "CONFIG_ERROR");
    this.name = "ConfigError";
  }
}

export class ConfigNotFoundError extends ConfigError {
  constructor(path: string) {
    super(`Config file not found: ${path}`);
    this.name = "ConfigNotFoundError";
  }
}

// ─── Provider Errors ─────────────────────────────────────────

export class ProviderError extends DevAgentError {
  constructor(message: string, code: string = "PROVIDER_ERROR") {
    super(message, code);
    this.name = "ProviderError";
  }
}

export class RateLimitError extends ProviderError {
  readonly retryAfterMs: number | null;

  constructor(message: string, retryAfterMs: number | null = null) {
    super(message, "RATE_LIMIT");
    this.name = "RateLimitError";
    this.retryAfterMs = retryAfterMs;
  }
}

export class ProviderTimeoutError extends ProviderError {
  constructor(message: string) {
    super(message, "PROVIDER_TIMEOUT");
    this.name = "ProviderTimeoutError";
  }
}

export class ProviderConnectionError extends ProviderError {
  constructor(message: string) {
    super(message, "PROVIDER_CONNECTION");
    this.name = "ProviderConnectionError";
  }
}

// ─── Tool Errors ─────────────────────────────────────────────

export class ToolError extends DevAgentError {
  readonly toolName: string;

  constructor(toolName: string, message: string) {
    super(`Tool "${toolName}": ${message}`, "TOOL_ERROR");
    this.name = "ToolError";
    this.toolName = toolName;
  }
}

export class ToolNotFoundError extends ToolError {
  constructor(toolName: string) {
    super(toolName, "not found in registry");
    this.name = "ToolNotFoundError";
  }
}

export class ToolValidationError extends ToolError {
  constructor(toolName: string, message: string) {
    super(toolName, `validation failed: ${message}`);
    this.name = "ToolValidationError";
  }
}

// ─── Approval Errors ─────────────────────────────────────────

export class ApprovalDeniedError extends DevAgentError {
  readonly toolName: string;

  constructor(toolName: string, action: string) {
    super(
      `Approval denied for "${toolName}": ${action}`,
      "APPROVAL_DENIED",
    );
    this.name = "ApprovalDeniedError";
    this.toolName = toolName;
  }
}

// ─── Budget Errors ───────────────────────────────────────────

export class BudgetExceededError extends DevAgentError {
  constructor(message: string) {
    super(message, "BUDGET_EXCEEDED");
    this.name = "BudgetExceededError";
  }
}

// ─── Session Errors ──────────────────────────────────────────

export class SessionError extends DevAgentError {
  constructor(message: string) {
    super(message, "SESSION_ERROR");
    this.name = "SessionError";
  }
}
