import { ConfigError } from "./errors.js";
import type {
  ApprovalPolicyMode,
  BudgetConfig,
  ContextConfig,
  DevAgentConfig,
  NetworkAccessMode,
  SandboxMode,
} from "./types.js";
import { SafetyMode } from "./types.js";

export interface SafetyConfig {
  readonly mode: SafetyMode;
  readonly approvalPolicy: ApprovalPolicyMode;
  readonly sandboxMode: SandboxMode;
  readonly networkAccess: NetworkAccessMode;
}

const VALID_AGENT_TYPES = new Set<string>([
  "general",
  "reviewer",
  "architect",
  "explore",
]);

export function validateBudgetConfig(budget: BudgetConfig): void {
  validateNonNegativeInteger("budget.maxIterations", budget.maxIterations);
  validateNonNegativeNumber("budget.maxContextTokens", budget.maxContextTokens);
  validateNonNegativeNumber("budget.responseHeadroom", budget.responseHeadroom);
  validateResponseHeadroom(budget);
  validateNonNegativeNumber("budget.costWarningThreshold", budget.costWarningThreshold);
}

export function validateContextConfig(context: ContextConfig): void {
  validatePruningStrategy(context.pruningStrategy);
  validateTriggerRatio(context.triggerRatio);
  validatePositiveInteger("context.keepRecentMessages", context.keepRecentMessages);
  validateOptionalNonNegativeInteger("context.midpointBriefingInterval", context.midpointBriefingInterval);
  if (context.briefingStrategy !== undefined) validateBriefingStrategy(context.briefingStrategy);
}

export function validateSafetyConfig(safety: SafetyConfig): void {
  validateSetMember("safety.mode", safety.mode, new Set([SafetyMode.DEFAULT, SafetyMode.AUTOPILOT]));
  validateSetMember("safety.approvalPolicy", safety.approvalPolicy, new Set(["strict", "on-request", "never"]));
  validateSetMember("safety.sandboxMode", safety.sandboxMode, new Set(["read-only", "workspace-write", "danger-full-access"]));
  validateSetMember("safety.networkAccess", safety.networkAccess, new Set(["off", "on"]));
}

export function validateSubagentConfig(config: DevAgentConfig): void {
  for (const [agentType, cap] of Object.entries(config.agentIterationCaps ?? {})) {
    if (VALID_AGENT_TYPES.has(agentType) && Number.isInteger(cap) && cap >= 1) continue;
    throw new ConfigError(
      `Invalid agentIterationCaps.${agentType}: expected integer >= 1, got ${String(cap)}`,
    );
  }

  validateOptionalNonNegativeInteger("subagentTimeoutMs", config.subagentTimeoutMs);
}

function validateResponseHeadroom(budget: BudgetConfig): void {
  if (budget.maxContextTokens <= 0 || budget.responseHeadroom < budget.maxContextTokens) return;
  throw new ConfigError(
    `Invalid budget.responseHeadroom: must be < budget.maxContextTokens (${budget.responseHeadroom} >= ${budget.maxContextTokens})`,
  );
}

function validatePruningStrategy(strategy: ContextConfig["pruningStrategy"]): void {
  validateSetMember("context.pruningStrategy", strategy, new Set(["sliding_window", "summarize", "hybrid"]));
}

function validateTriggerRatio(value: number): void {
  if (Number.isFinite(value) && value > 0 && value <= 1) return;
  throw new ConfigError(
    `Invalid context.triggerRatio: expected number in (0, 1], got ${String(value)}`,
  );
}

function validatePositiveInteger(name: string, value: number): void {
  if (Number.isInteger(value) && value >= 1) return;
  throw new ConfigError(`Invalid ${name}: expected integer >= 1, got ${String(value)}`);
}

function validateOptionalNonNegativeInteger(name: string, value: number | undefined): void {
  if (value !== undefined) validateNonNegativeInteger(name, value);
}

function validateBriefingStrategy(value: NonNullable<ContextConfig["briefingStrategy"]>): void {
  validateSetMember("context.briefingStrategy", value, new Set(["heuristic", "llm", "auto"]));
}

function validateNonNegativeInteger(name: string, value: number): void {
  if (Number.isInteger(value) && value >= 0) return;
  throw new ConfigError(`Invalid ${name}: expected integer >= 0, got ${String(value)}`);
}

function validateNonNegativeNumber(name: string, value: number): void {
  if (Number.isFinite(value) && value >= 0) return;
  throw new ConfigError(`Invalid ${name}: expected number >= 0, got ${String(value)}`);
}

function validateSetMember<T>(name: string, value: T, validValues: ReadonlySet<T>): void {
  if (validValues.has(value)) return;
  throw new ConfigError(`Invalid ${name}: ${String(value)}`);
}
