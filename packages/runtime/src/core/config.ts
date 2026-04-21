/**
 * Configuration loading with TOML support.
 * Hierarchy: defaults → global → project → env → CLI flags.
 * Fail fast: missing required config throws, never guesses.
 */

import { readFileSync, existsSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { parse as parseToml } from "smol-toml";

import {
  parseDoubleCheckConfig,
  parseLoggingConfig,
  parseLspConfig,
  parseSessionStateConfig,
} from "./config-optional-sections.js";
import { getDefaultSubagentProfiles } from "./config-subagent-defaults.js";
import type { SafetyConfig } from "./config-validation.js";
import {
  validateBudgetConfig,
  validateContextConfig,
  validateSafetyConfig,
  validateSubagentConfig,
} from "./config-validation.js";
import { CredentialStore } from "./credentials.js";
import { ConfigError , extractErrorMessage } from "./errors.js";
import { resolveProviderCredentialStatus } from "./provider-credentials.js";
import type {
  DevAgentConfig,
  ApprovalPolicy,
  ApprovalPolicyMode,
  SandboxMode,
  NetworkAccessMode,
  BudgetConfig,
  ContextConfig,
  ProviderConfig,
  ModelCapabilities,
  AgentToolPermissionOverride,
  ReasoningEffort,
  AgentType,
} from "./types.js";
import { SafetyMode } from "./types.js";

export { resolveProviderCredentials } from "./config-credentials.js";
export { findProjectRoot } from "./project-root.js";

// ─── Defaults ────────────────────────────────────────────────

const DEFAULT_SAFETY: SafetyConfig = {
  mode: SafetyMode.AUTOPILOT,
  approvalPolicy: "never" as ApprovalPolicyMode,
  sandboxMode: "danger-full-access" as SandboxMode,
  networkAccess: "on" as NetworkAccessMode,
};

const DEFAULT_APPROVAL: ApprovalPolicy = {
  ...DEFAULT_SAFETY,
  auditLog: false,
  toolOverrides: {},
  pathRules: [],
};

export const DEFAULT_BUDGET: BudgetConfig = {
  maxIterations: 0,
  maxContextTokens: 100_000,
  responseHeadroom: 2_000,
  costWarningThreshold: 1.0,
  enableCostTracking: true,
};

export const DEFAULT_CONTEXT: ContextConfig = {
  pruningStrategy: "hybrid",
  triggerRatio: 0.9,
  keepRecentMessages: 40,
  turnIsolation: true,
  midpointBriefingInterval: 15,
  briefingStrategy: "auto",
};

const DEFAULT_CONFIG: DevAgentConfig = {
  provider: "anthropic",
  model: "claude-sonnet-4-20250514",
  providers: {},
  approval: DEFAULT_APPROVAL,
  budget: DEFAULT_BUDGET,
  context: DEFAULT_CONTEXT,
};

const VALID_AGENT_TYPES = new Set<string>([
  "general",
  "reviewer",
  "architect",
  "explore",
]);

const VALID_TOOL_CATEGORIES = new Set<string>([
  "readonly",
  "mutating",
  "workflow",
  "external",
  "state",
]);

const VALID_REASONING_EFFORTS = new Set<ReasoningEffort>([
  "low",
  "medium",
  "high",
  "xhigh",
]);

interface LoadedConfigContext {
  readonly fileConfig: Record<string, unknown>;
  readonly envModel: string | undefined;
  readonly provider: string;
  readonly providers: Record<string, ProviderConfig>;
  readonly approval: ApprovalPolicy;
  readonly budget: BudgetConfig;
  readonly context: ContextConfig;
  readonly overrides: Partial<DevAgentConfig> | undefined;
}

interface SubagentConfigSections {
  readonly agentModelOverrides?: Partial<Record<AgentType, string>>;
  readonly agentReasoningOverrides?: Partial<Record<AgentType, ReasoningEffort>>;
  readonly agentIterationCaps?: Partial<Record<AgentType, number>>;
  readonly agentPermissionOverrides?: Partial<Record<AgentType, AgentToolPermissionOverride>>;
  readonly allowedChildAgents?: Partial<Record<AgentType, ReadonlyArray<AgentType>>>;
  readonly subagentTimeoutMs?: number;
}

// ─── Config Loading ──────────────────────────────────────────

/**
 * Search paths for config files, in priority order (first found wins per level).
 */
function getConfigPaths(_projectRoot?: string): ReadonlyArray<string> {
  const home = process.env["HOME"] ?? homedir();
  return [join(home, ".config", "devagent", "config.toml")];
}

function readTomlFile(filePath: string): Record<string, unknown> {
  const content = readFileSync(filePath, "utf-8");
  return parseToml(content) as Record<string, unknown>;
}

function resolveEnvValue(
  value: unknown,
  options?: { readonly allowMissing?: boolean },
): unknown {
  if (typeof value === "string" && value.startsWith("env:")) {
    const envKey = value.slice(4);
    const envValue = process.env[envKey];
    if (envValue === undefined) {
      if (options?.allowMissing) {
        return undefined;
      }
      throw new Error(
        `Environment variable "${envKey}" referenced in config but not set`,
      );
    }
    return envValue;
  }
  return value;
}

function parseModelCapabilities(
  raw: Record<string, unknown>,
): ModelCapabilities | undefined {
  const useResponsesApi = (raw["use_responses_api"] ?? raw["useResponsesApi"]) as boolean | undefined;
  const reasoning = (raw["reasoning"]) as boolean | undefined;
  const supportsTemperature = (raw["supports_temperature"] ?? raw["supportsTemperature"]) as boolean | undefined;
  const defaultMaxTokens = (raw["default_max_tokens"] ?? raw["defaultMaxTokens"]) as number | undefined;

  // Only return capabilities object if at least one field is set
  if (
    useResponsesApi === undefined &&
    reasoning === undefined &&
    supportsTemperature === undefined &&
    defaultMaxTokens === undefined
  ) {
    return undefined;
  }

  return { useResponsesApi, reasoning, supportsTemperature, defaultMaxTokens };
}

function mergeProviderConfig(
  raw: Record<string, unknown>,
  options?: { readonly allowMissingApiKeyEnv?: boolean },
): ProviderConfig {
  const capabilities = parseModelCapabilities(raw);
  return {
    apiKey: resolveEnvValue(raw["api_key"] ?? raw["apiKey"], {
      allowMissing: options?.allowMissingApiKeyEnv,
    }) as
      | string
      | undefined,
    baseUrl: raw["base_url"] as string | undefined ?? raw["baseUrl"] as string | undefined,
    model: (raw["model"] as string) ?? "claude-sonnet-4-20250514",
    maxTokens: raw["max_tokens"] as number | undefined ?? raw["maxTokens"] as number | undefined,
    temperature: raw["temperature"] as number | undefined,
    reasoningEffort: raw["reasoning_effort"] as "low" | "medium" | "high" | "xhigh" | undefined,
    ...(capabilities ? { capabilities } : {}),
  };
}

function getRawApiKeyValue(raw: Record<string, unknown> | undefined): string | undefined {
  if (!raw) {
    return undefined;
  }
  return (raw["api_key"] ?? raw["apiKey"]) as string | undefined;
}

function parseSafety(
  raw: Record<string, unknown>,
): Partial<SafetyConfig> {
  return {
    mode: raw["mode"] as SafetyMode | undefined,
    approvalPolicy: (raw["approval_policy"] ?? raw["approvalPolicy"]) as ApprovalPolicyMode | undefined,
    sandboxMode: (raw["sandbox_mode"] ?? raw["sandboxMode"]) as SandboxMode | undefined,
    networkAccess: (raw["network_access"] ?? raw["networkAccess"]) as NetworkAccessMode | undefined,
  };
}
function mergeSafetyConfig(
  parsed: Partial<SafetyConfig>,
  overrides?: Partial<ApprovalPolicy>,
): SafetyConfig {
  const mode = getConfiguredSafetyMode(parsed, overrides);
  const preset = getSafetyPreset(mode);
  return {
    mode,
    approvalPolicy: resolveSafetyField(overrides?.approvalPolicy, parsed.approvalPolicy, preset.approvalPolicy),
    sandboxMode: resolveSafetyField(overrides?.sandboxMode, parsed.sandboxMode, preset.sandboxMode),
    networkAccess: resolveSafetyField(overrides?.networkAccess, parsed.networkAccess, preset.networkAccess),
  };
}

function getConfiguredSafetyMode(
  parsed: Partial<SafetyConfig>,
  overrides?: Partial<ApprovalPolicy>,
): SafetyMode {
  return isSupportedSafetyMode(overrides?.mode)
    ? overrides.mode
    : parsed.mode ?? DEFAULT_SAFETY.mode;
}

function isSupportedSafetyMode(mode: unknown): mode is SafetyMode {
  return mode === SafetyMode.DEFAULT || mode === SafetyMode.AUTOPILOT;
}

function resolveSafetyField<T>(overrideValue: T | undefined, parsedValue: T | undefined, presetValue: T): T {
  return overrideValue ?? parsedValue ?? presetValue;
}

function getSafetyPreset(mode: SafetyMode): Omit<SafetyConfig, "mode"> {
  switch (mode) {
    case SafetyMode.AUTOPILOT:
      return {
        approvalPolicy: "never",
        sandboxMode: "danger-full-access",
        networkAccess: "on",
      };
    case SafetyMode.DEFAULT:
    default:
      return {
        approvalPolicy: "on-request",
        sandboxMode: "workspace-write",
        networkAccess: "off",
      };
  }
}

function parseBudget(
  raw: Record<string, unknown>,
): Partial<BudgetConfig> {
  return {
    maxIterations: raw["max_iterations"] as number | undefined,
    maxContextTokens: raw["max_context_tokens"] as number | undefined,
    responseHeadroom: raw["response_headroom"] as number | undefined,
    costWarningThreshold: raw["cost_warning_threshold"] as number | undefined,
    enableCostTracking: raw["enable_cost_tracking"] as boolean | undefined,
  };
}

function parseContext(
  raw: Record<string, unknown>,
): Partial<ContextConfig> {
  return {
    pruningStrategy: (raw["pruning_strategy"] ??
      raw["pruningStrategy"]) as ContextConfig["pruningStrategy"] | undefined,
    triggerRatio: (raw["trigger_ratio"] ??
      raw["triggerRatio"]) as number | undefined,
    keepRecentMessages: (raw["keep_recent_messages"] ??
      raw["keepRecentMessages"]) as number | undefined,
    turnIsolation: (raw["turn_isolation"] ??
      raw["turnIsolation"]) as boolean | undefined,
    midpointBriefingInterval: (raw["midpoint_briefing_interval"] ??
      raw["midpointBriefingInterval"]) as number | undefined,
    briefingStrategy: (raw["briefing_strategy"] ??
      raw["briefingStrategy"]) as ContextConfig["briefingStrategy"] | undefined,
    pruneProtectTokens: (raw["prune_protect_tokens"] ??
      raw["pruneProtectTokens"]) as number | undefined,
  };
}

function parseAgentValueMap<T>(
  raw: unknown,
  validate: (value: unknown) => T | undefined,
): Partial<Record<AgentType, T>> | undefined {
  if (!raw || typeof raw !== "object") return undefined;
  const out: Partial<Record<AgentType, T>> = {};
  for (const [key, value] of Object.entries(raw as Record<string, unknown>)) {
    if (!VALID_AGENT_TYPES.has(key)) {
      continue;
    }
    const validated = validate(value);
    if (validated !== undefined) {
      out[key as AgentType] = validated;
    }
  }
  return Object.keys(out).length > 0 ? out : undefined;
}

function parseAgentStringMap(
  raw: unknown,
): Partial<Record<AgentType, string>> | undefined {
  return parseAgentValueMap(raw, (value) => {
    if (typeof value !== "string" || value.trim().length === 0) {
      return undefined;
    }
    return value;
  });
}

function parseAgentNumberMap(
  raw: unknown,
): Partial<Record<AgentType, number>> | undefined {
  return parseAgentValueMap(raw, (value) =>
    typeof value === "number" ? value : undefined
  );
}

function parseAgentReasoningMap(
  raw: unknown,
): Partial<Record<AgentType, ReasoningEffort>> | undefined {
  return parseAgentValueMap(raw, (value) => {
    if (typeof value !== "string" || !VALID_REASONING_EFFORTS.has(value as ReasoningEffort)) {
      return undefined;
    }
    return value as ReasoningEffort;
  });
}

function parseAllowedChildAgents(
  raw: unknown,
): Partial<Record<AgentType, ReadonlyArray<AgentType>>> | undefined {
  if (!raw || typeof raw !== "object") return undefined;
  const out: Partial<Record<AgentType, ReadonlyArray<AgentType>>> = {};
  for (const [key, value] of Object.entries(raw as Record<string, unknown>)) {
    if (!VALID_AGENT_TYPES.has(key) || !Array.isArray(value)) continue;
    const valid = value.filter(
      (entry): entry is AgentType =>
        typeof entry === "string" && VALID_AGENT_TYPES.has(entry),
    );
    if (valid.length > 0) {
      out[key as AgentType] = valid;
    }
  }
  return Object.keys(out).length > 0 ? out : undefined;
}
function parseAgentPermissionOverrides(
  raw: unknown,
): Partial<Record<AgentType, AgentToolPermissionOverride>> | undefined {
  if (!raw || typeof raw !== "object") return undefined;
  const out: Partial<Record<AgentType, AgentToolPermissionOverride>> = {};
  for (const [agentType, permissions] of Object.entries(raw as Record<string, unknown>)) {
    const parsed = parseAgentPermissionEntry(permissions);
    if (VALID_AGENT_TYPES.has(agentType) && parsed) {
      out[agentType as AgentType] = parsed as AgentToolPermissionOverride;
    }
  }
  return Object.keys(out).length > 0 ? out : undefined;
}

function parseAgentPermissionEntry(permissions: unknown): Partial<Record<string, "allow" | "deny">> | undefined {
  if (!permissions || typeof permissions !== "object") return undefined;
  const parsed: Partial<Record<string, "allow" | "deny">> = {};
  for (const [category, action] of Object.entries(permissions as Record<string, unknown>)) {
    if (VALID_TOOL_CATEGORIES.has(category) && (action === "allow" || action === "deny")) {
      parsed[category] = action;
    }
  }
  return Object.keys(parsed).length > 0 ? parsed : undefined;
}

function mergeAgentMaps<T>(
  ...maps: ReadonlyArray<Partial<Record<AgentType, T>> | undefined>
): Partial<Record<AgentType, T>> | undefined {
  const merged: Partial<Record<AgentType, T>> = {};
  for (const map of maps) {
    if (!map) continue;
    Object.assign(merged, map);
  }
  return Object.keys(merged).length > 0 ? merged : undefined;
}

/**
 * Load configuration from TOML files + environment.
 * Fail fast: throws on invalid env references, never silently uses defaults for provider keys.
 */
export function loadConfig(
  projectRoot?: string,
  overrides?: Partial<DevAgentConfig>,
): DevAgentConfig {
  const fileConfig = loadFileConfig(projectRoot);
  const envProvider = process.env["DEVAGENT_PROVIDER"];
  const envModel = process.env["DEVAGENT_MODEL"];
  const provider = resolveConfigProvider(fileConfig, overrides, envProvider);
  const rawProviders = getRawProviders(fileConfig);
  const providers = buildProviderConfigs(rawProviders);

  const credentialStore = new CredentialStore();

  if (fileConfig["approval"] !== undefined) {
    throw new ConfigError(
      'The [approval] section has been removed from interactive config. Use [safety] with mode = "default" or "autopilot" instead.',
    );
  }

  const approval = buildApprovalPolicy(fileConfig, overrides);
  const budget = buildBudgetConfig(fileConfig, overrides);
  const context = buildContextConfig(fileConfig, overrides);
  validateBudgetConfig(budget);
  validateContextConfig(context);

  applyActiveCredential({
    fileConfig,
    envModel,
    overrides,
    provider,
    providers,
    rawProviders,
    credentialStore,
  });

  const config = buildLoadedConfig({
    fileConfig,
    envModel,
    provider,
    providers,
    approval,
    budget,
    context,
    overrides,
  });

  validateSubagentConfig(config);
  return config;
}

function loadFileConfig(projectRoot?: string): Record<string, unknown> {
  for (const configPath of getConfigPaths(projectRoot)) {
    if (!existsSync(configPath)) continue;
    try {
      return readTomlFile(configPath);
    } catch (err) {
      const message = extractErrorMessage(err);
      throw new Error(`Failed to parse config file "${configPath}": ${message}`);
    }
  }
  return {};
}

function resolveConfigProvider(
  fileConfig: Record<string, unknown>,
  overrides: Partial<DevAgentConfig> | undefined,
  envProvider: string | undefined,
) {
  return envProvider ?? overrides?.provider ?? (fileConfig["provider"] as string) ?? DEFAULT_CONFIG.provider;
}

function getRawProviders(fileConfig: Record<string, unknown>) {
  return (fileConfig["providers"] ?? {}) as Record<string, Record<string, unknown>>;
}

function buildProviderConfigs(rawProviders: Record<string, Record<string, unknown>>) {
  const providers: Record<string, ProviderConfig> = {};
  for (const [key, value] of Object.entries(rawProviders)) {
    providers[key] = mergeProviderConfig(value, { allowMissingApiKeyEnv: true });
  }
  return providers;
}

function buildApprovalPolicy(
  fileConfig: Record<string, unknown>,
  overrides: Partial<DevAgentConfig> | undefined,
): ApprovalPolicy {
  const rawSafety = (fileConfig["safety"] ?? {}) as Record<string, unknown>;
  const safety = mergeSafetyConfig(parseSafety(rawSafety), overrides?.approval);
  validateSafetyConfig(safety);
  return {
    ...DEFAULT_APPROVAL,
    mode: safety.mode,
    approvalPolicy: safety.approvalPolicy,
    sandboxMode: safety.sandboxMode,
    networkAccess: safety.networkAccess,
    ...overrides?.approval,
  };
}

function buildBudgetConfig(
  fileConfig: Record<string, unknown>,
  overrides: Partial<DevAgentConfig> | undefined,
): BudgetConfig {
  const rawBudget = (fileConfig["budget"] ?? {}) as Record<string, unknown>;
  return {
    ...DEFAULT_BUDGET,
    ...stripUndefined(parseBudget(rawBudget)),
    ...overrides?.budget,
  };
}

function buildContextConfig(
  fileConfig: Record<string, unknown>,
  overrides: Partial<DevAgentConfig> | undefined,
): ContextConfig {
  const rawContext = (fileConfig["context"] ?? {}) as Record<string, unknown>;
  return {
    ...DEFAULT_CONTEXT,
    ...stripUndefined(parseContext(rawContext)),
    ...overrides?.context,
  };
}

function applyActiveCredential(options: {
  readonly fileConfig: Record<string, unknown>;
  readonly envModel: string | undefined;
  readonly overrides: Partial<DevAgentConfig> | undefined;
  readonly provider: string;
  readonly providers: Record<string, ProviderConfig>;
  readonly rawProviders: Record<string, Record<string, unknown>>;
  readonly credentialStore: CredentialStore;
}) {
  const activeCredential = resolveProviderCredentialStatus({
    providerId: options.provider,
    providerConfig: options.providers[options.provider],
    providerConfigApiKey: getRawApiKeyValue(options.rawProviders[options.provider]) ?? null,
    topLevelApiKey: options.fileConfig["api_key"] as string | undefined,
    storedCredential: options.credentialStore.get(options.provider),
  });

  if (activeCredential.credentialMode === "api" && !activeCredential.hasCredential && activeCredential.envVar) {
    throw new Error(`Environment variable "${activeCredential.envVar}" referenced in config but not set`);
  }
  if (activeCredential.credentialMode !== "api" || !activeCredential.apiKey) return;

  applyApiCredential(options, activeCredential.apiKey);
}

function applyApiCredential(
  options: {
    readonly fileConfig: Record<string, unknown>;
    readonly envModel: string | undefined;
    readonly overrides: Partial<DevAgentConfig> | undefined;
    readonly provider: string;
    readonly providers: Record<string, ProviderConfig>;
  },
  apiKey: string,
) {
  const providerConfig = options.providers[options.provider];
  if (providerConfig) {
    if (!providerConfig.apiKey) options.providers[options.provider] = { ...providerConfig, apiKey };
    return;
  }

  options.providers[options.provider] = {
    model: resolveConfigModel(options.fileConfig, options.overrides, options.envModel),
    apiKey,
    baseUrl: options.fileConfig["base_url"] as string | undefined,
  };
}

function buildLoadedConfig(ctx: LoadedConfigContext): DevAgentConfig {
  const optionalSections = parseOptionalConfigSections(ctx.fileConfig);
  return {
    provider: ctx.provider,
    model: resolveConfigModel(ctx.fileConfig, ctx.overrides, ctx.envModel),
    providers: ctx.providers,
    approval: ctx.approval,
    budget: ctx.budget,
    context: ctx.context,
    ...optionalSections,
    ...parseSubagentConfigSections(ctx.fileConfig, ctx.provider, ctx.overrides),
  };
}

function resolveConfigModel(
  fileConfig: Record<string, unknown>,
  overrides: Partial<DevAgentConfig> | undefined,
  envModel: string | undefined,
) {
  return envModel ?? overrides?.model ?? (fileConfig["model"] as string) ?? DEFAULT_CONFIG.model;
}

function parseOptionalConfigSections(fileConfig: Record<string, unknown>) {
  const doubleCheck = parseDoubleCheckConfig(fileConfig["double_check"] as Record<string, unknown> | undefined);
  const lsp = parseLspConfig(fileConfig["lsp"] as Record<string, unknown> | undefined);
  const logging = parseLoggingConfig(fileConfig["logging"] as Record<string, unknown> | undefined);
  const sessionState = parseSessionStateConfig(fileConfig["session_state"] as Record<string, unknown> | undefined);
  return {
    ...(logging ? { logging } : {}),
    ...(doubleCheck ? { doubleCheck } : {}),
    ...(lsp ? { lsp } : {}),
    ...(sessionState ? { sessionState } : {}),
  };
}

function parseSubagentConfigSections(
  fileConfig: Record<string, unknown>,
  provider: string,
  overrides: Partial<DevAgentConfig> | undefined,
): SubagentConfigSections {
  const rawSubagents = (fileConfig["subagents"] ?? {}) as Record<string, unknown>;
  const defaultProfiles = getDefaultSubagentProfiles(provider);
  const agentModelOverrides = mergeAgentMaps(
    defaultProfiles.agentModelOverrides,
    parseAgentStringMap(rawSubagents["agent_model_overrides"] ?? rawSubagents["agentModelOverrides"]),
    overrides?.agentModelOverrides,
  );
  const agentReasoningOverrides = mergeAgentMaps(
    defaultProfiles.agentReasoningOverrides,
    parseAgentReasoningMap(rawSubagents["agent_reasoning_overrides"] ?? rawSubagents["agentReasoningOverrides"]),
    overrides?.agentReasoningOverrides,
  );
  const subagentTimeoutMs = (rawSubagents["subagent_timeout_ms"] ??
    rawSubagents["subagentTimeoutMs"]) as number | undefined;
  return {
    ...(agentModelOverrides ? { agentModelOverrides } : {}),
    ...(agentReasoningOverrides ? { agentReasoningOverrides } : {}),
    ...parseSubagentMaps(rawSubagents),
    ...(subagentTimeoutMs !== undefined ? { subagentTimeoutMs } : {}),
  };
}

function parseSubagentMaps(rawSubagents: Record<string, unknown>): SubagentConfigSections {
  const agentIterationCaps = parseAgentNumberMap(
    rawSubagents["agent_iteration_caps"] ?? rawSubagents["agentIterationCaps"],
  );
  const agentPermissionOverrides = parseAgentPermissionOverrides(
    rawSubagents["agent_permission_overrides"] ?? rawSubagents["agentPermissionOverrides"],
  );
  const allowedChildAgents = parseAllowedChildAgents(
    rawSubagents["allowed_child_agents"] ?? rawSubagents["allowedChildAgents"],
  );
  return {
    ...(agentIterationCaps ? { agentIterationCaps } : {}),
    ...(agentPermissionOverrides ? { agentPermissionOverrides } : {}),
    ...(allowedChildAgents ? { allowedChildAgents } : {}),
  };
}

/**
 * Strip undefined values from an object (for clean merging).
 */
function stripUndefined<T extends Record<string, unknown>>(
  obj: T,
): Partial<T> {
  const result: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(obj)) {
    if (value !== undefined) {
      result[key] = value;
    }
  }
  return result as Partial<T>;
}
