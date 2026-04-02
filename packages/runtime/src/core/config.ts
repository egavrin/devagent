/**
 * Configuration loading with TOML support.
 * Hierarchy: defaults → global → project → env → CLI flags.
 * Fail fast: missing required config throws, never guesses.
 */

import { parse as parseToml } from "smol-toml";
import { readFileSync, existsSync } from "node:fs";
import { join, resolve } from "node:path";
import { homedir } from "node:os";
import { CredentialStore } from "./credentials.js";
import { ConfigError , extractErrorMessage } from "./errors.js";
import { LANGUAGE_EXTENSIONS } from "./languages.js";
import type {
  DevAgentConfig,
  ApprovalPolicy,
  ApprovalMode,
  BudgetConfig,
  ContextConfig,
  ArkTSConfig,
  ProviderConfig,
  ModelCapabilities,
  AgentToolPermissionOverride,
  ReasoningEffort,
} from "./types.js";
import { AgentType } from "./types.js";

// ─── Defaults ────────────────────────────────────────────────

const DEFAULT_APPROVAL: ApprovalPolicy = {
  mode: "suggest" as ApprovalMode,
  auditLog: false,
  toolOverrides: {},
  pathRules: [],
};

export const DEFAULT_BUDGET: BudgetConfig = {
  maxIterations: 30,
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

const DEFAULT_ARKTS: ArkTSConfig = {
  enabled: false,
  strictMode: false,
  targetVersion: "5.0",
};

const DEFAULT_CONFIG: DevAgentConfig = {
  provider: "anthropic",
  model: "claude-sonnet-4-20250514",
  providers: {},
  approval: DEFAULT_APPROVAL,
  budget: DEFAULT_BUDGET,
  context: DEFAULT_CONTEXT,
  arkts: DEFAULT_ARKTS,
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

const OPENAI_FAMILY_SUBAGENT_MODEL_DEFAULTS: Partial<Record<AgentType, string>> = {
  [AgentType.EXPLORE]: "gpt-5.4-mini",
  [AgentType.REVIEWER]: "gpt-5.4",
  [AgentType.ARCHITECT]: "gpt-5.4",
};

const OPENAI_FAMILY_SUBAGENT_REASONING_DEFAULTS: Partial<Record<AgentType, ReasoningEffort>> = {
  [AgentType.EXPLORE]: "low",
  [AgentType.REVIEWER]: "high",
  [AgentType.ARCHITECT]: "high",
};

const DEVAGENT_API_SUBAGENT_REASONING_DEFAULTS: Partial<Record<AgentType, ReasoningEffort>> = {
  [AgentType.EXPLORE]: "low",
  [AgentType.REVIEWER]: "high",
  [AgentType.ARCHITECT]: "high",
};

// ─── Config Loading ──────────────────────────────────────────

/**
 * Search paths for config files, in priority order (first found wins per level).
 */
function getConfigPaths(projectRoot?: string): ReadonlyArray<string> {
  const paths: string[] = [];

  // Project-level (highest file priority)
  if (projectRoot) {
    paths.push(join(projectRoot, ".devagent.toml"));
    paths.push(join(projectRoot, "devagent.toml"));
  }

  // Global config
  const home = homedir();
  paths.push(join(home, ".config", "devagent", "config.toml"));
  paths.push(join(home, ".devagent.toml"));

  return paths;
}

function readTomlFile(filePath: string): Record<string, unknown> {
  const content = readFileSync(filePath, "utf-8");
  return parseToml(content) as Record<string, unknown>;
}

function resolveEnvValue(value: unknown): unknown {
  if (typeof value === "string" && value.startsWith("env:")) {
    const envKey = value.slice(4);
    const envValue = process.env[envKey];
    if (envValue === undefined) {
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
): ProviderConfig {
  const capabilities = parseModelCapabilities(raw);
  return {
    apiKey: resolveEnvValue(raw["api_key"] ?? raw["apiKey"]) as
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

function parseApproval(
  raw: Record<string, unknown>,
): Partial<ApprovalPolicy> {
  return {
    mode: raw["mode"] as ApprovalMode | undefined,
    auditLog: raw["audit_log"] as boolean | undefined,
  };
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
    if (!VALID_AGENT_TYPES.has(agentType) || !permissions || typeof permissions !== "object") {
      continue;
    }
    const parsed: Partial<Record<string, "allow" | "deny">> = {};
    for (const [category, action] of Object.entries(permissions as Record<string, unknown>)) {
      if (!VALID_TOOL_CATEGORIES.has(category)) continue;
      if (action === "allow" || action === "deny") {
        parsed[category] = action;
      }
    }
    if (Object.keys(parsed).length > 0) {
      out[agentType as AgentType] = parsed as AgentToolPermissionOverride;
    }
  }
  return Object.keys(out).length > 0 ? out : undefined;
}

function validateBudgetConfig(budget: BudgetConfig): void {
  if (!Number.isInteger(budget.maxIterations) || budget.maxIterations < 0) {
    throw new ConfigError(
      `Invalid budget.maxIterations: expected integer >= 0, got ${String(budget.maxIterations)}`,
    );
  }
  if (!Number.isFinite(budget.maxContextTokens) || budget.maxContextTokens < 0) {
    throw new ConfigError(
      `Invalid budget.maxContextTokens: expected number >= 0, got ${String(budget.maxContextTokens)}`,
    );
  }
  if (!Number.isFinite(budget.responseHeadroom) || budget.responseHeadroom < 0) {
    throw new ConfigError(
      `Invalid budget.responseHeadroom: expected number >= 0, got ${String(budget.responseHeadroom)}`,
    );
  }
  if (budget.maxContextTokens > 0 && budget.responseHeadroom >= budget.maxContextTokens) {
    throw new ConfigError(
      `Invalid budget.responseHeadroom: must be < budget.maxContextTokens (${budget.responseHeadroom} >= ${budget.maxContextTokens})`,
    );
  }
  if (!Number.isFinite(budget.costWarningThreshold) || budget.costWarningThreshold < 0) {
    throw new ConfigError(
      `Invalid budget.costWarningThreshold: expected number >= 0, got ${String(budget.costWarningThreshold)}`,
    );
  }
}

function validateContextConfig(context: ContextConfig): void {
  const pruningStrategies = new Set(["sliding_window", "summarize", "hybrid"]);
  if (!pruningStrategies.has(context.pruningStrategy)) {
    throw new ConfigError(
      `Invalid context.pruningStrategy: ${String(context.pruningStrategy)}`,
    );
  }
  if (!Number.isFinite(context.triggerRatio) || context.triggerRatio <= 0 || context.triggerRatio > 1) {
    throw new ConfigError(
      `Invalid context.triggerRatio: expected number in (0, 1], got ${String(context.triggerRatio)}`,
    );
  }
  if (!Number.isInteger(context.keepRecentMessages) || context.keepRecentMessages < 1) {
    throw new ConfigError(
      `Invalid context.keepRecentMessages: expected integer >= 1, got ${String(context.keepRecentMessages)}`,
    );
  }
  if (
    context.midpointBriefingInterval !== undefined &&
    (!Number.isInteger(context.midpointBriefingInterval) || context.midpointBriefingInterval < 0)
  ) {
    throw new ConfigError(
      `Invalid context.midpointBriefingInterval: expected integer >= 0, got ${String(context.midpointBriefingInterval)}`,
    );
  }
  if (context.briefingStrategy !== undefined) {
    const briefingStrategies = new Set(["heuristic", "llm", "auto"]);
    if (!briefingStrategies.has(context.briefingStrategy)) {
      throw new ConfigError(
        `Invalid context.briefingStrategy: ${String(context.briefingStrategy)}`,
      );
    }
  }
}

function validateSubagentConfig(config: DevAgentConfig): void {
  for (const [agentType, cap] of Object.entries(config.agentIterationCaps ?? {})) {
    if (!VALID_AGENT_TYPES.has(agentType) || !Number.isInteger(cap) || cap < 1) {
      throw new ConfigError(
        `Invalid agentIterationCaps.${agentType}: expected integer >= 1, got ${String(cap)}`,
      );
    }
  }

  if (
    config.subagentTimeoutMs !== undefined &&
    (!Number.isInteger(config.subagentTimeoutMs) || config.subagentTimeoutMs < 0)
  ) {
    throw new ConfigError(
      `Invalid subagentTimeoutMs: expected integer >= 0, got ${String(config.subagentTimeoutMs)}`,
    );
  }
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

function getDefaultSubagentProfiles(
  provider: string,
): {
  readonly agentModelOverrides?: Partial<Record<AgentType, string>>;
  readonly agentReasoningOverrides?: Partial<Record<AgentType, ReasoningEffort>>;
} {
  if (provider !== "openai" && provider !== "chatgpt") {
    if (provider === "devagent-api") {
      return {
        agentReasoningOverrides: DEVAGENT_API_SUBAGENT_REASONING_DEFAULTS,
      };
    }
    return {};
  }

  return {
    agentModelOverrides: OPENAI_FAMILY_SUBAGENT_MODEL_DEFAULTS,
    agentReasoningOverrides: OPENAI_FAMILY_SUBAGENT_REASONING_DEFAULTS,
  };
}

/**
 * Load configuration from TOML files + environment.
 * Fail fast: throws on invalid env references, never silently uses defaults for provider keys.
 */
export function loadConfig(
  projectRoot?: string,
  overrides?: Partial<DevAgentConfig>,
): DevAgentConfig {
  let fileConfig: Record<string, unknown> = {};

  // Find and load the first available config file at each level
  const paths = getConfigPaths(projectRoot);
  for (const configPath of paths) {
    if (existsSync(configPath)) {
      try {
        fileConfig = readTomlFile(configPath);
        break; // Use first found config file
      } catch (err) {
        const message = extractErrorMessage(err);
        throw new Error(
          `Failed to parse config file "${configPath}": ${message}`,
        );
      }
    }
  }

  // Build providers map
  const rawProviders = (fileConfig["providers"] ?? {}) as Record<
    string,
    Record<string, unknown>
  >;
  const providers: Record<string, ProviderConfig> = {};
  for (const [key, value] of Object.entries(rawProviders)) {
    providers[key] = mergeProviderConfig(value);
  }

  const credentialStore = new CredentialStore();

  // Check env overrides
  const envProvider = process.env["DEVAGENT_PROVIDER"];
  const envModel = process.env["DEVAGENT_MODEL"];
  const envApiKey = process.env["DEVAGENT_API_KEY"];

  // Merge approval
  const rawApproval = (fileConfig["approval"] ?? {}) as Record<
    string,
    unknown
  >;
  const approvalPartial = parseApproval(rawApproval);
  const approval: ApprovalPolicy = {
    ...DEFAULT_APPROVAL,
    ...stripUndefined(approvalPartial),
    ...overrides?.approval,
  };

  // Merge budget
  const rawBudget = (fileConfig["budget"] ?? {}) as Record<
    string,
    unknown
  >;
  const budgetPartial = parseBudget(rawBudget);
  const budget: BudgetConfig = {
    ...DEFAULT_BUDGET,
    ...stripUndefined(budgetPartial),
    ...overrides?.budget,
  };

  // Merge context
  const rawContext = (fileConfig["context"] ?? {}) as Record<string, unknown>;
  const contextPartial = parseContext(rawContext);
  const context: ContextConfig = {
    ...DEFAULT_CONTEXT,
    ...stripUndefined(contextPartial),
    ...overrides?.context,
  };

  // Merge arkts
  const rawArkts = (fileConfig["arkts"] ?? {}) as Record<string, unknown>;
  const arkts: ArkTSConfig = {
    ...DEFAULT_ARKTS,
    enabled: (rawArkts["enabled"] as boolean) ?? DEFAULT_ARKTS.enabled,
    strictMode:
      (rawArkts["strict_mode"] as boolean) ?? DEFAULT_ARKTS.strictMode,
    targetVersion:
      (rawArkts["target_version"] as string) ??
      DEFAULT_ARKTS.targetVersion,
    linterPath: rawArkts["linter_path"] as string | undefined,
    ...overrides?.arkts,
  };

  validateBudgetConfig(budget);
  validateContextConfig(context);

  // Resolve top-level api_key: TOML > env var > stored credentials
  const topLevelApiKey = fileConfig["api_key"] as string | undefined;

  // Determine provider early so we can look up stored credentials
  const provider =
    envProvider ??
    overrides?.provider ??
    (fileConfig["provider"] as string) ??
    DEFAULT_CONFIG.provider;

  const rawSubagents = (fileConfig["subagents"] ?? {}) as Record<string, unknown>;
  const explicitAgentModelOverrides = parseAgentStringMap(
    rawSubagents["agent_model_overrides"] ?? rawSubagents["agentModelOverrides"],
  );
  const explicitAgentReasoningOverrides = parseAgentReasoningMap(
    rawSubagents["agent_reasoning_overrides"] ?? rawSubagents["agentReasoningOverrides"],
  );
  const agentIterationCaps = parseAgentNumberMap(
    rawSubagents["agent_iteration_caps"] ?? rawSubagents["agentIterationCaps"],
  );
  const agentPermissionOverrides = parseAgentPermissionOverrides(
    rawSubagents["agent_permission_overrides"] ?? rawSubagents["agentPermissionOverrides"],
  );
  const allowedChildAgents = parseAllowedChildAgents(
    rawSubagents["allowed_child_agents"] ?? rawSubagents["allowedChildAgents"],
  );
  const subagentTimeoutMs = (rawSubagents["subagent_timeout_ms"] ??
    rawSubagents["subagentTimeoutMs"]) as number | undefined;
  const defaultSubagentProfiles = getDefaultSubagentProfiles(provider);
  const agentModelOverrides = mergeAgentMaps(
    defaultSubagentProfiles.agentModelOverrides,
    explicitAgentModelOverrides,
    overrides?.agentModelOverrides,
  );
  const agentReasoningOverrides = mergeAgentMaps(
    defaultSubagentProfiles.agentReasoningOverrides,
    explicitAgentReasoningOverrides,
    overrides?.agentReasoningOverrides,
  );

  const storedCred = credentialStore.get(provider);
  const storedApiKey = storedCred?.type === "api" ? storedCred.key : undefined;
  const resolvedApiKey = topLevelApiKey
    ? (resolveEnvValue(topLevelApiKey) as string)
    : envApiKey ?? storedApiKey;

  // Apply resolved API key to active provider entry if it doesn't already define one.
  // Priority: provider-level config > top-level/env > stored credentials.
  if (resolvedApiKey) {
    if (providers[provider]) {
      if (!providers[provider].apiKey) {
        providers[provider] = { ...providers[provider], apiKey: resolvedApiKey };
      }
    } else {
      providers[provider] = {
        model:
          envModel ??
          overrides?.model ??
          (fileConfig["model"] as string) ??
          DEFAULT_CONFIG.model,
        apiKey: resolvedApiKey,
        baseUrl: fileConfig["base_url"] as string | undefined,
      };
    }
  }

  // Inject stored credentials into any provider entries still missing apiKey.
  for (const [key, provConfig] of Object.entries(providers)) {
    if (!provConfig.apiKey) {
      const stored = credentialStore.get(key);
      if (stored?.type === "api") {
        providers[key] = { ...provConfig, apiKey: stored.key };
      }
    }
  }

  // Parse optional double_check config
  const rawDoubleCheck = fileConfig["double_check"] as Record<string, unknown> | undefined;
  const doubleCheck = rawDoubleCheck
    ? {
        enabled: (rawDoubleCheck["enabled"] as boolean) ?? false,
        checkDiagnostics: rawDoubleCheck["check_diagnostics"] as boolean | undefined,
        runTests: rawDoubleCheck["run_tests"] as boolean | undefined,
        testCommand: rawDoubleCheck["test_command"] as string | null | undefined,
        diagnosticTimeout: rawDoubleCheck["diagnostic_timeout"] as number | undefined,
      }
    : undefined;

  // Parse optional LSP config — supports both legacy single-server and new multi-server format
  const rawLsp = fileConfig["lsp"] as Record<string, unknown> | undefined;
  let lsp: import("./types.js").LSPConfig | undefined;

  if (rawLsp) {
    if (rawLsp["servers"]) {
      // New multi-server format: [[lsp.servers]]
      const rawServers = rawLsp["servers"] as Array<Record<string, unknown>>;
      lsp = {
        servers: rawServers.map((s) => ({
          command: s["command"] as string,
          args: (s["args"] as string[] | undefined) ?? ["--stdio"],
          languages: (s["languages"] as string[] | undefined) ?? ["typescript"],
          extensions: (s["extensions"] as string[] | undefined) ?? [".ts"],
          timeout: (s["timeout"] as number | undefined) ?? 10_000,
          diagnosticTimeout: s["diagnostic_timeout"] as number | undefined,
        })),
      };
    } else if (rawLsp["command"]) {
      // Legacy single-server format: [lsp] command = "..."
      const languageId = (rawLsp["language_id"] as string | undefined) ?? "typescript";
      const defaults = getLanguageDefaults(languageId);
      lsp = {
        servers: [{
          command: rawLsp["command"] as string,
          args: (rawLsp["args"] as string[] | undefined) ?? ["--stdio"],
          languages: [languageId],
          extensions: defaults?.extensions ?? [".ts"],
          timeout: (rawLsp["timeout"] as number | undefined) ?? 10_000,
          diagnosticTimeout: rawLsp["diagnostic_timeout"] as number | undefined,
        }],
      };
    }
  }

  // Parse optional logging config
  const rawLogging = fileConfig["logging"] as Record<string, unknown> | undefined;
  const logging = rawLogging
    ? {
        enabled: (rawLogging["enabled"] as boolean) ?? true,
        logDir: rawLogging["log_dir"] as string | undefined,
        retentionDays: rawLogging["retention_days"] as number | undefined,
      }
    : undefined;

  // Parse optional session_state config
  const rawSessionState = fileConfig["session_state"] as Record<string, unknown> | undefined;
  const sessionState = rawSessionState
    ? {
        persist: rawSessionState["persist"] as boolean | undefined,
        trackPlan: rawSessionState["track_plan"] as boolean | undefined,
        trackFiles: rawSessionState["track_files"] as boolean | undefined,
        trackEnv: rawSessionState["track_env"] as boolean | undefined,
        trackToolResults: rawSessionState["track_tool_results"] as boolean | undefined,
        trackFindings: rawSessionState["track_findings"] as boolean | undefined,
        maxModifiedFiles: rawSessionState["max_modified_files"] as number | undefined,
        maxEnvFacts: rawSessionState["max_env_facts"] as number | undefined,
        maxToolSummaries: rawSessionState["max_tool_summaries"] as number | undefined,
        maxFindings: rawSessionState["max_findings"] as number | undefined,
      }
    : undefined;

  const config: DevAgentConfig = {
    provider,
    model:
      envModel ??
      overrides?.model ??
      (fileConfig["model"] as string) ??
      DEFAULT_CONFIG.model,
    providers,
    approval,
    budget,
    context,
    arkts,
    ...(logging ? { logging } : {}),
    ...(doubleCheck ? { doubleCheck } : {}),
    ...(lsp ? { lsp } : {}),
    ...(sessionState ? { sessionState } : {}),
    ...(agentModelOverrides ? { agentModelOverrides } : {}),
    ...(agentReasoningOverrides ? { agentReasoningOverrides } : {}),
    ...(agentIterationCaps ? { agentIterationCaps } : {}),
    ...(agentPermissionOverrides ? { agentPermissionOverrides } : {}),
    ...(allowedChildAgents ? { allowedChildAgents } : {}),
    ...(subagentTimeoutMs !== undefined ? { subagentTimeoutMs } : {}),
  };

  validateSubagentConfig(config);
  return config;
}

/**
 * Get default file extensions for a language ID.
 * Used when converting legacy single-server LSP config to multi-server format.
 */
function getLanguageDefaults(
  languageId: string,
): { extensions: string[] } | undefined {
  const exts = LANGUAGE_EXTENSIONS[languageId];
  return exts ? { extensions: [...exts] } : undefined;
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

/**
 * Resolve the project root by walking upward once, checking ALL markers at
 * each directory level. Priority order (highest first):
 *   1. .devagent.toml / devagent.toml  — immediate return
 *   2. .agents/skills                 — immediate return
 *   3. package.json (start dir only)  — remembered as fallback
 *   4. .git                           — first match remembered as fallback
 */
export function findProjectRoot(startDir?: string): string | null {
  const start = resolve(startDir ?? process.cwd());
  const root = resolve("/");

  let gitFallback: string | null = null;
  let packageJsonFallback: string | null = null;
  let dir = start;

  while (dir !== root) {
    // Highest priority: devagent config — return immediately
    if (
      existsSync(join(dir, ".devagent.toml")) ||
      existsSync(join(dir, "devagent.toml"))
    ) {
      return dir;
    }

    // Standalone agent workspaces should not float up to a parent git root.
    if (existsSync(join(dir, ".agents", "skills"))) {
      return dir;
    }

    // package.json fallback — only at the starting directory
    if (dir === start && packageJsonFallback === null && existsSync(join(dir, "package.json"))) {
      packageJsonFallback = dir;
    }

    // .git fallback — keep the first (closest) match
    if (gitFallback === null && existsSync(join(dir, ".git"))) {
      gitFallback = dir;
    }

    const parent = resolve(dir, "..");
    if (parent === dir) break;
    dir = parent;
  }

  // Fall back: package.json at start dir takes priority over .git
  return packageJsonFallback ?? gitFallback ?? null;
}

// ─── OAuth Credential Resolution ────────────────────────────

import type { OAuthCredential } from "./credentials.js";
import { getOAuthProvider } from "./oauth-providers.js";
import { refreshAccessToken, exchangeCopilotSessionToken } from "./oauth.js";
import { OAuthError } from "./errors.js";

/**
 * Resolve OAuth credentials for all providers that have stored OAuth tokens.
 * Refreshes expired tokens automatically. Call after loadConfig() in main().
 * Keeps loadConfig() synchronous — async refresh is deferred to this function.
 */
export async function resolveProviderCredentials(
  config: DevAgentConfig,
): Promise<DevAgentConfig> {
  const credentialStore = new CredentialStore();
  const updatedProviders: Record<string, ProviderConfig> = { ...config.providers };

  // Check the active provider for OAuth credentials
  const providersToCheck = new Set(Object.keys(updatedProviders));
  providersToCheck.add(config.provider);

  for (const key of providersToCheck) {
    const provConfig = updatedProviders[key];
    // Skip if already has apiKey or oauthToken
    if (provConfig?.apiKey || provConfig?.oauthToken) continue;

    const stored = credentialStore.get(key);
    if (stored?.type !== "oauth") continue;

    let accessToken = stored.accessToken;

    // Refresh if expired (1-minute buffer).
    // Skip refresh for tokens without expiresAt (e.g., GitHub OAuth — non-expiring).
    const isExpired = stored.expiresAt != null && stored.expiresAt < Date.now() + 60_000;
    if (isExpired && stored.refreshToken) {
      const oauthConfig = getOAuthProvider(key);
      if (!oauthConfig) {
        throw new OAuthError(
          `OAuth token for "${key}" is expired but no OAuth config found to refresh. Run "devagent auth login" to re-authenticate.`,
        );
      }

      try {
        const newTokens = await refreshAccessToken(
          oauthConfig.tokenUrl,
          stored.refreshToken,
          oauthConfig.clientId,
        );
        const updated: OAuthCredential = {
          type: "oauth",
          accessToken: newTokens.access_token,
          ...(newTokens.refresh_token ? { refreshToken: newTokens.refresh_token } : {}),
          ...(newTokens.expires_in ? { expiresAt: Date.now() + newTokens.expires_in * 1000 } : {}),
          accountId: stored.accountId,
          storedAt: Date.now(),
        };
        credentialStore.set(key, updated);
        accessToken = updated.accessToken;
      } catch (err) {
        const msg = extractErrorMessage(err);
        throw new OAuthError(
          `Failed to refresh OAuth token for "${key}": ${msg}. Run "devagent auth login" to re-authenticate.`,
        );
      }
    } else if (isExpired && !stored.refreshToken) {
      throw new OAuthError(
        `OAuth token for "${key}" is expired and cannot be refreshed (no refresh token). Run "devagent auth login" to re-authenticate.`,
      );
    }

    // Inject OAuth token into provider config
    const existingConfig = updatedProviders[key] ?? { model: config.model };

    if (key === "github-copilot") {
      // GitHub Copilot requires exchanging the GitHub OAuth token for a
      // short-lived Copilot session JWT before API calls.
      try {
        const session = await exchangeCopilotSessionToken(accessToken);
        updatedProviders[key] = {
          ...existingConfig,
          oauthToken: session.token,
          baseUrl: session.endpoint ?? existingConfig.baseUrl,
        };
      } catch (err) {
        const msg = extractErrorMessage(err);
        throw new OAuthError(
          `Failed to obtain Copilot session token: ${msg}. Run "devagent auth login" to re-authenticate.`,
        );
      }
    } else {
      updatedProviders[key] = {
        ...existingConfig,
        oauthToken: accessToken,
        oauthAccountId: stored.accountId,
      };
    }
  }

  return { ...config, providers: updatedProviders };
}
