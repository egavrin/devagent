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
import type {
  DevAgentConfig,
  ApprovalPolicy,
  ApprovalMode,
  BudgetConfig,
  ContextConfig,
  MemoryConfig,
  ArkTSConfig,
  ProviderConfig,
  ModelCapabilities,
} from "./types.js";

// ─── Defaults ────────────────────────────────────────────────

const DEFAULT_APPROVAL: ApprovalPolicy = {
  mode: "suggest" as ApprovalMode,
  autoApprovePlan: false,
  autoApproveCode: false,
  autoApproveShell: false,
  auditLog: false,
  toolOverrides: {},
  pathRules: [],
};

const DEFAULT_BUDGET: BudgetConfig = {
  maxIterations: 30,
  maxContextTokens: 100_000,
  responseHeadroom: 2_000,
  costWarningThreshold: 1.0,
  enableCostTracking: true,
};

const DEFAULT_CONTEXT: ContextConfig = {
  pruningStrategy: "hybrid",
  triggerRatio: 0.8,
  keepRecentMessages: 10,
  turnIsolation: true,
  midpointBriefingInterval: 15,
  briefingStrategy: "auto",
};

const DEFAULT_MEMORY: MemoryConfig = {
  enabled: true,
  dailyDecay: 0.02,
  minRelevance: 0.1,
  accessBoost: 0.1,
  recallMinRelevance: 0.3,
  recallLimit: 10,
  promptMaxMemories: 10,
  promptMaxChars: 2000,
  maintenanceOnStartup: true,
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
  memory: DEFAULT_MEMORY,
  arkts: DEFAULT_ARKTS,
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
    reasoningEffort: raw["reasoning_effort"] as "low" | "medium" | "high" | undefined,
    ...(capabilities ? { capabilities } : {}),
  };
}

function parseApproval(
  raw: Record<string, unknown>,
): Partial<ApprovalPolicy> {
  return {
    mode: raw["mode"] as ApprovalMode | undefined,
    autoApprovePlan: raw["auto_approve_plan"] as boolean | undefined,
    autoApproveCode: raw["auto_approve_code"] as boolean | undefined,
    autoApproveShell: raw["auto_approve_shell"] as boolean | undefined,
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
  };
}

function parseMemory(
  raw: Record<string, unknown>,
): Partial<MemoryConfig> {
  return {
    enabled: raw["enabled"] as boolean | undefined,
    dailyDecay: raw["daily_decay"] as number | undefined,
    minRelevance: raw["min_relevance"] as number | undefined,
    accessBoost: raw["access_boost"] as number | undefined,
    recallMinRelevance: raw["recall_min_relevance"] as number | undefined,
    recallLimit: raw["recall_limit"] as number | undefined,
    promptMaxMemories: raw["prompt_max_memories"] as number | undefined,
    promptMaxChars: raw["prompt_max_chars"] as number | undefined,
    maintenanceOnStartup: raw["maintenance_on_startup"] as boolean | undefined,
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
        const message = err instanceof Error ? err.message : String(err);
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

  // Merge memory
  const rawMemory = (fileConfig["memory"] ?? {}) as Record<string, unknown>;
  const memoryPartial = parseMemory(rawMemory);
  const memory: MemoryConfig = {
    ...DEFAULT_MEMORY,
    ...stripUndefined(memoryPartial),
    ...overrides?.memory,
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

  // Resolve top-level api_key: TOML > env var > stored credentials
  const topLevelApiKey = fileConfig["api_key"] as string | undefined;

  // Determine provider early so we can look up stored credentials
  const provider =
    envProvider ??
    overrides?.provider ??
    (fileConfig["provider"] as string) ??
    DEFAULT_CONFIG.provider;

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

  // Parse optional checkpoints config
  const rawCheckpoints = fileConfig["checkpoints"] as Record<string, unknown> | undefined;
  const checkpoints = rawCheckpoints
    ? { enabled: (rawCheckpoints["enabled"] as boolean) ?? false }
    : undefined;

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

  return {
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
    memory,
    arkts,
    ...(checkpoints ? { checkpoints } : {}),
    ...(doubleCheck ? { doubleCheck } : {}),
    ...(lsp ? { lsp } : {}),
  };
}

/**
 * Get default file extensions for a language ID.
 * Used when converting legacy single-server LSP config to multi-server format.
 */
function getLanguageDefaults(
  languageId: string,
): { extensions: string[] } | undefined {
  const map: Record<string, string[]> = {
    typescript: [".ts", ".tsx", ".mts", ".cts"],
    javascript: [".js", ".jsx", ".mjs", ".cjs"],
    python: [".py", ".pyi"],
    c: [".c", ".h"],
    cpp: [".cpp", ".cxx", ".cc", ".hpp", ".hxx", ".hh"],
    rust: [".rs"],
    shellscript: [".sh", ".bash", ".zsh"],
    arkts: [".ets"],
  };
  const exts = map[languageId];
  return exts ? { extensions: exts } : undefined;
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
 * Resolve the project root by searching for .devagent.toml or .git upward.
 */
export function findProjectRoot(startDir?: string): string | null {
  let dir = resolve(startDir ?? process.cwd());
  const root = resolve("/");

  // First pass: look for devagent config (highest priority)
  let searchDir = dir;
  while (searchDir !== root) {
    if (
      existsSync(join(searchDir, ".devagent.toml")) ||
      existsSync(join(searchDir, "devagent.toml"))
    ) {
      return searchDir;
    }
    const parent = resolve(searchDir, "..");
    if (parent === searchDir) break;
    searchDir = parent;
  }

  // Second pass: use cwd if it has package.json (monorepo workspace root)
  if (existsSync(join(dir, "package.json"))) {
    return dir;
  }

  // Third pass: walk up for .git
  searchDir = dir;
  while (searchDir !== root) {
    if (existsSync(join(searchDir, ".git"))) {
      return searchDir;
    }
    const parent = resolve(searchDir, "..");
    if (parent === searchDir) break;
    searchDir = parent;
  }

  return null;
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
        const msg = err instanceof Error ? err.message : String(err);
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
        const msg = err instanceof Error ? err.message : String(err);
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
