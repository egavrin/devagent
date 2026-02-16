/**
 * Configuration loading with TOML support.
 * Hierarchy: defaults → global → project → env → CLI flags.
 * Fail fast: missing required config throws, never guesses.
 */

import { parse as parseToml } from "smol-toml";
import { readFileSync, existsSync } from "node:fs";
import { join, resolve } from "node:path";
import { homedir } from "node:os";
import type {
  DevAgentConfig,
  ApprovalPolicy,
  ApprovalMode,
  BudgetConfig,
  ContextConfig,
  ArkTSConfig,
  ProviderConfig,
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

function mergeProviderConfig(
  raw: Record<string, unknown>,
): ProviderConfig {
  return {
    apiKey: resolveEnvValue(raw["api_key"] ?? raw["apiKey"]) as
      | string
      | undefined,
    baseUrl: raw["base_url"] as string | undefined ?? raw["baseUrl"] as string | undefined,
    model: (raw["model"] as string) ?? "claude-sonnet-4-20250514",
    maxTokens: raw["max_tokens"] as number | undefined ?? raw["maxTokens"] as number | undefined,
    temperature: raw["temperature"] as number | undefined,
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
  const context: ContextConfig = {
    ...DEFAULT_CONTEXT,
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
    ...overrides?.arkts,
  };

  // Resolve top-level api_key if it references env
  const topLevelApiKey = fileConfig["api_key"] as string | undefined;
  const resolvedApiKey = topLevelApiKey
    ? (resolveEnvValue(topLevelApiKey) as string)
    : envApiKey;

  // If we have a top-level API key, inject it into the provider config
  const provider =
    envProvider ??
    overrides?.provider ??
    (fileConfig["provider"] as string) ??
    DEFAULT_CONFIG.provider;

  if (resolvedApiKey && !providers[provider]) {
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
    arkts,
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
