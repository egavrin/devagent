import { createDefaultRegistry, validateOllamaModel } from "@devagent/providers";
import {
  DEFAULT_BUDGET,
  DEFAULT_CONTEXT,
  extractErrorMessage,
  findProjectRoot,
  getProviderCredentialEnvVar,
  loadConfig,
  loadModelRegistry,
  lookupModelEntry,
  resolveProviderCredentials,
} from "@devagent/runtime";
import { dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { buildConfigOverridesFromCliArgs } from "./cli-args.js";
import { dim, formatError } from "./format.js";
import {
  migrateLegacyGlobalConfigIfNeeded,
  migrateLegacyGlobalTomlIfNeeded,
  normalizeGlobalConfigIfNeeded,
} from "./global-config.js";
import type { CliArgs, ConfigSetupResult, ProviderSetupResult } from "./main-types.js";
import { resolveBundledModelsDir } from "./model-registry-path.js";
import { buildProviderConfig } from "./provider-config.js";
import {
  formatProviderModelCompatibilityError,
  formatProviderModelCompatibilityHint,
  getProviderModelCompatibilityIssue,
} from "./provider-model-compat.js";
import type { DevAgentConfig } from "@devagent/runtime";

function migrateGlobalConfigFiles(): void {
  const migration = migrateLegacyGlobalConfigIfNeeded();
  if (migration.migrated) process.stderr.write(dim("[config] Migrated legacy config.json to config.toml") + "\n");
  const tomlMigration = migrateLegacyGlobalTomlIfNeeded();
  if (tomlMigration.migrated) process.stderr.write(dim("[config] Migrated legacy ~/.devagent.toml to config.toml") + "\n");
  normalizeGlobalConfigIfNeeded();
}

async function resolveConfigCredentials(config: DevAgentConfig): Promise<DevAgentConfig> {
  try {
    return await resolveProviderCredentials(config);
  } catch (err) {
    const msg = extractErrorMessage(err);
    process.stderr.write(formatError(`OAuth credential error: ${msg}`) + "\n");
    process.stderr.write(dim('Run "devagent auth login" to re-authenticate.') + "\n");
    process.exit(1);
  }
}

function applyIterationOverride(config: DevAgentConfig, cliArgs: CliArgs): DevAgentConfig {
  return cliArgs.maxIterations === null
    ? config
    : { ...config, budget: { ...config.budget, maxIterations: cliArgs.maxIterations } };
}

function loadBundledModels(projectRoot: string): void {
  const cliDir = dirname(fileURLToPath(import.meta.url));
  loadModelRegistry(projectRoot, [resolveBundledModelsDir(cliDir)]);
}

function validateProviderModelPairing(config: DevAgentConfig): void {
  const providerModelIssue = getProviderModelCompatibilityIssue(config.provider, config.model);
  if (!providerModelIssue) return;
  process.stderr.write(formatError(formatProviderModelCompatibilityError(providerModelIssue)) + "\n");
  const hint = formatProviderModelCompatibilityHint(providerModelIssue);
  if (hint) process.stderr.write(dim(hint) + "\n");
  process.exit(1);
}

function applyRegistryContextBudget(config: DevAgentConfig): DevAgentConfig {
  const registryEntry = lookupModelEntry(config.model, config.provider);
  if (!registryEntry || config.budget.maxContextTokens !== DEFAULT_BUDGET.maxContextTokens) return config;
  return {
    ...config,
    budget: {
      ...config.budget,
      maxContextTokens: registryEntry.contextWindow,
      responseHeadroom: registryEntry.responseHeadroom,
    },
  };
}

function applyScaledKeepRecentMessages(config: DevAgentConfig): DevAgentConfig {
  if (config.context.keepRecentMessages !== DEFAULT_CONTEXT.keepRecentMessages) return config;
  const scaledKeep = Math.floor((config.budget.maxContextTokens - config.budget.responseHeadroom) / 1500);
  return scaledKeep <= DEFAULT_CONTEXT.keepRecentMessages
    ? config
    : { ...config, context: { ...config.context, keepRecentMessages: scaledKeep } };
}

export async function setupConfig(cliArgs: CliArgs): Promise<ConfigSetupResult> {
  migrateGlobalConfigFiles();
  const projectRoot = findProjectRoot() ?? process.cwd();
  let config = await resolveConfigCredentials(loadConfig(projectRoot, buildConfigOverridesFromCliArgs(cliArgs)));
  config = applyIterationOverride(config, cliArgs);
  loadBundledModels(projectRoot);
  validateProviderModelPairing(config);
  return { config: applyScaledKeepRecentMessages(applyRegistryContextBudget(config)), projectRoot };
}

export function setupProvider(config: DevAgentConfig, cliArgs: CliArgs): ProviderSetupResult {
  const providerRegistry = createDefaultRegistry();
  const providerConfig = buildProviderConfig(config, cliArgs.reasoning ?? undefined);
  const noKeyProviders = new Set(["ollama", "chatgpt", "github-copilot"]);
  if (!providerConfig.apiKey && !providerConfig.oauthToken && !noKeyProviders.has(config.provider)) {
    const envVar = getProviderCredentialEnvVar(config.provider);
    process.stderr.write(formatError(`No API key configured for provider "${config.provider}".`) + "\n");
    process.stderr.write(dim(envVar ? `Run "devagent auth login" to store a key, or set ${envVar}` : 'Run "devagent auth login" to store a key.') + "\n");
    process.exit(1);
  }
  return { provider: providerRegistry.get(config.provider, providerConfig), providerRegistry };
}

export async function validateOllamaAvailability(config: DevAgentConfig): Promise<void> {
  if (config.provider !== "ollama") return;
  try {
    const ollamaConfig = config.providers[config.provider];
    await validateOllamaModel(config.model, ollamaConfig?.baseUrl ?? "http://localhost:11434/v1");
  } catch (err) {
    process.stderr.write(formatError(extractErrorMessage(err)) + "\n");
    process.exit(1);
  }
}
