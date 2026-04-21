/**
 * Model registry — loads model definitions from TOML files
 * and provides capability lookup by model name.
 *
 * Registry files are searched in:
 *   1. <projectRoot>/models/*.toml
 *   2. <repoRoot>/models/*.toml  (devagent repo itself)
 *   3. ~/.config/devagent/models/*.toml
 *
 * Each TOML file has a top-level `provider` and `base_url`,
 * then `[model-name]` sections with per-model capabilities.
 */

import { readFileSync, readdirSync, existsSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { parse as parseToml } from "smol-toml";

import { EMBEDDED_MODEL_TOML } from "./embedded-models.js";
import { extractErrorMessage } from "./errors.js";
import type { ModelCapabilities } from "./types.js";

// ─── Types ──────────────────────────────────────────────────

export interface ModelPricing {
  readonly inputPricePerMillion: number;
  readonly outputPricePerMillion: number;
}

export interface ModelRegistryEntry {
  readonly provider: string;
  readonly baseUrl: string | undefined;
  readonly contextWindow: number;
  readonly responseHeadroom: number;
  readonly capabilities: ModelCapabilities;
  readonly pricing: ModelPricing | undefined;
}

// ─── Registry ───────────────────────────────────────────────

const registry = new Map<string, ModelRegistryEntry[]>();
const registryByProvider = new Map<string, Map<string, ModelRegistryEntry>>();
let loaded = false;

/** Reset registry state. Test-only — not exported from package barrel. */
export function _resetRegistryForTesting(): void {
  registry.clear();
  registryByProvider.clear();
  loaded = false;
}

/**
 * Load all model TOML files from known directories.
 * Idempotent — only loads once per process.
 *
 * Search order:
 *   1. Explicit additional directories (e.g., devagent repo models/)
 *   2. Project-level models/ directory
 *   3. ~/.config/devagent/models/
 *
 * First-found wins: if a model is defined in multiple files, the first
 * loaded definition is used.
 */
export function loadModelRegistry(
  projectRoot?: string,
  additionalDirs?: ReadonlyArray<string>,
): void {
  if (loaded) return;
  loaded = true;

  const searchDirs: string[] = [];

  // Explicit additional directories (highest priority)
  if (additionalDirs) {
    for (const dir of additionalDirs) {
      searchDirs.push(dir);
    }
  }

  // Project-level models directory
  if (projectRoot) {
    searchDirs.push(join(projectRoot, "models"));
  }

  // Global models directory
  searchDirs.push(join(process.env["HOME"] ?? homedir(), ".config", "devagent", "models"));

  for (const dir of searchDirs) {
    loadModelsFromDir(dir);
  }

  // Fallback: load embedded model definitions (for bundled CLI)
  loadModelsFromEmbedded();
}

/**
 * Look up capabilities for a model by name.
 * Returns undefined if the model is not in the registry.
 */
export function lookupModelCapabilities(
  modelName: string,
  provider?: string,
): ModelCapabilities | undefined {
  const entry = lookupModelEntry(modelName, provider);
  return entry?.capabilities;
}

/**
 * Look up a full model registry entry by name.
 */
export function lookupModelEntry(
  modelName: string,
  provider?: string,
): ModelRegistryEntry | undefined {
  if (provider) {
    return registryByProvider.get(provider)?.get(modelName);
  }
  return registry.get(modelName)?.[0];
}

/**
 * Get all registered model names.
 */
export function getRegisteredModels(provider?: string): ReadonlyArray<string> {
  if (provider) {
    return Array.from(registryByProvider.get(provider)?.keys() ?? []);
  }
  return Array.from(registry.keys());
}

/**
 * Return every provider that registers the given model name.
 */
export function getProvidersForModel(modelName: string): ReadonlyArray<string> {
  return (registry.get(modelName) ?? []).map((entry) => entry.provider);
}

/**
 * Return whether a model is registered for a provider.
 */
export function isModelRegisteredForProvider(
  provider: string,
  modelName: string,
): boolean {
  return registryByProvider.get(provider)?.has(modelName) ?? false;
}

/**
 * Look up pricing for a model by name.
 * Returns undefined if the model is unknown or has no pricing data.
 */
export function lookupModelPricing(
  modelName: string,
  provider?: string,
): ModelPricing | undefined {
  return lookupModelEntry(modelName, provider)?.pricing;
}

// ─── Internal ───────────────────────────────────────────────

function loadModelsFromDir(dir: string): void {
  if (!existsSync(dir)) return;

  let entries: string[];
  try {
    entries = readdirSync(dir);
  } catch {
    return;
  }

  for (const filename of entries) {
    if (!filename.endsWith(".toml")) continue;
    const filePath = join(dir, filename);
    try {
      loadModelFile(filePath);
    } catch (err) {
      const message = extractErrorMessage(err);
      console.error(`[model-registry] Failed to load ${filePath}: ${message}`);
    }
  }
}

function loadModelFile(filePath: string): void {
  const content = readFileSync(filePath, "utf-8");
  parseModelToml(content);
}
function parseModelToml(content: string): void {
  const parsed = parseToml(content) as Record<string, unknown>;

  const provider = parsed["provider"] as string | undefined;
  if (!provider) return;

  for (const [key, value] of Object.entries(parsed)) {
    if (!isModelTomlEntry(key, value)) continue;
    registerModelTomlEntry(provider, parsed["base_url"] as string | undefined, key, value);
  }
}

function isModelTomlEntry(key: string, value: unknown): value is Record<string, unknown> {
  if (key === "provider" || key === "base_url") return false;
  return typeof value === "object" && value !== null;
}

function registerModelTomlEntry(
  provider: string,
  baseUrl: string | undefined,
  key: string,
  modelDef: Record<string, unknown>,
) {
  const providerRegistry = getProviderRegistry(provider);
  if (providerRegistry.has(key)) return;

  const entry = createModelRegistryEntry(provider, baseUrl, modelDef);
  providerRegistry.set(key, entry);

  const entries = registry.get(key) ?? [];
  registry.set(key, [...entries, entry]);
}

function getProviderRegistry(provider: string) {
  const providerRegistry = registryByProvider.get(provider) ?? new Map<string, ModelRegistryEntry>();
  registryByProvider.set(provider, providerRegistry);
  return providerRegistry;
}

function createModelRegistryEntry(
  provider: string,
  baseUrl: string | undefined,
  modelDef: Record<string, unknown>,
): ModelRegistryEntry {
  return {
    provider,
    baseUrl,
    contextWindow: (modelDef["context_window"] as number | undefined) ?? 128000,
    responseHeadroom: (modelDef["response_headroom"] as number | undefined) ?? 4096,
    capabilities: parseModelCapabilities(modelDef),
    pricing: parseModelPricing(modelDef),
  };
}

function parseModelCapabilities(modelDef: Record<string, unknown>): ModelCapabilities {
  const defaultMaxTokens =
    (modelDef["default_max_tokens"] as number | undefined) ??
    (modelDef["response_headroom"] as number | undefined) ??
    4096;
  return {
    useResponsesApi: (modelDef["use_responses_api"] as boolean | undefined) ?? false,
    reasoning: inferReasoning(modelDef),
    supportsTemperature: (modelDef["supports_temperature"] as boolean | undefined) ?? true,
    defaultMaxTokens,
  };
}

function parseModelPricing(modelDef: Record<string, unknown>): ModelPricing | undefined {
  const inputPrice = modelDef["input_price_per_million"] as number | undefined;
  const outputPrice = modelDef["output_price_per_million"] as number | undefined;
  return inputPrice != null && outputPrice != null
    ? { inputPricePerMillion: inputPrice, outputPricePerMillion: outputPrice }
    : undefined;
}

function loadModelsFromEmbedded(): void {
  for (const [filename, content] of Object.entries(EMBEDDED_MODEL_TOML)) {
    try {
      parseModelToml(content);
    } catch (err) {
      const message = extractErrorMessage(err);
      console.error(`[model-registry] Failed to parse embedded ${filename}: ${message}`);
    }
  }
}

/**
 * Infer whether a model supports reasoning from its TOML definition.
 * A model is considered a reasoning model if it has use_responses_api=true
 * or supports_temperature=false (reasoning models reject temperature).
 */
function inferReasoning(modelDef: Record<string, unknown>): boolean {
  if (modelDef["use_responses_api"] === true) return true;
  if (modelDef["supports_temperature"] === false) return true;
  return false;
}
