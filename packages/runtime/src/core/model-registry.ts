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

import { parse as parseToml } from "smol-toml";
import { readFileSync, readdirSync, existsSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";
import type { ModelCapabilities } from "./types.js";
import { extractErrorMessage } from "./errors.js";
import { EMBEDDED_MODEL_TOML } from "./embedded-models.js";

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

const registry = new Map<string, ModelRegistryEntry>();
let loaded = false;

/** Reset registry state. Test-only — not exported from package barrel. */
export function _resetRegistryForTesting(): void {
  registry.clear();
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
): ModelCapabilities | undefined {
  const entry = registry.get(modelName);
  return entry?.capabilities;
}

/**
 * Look up a full model registry entry by name.
 */
export function lookupModelEntry(
  modelName: string,
): ModelRegistryEntry | undefined {
  return registry.get(modelName);
}

/**
 * Get all registered model names.
 */
export function getRegisteredModels(): ReadonlyArray<string> {
  return Array.from(registry.keys());
}

/**
 * Look up pricing for a model by name.
 * Returns undefined if the model is unknown or has no pricing data.
 */
export function lookupModelPricing(
  modelName: string,
): ModelPricing | undefined {
  return registry.get(modelName)?.pricing;
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
  const baseUrl = parsed["base_url"] as string | undefined;

  if (!provider) return;

  for (const [key, value] of Object.entries(parsed)) {
    if (key === "provider" || key === "base_url") continue;
    if (typeof value !== "object" || value === null) continue;

    const modelDef = value as Record<string, unknown>;

    // Skip if already registered (first-found wins)
    if (registry.has(key)) continue;

    const defaultMaxTokens =
      (modelDef["default_max_tokens"] as number | undefined) ??
      (modelDef["response_headroom"] as number | undefined) ??
      4096;

    const capabilities: ModelCapabilities = {
      useResponsesApi: (modelDef["use_responses_api"] as boolean | undefined) ?? false,
      reasoning: inferReasoning(modelDef),
      supportsTemperature: (modelDef["supports_temperature"] as boolean | undefined) ?? true,
      defaultMaxTokens,
    };

    const inputPrice = modelDef["input_price_per_million"] as number | undefined;
    const outputPrice = modelDef["output_price_per_million"] as number | undefined;
    const pricing: ModelPricing | undefined =
      inputPrice != null && outputPrice != null
        ? { inputPricePerMillion: inputPrice, outputPricePerMillion: outputPrice }
        : undefined;

    registry.set(key, {
      provider,
      baseUrl,
      contextWindow: (modelDef["context_window"] as number | undefined) ?? 128000,
      responseHeadroom: (modelDef["response_headroom"] as number | undefined) ?? 4096,
      capabilities,
      pricing,
    });
  }
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
