import {
  existsSync,
  mkdirSync,
  readFileSync,
  renameSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { parse as parseToml, stringify as stringifyToml } from "smol-toml";

const GLOBAL_CONFIG_DIRNAME = join(".config", "devagent");
const GLOBAL_CONFIG_FILENAME = "config.toml";
const LEGACY_CONFIG_FILENAME = "config.json";
const LEGACY_GLOBAL_TOML_FILENAME = ".devagent.toml";
const AGENT_TYPES = new Set(["general", "explore", "reviewer", "architect"]);
const TOOL_CATEGORIES = new Set(["readonly", "mutating", "workflow", "external", "state"]);
const ROOT_KEYS = new Set(["provider", "model", "api_key"]);
const SECTION_FIELDS = new Map<string, ReadonlySet<string>>([
  ["safety", new Set(["mode"])],
  ["budget", new Set(["max_iterations", "max_context_tokens", "response_headroom", "cost_warning_threshold", "enable_cost_tracking"])],
  ["context", new Set(["pruning_strategy", "trigger_ratio", "keep_recent_messages", "turn_isolation", "midpoint_briefing_interval", "briefing_strategy", "prune_protect_tokens"])],
  ["arkts", new Set(["enabled", "strict_mode", "target_version", "linter_path"])],
  ["logging", new Set(["enabled", "log_dir", "retention_days"])],
  ["double_check", new Set(["enabled", "check_diagnostics", "run_tests", "test_command", "diagnostic_timeout"])],
  ["session_state", new Set(["persist", "track_plan", "track_files", "track_env", "track_tool_results", "track_findings", "track_knowledge", "max_modified_files", "max_env_facts", "max_tool_summaries", "max_findings", "max_knowledge"])],
]);
const PROVIDER_FIELDS = new Set([
  "api_key",
  "base_url",
  "model",
  "max_tokens",
  "temperature",
  "reasoning_effort",
  "use_responses_api",
  "reasoning",
  "supports_temperature",
  "default_max_tokens",
  "fallback_model",
]);
const MIGRATION_NOTICE_ONCE = new Set<string>();
const SAFETY_MODES = new Set(["default", "autopilot"]);
const LEGACY_APPROVAL_MODE_MAP = new Map<string, string>([
  ["suggest", "default"],
  ["auto-edit", "default"],
  ["full-auto", "autopilot"],
]);

interface GlobalConfigMigrationResult {
  readonly migrated: boolean;
  readonly backupPath?: string;
}

function getGlobalConfigDir(home: string = process.env["HOME"] ?? homedir()): string {
  return join(home, GLOBAL_CONFIG_DIRNAME);
}

export function getGlobalConfigPath(home: string = process.env["HOME"] ?? homedir()): string {
  return join(getGlobalConfigDir(home), GLOBAL_CONFIG_FILENAME);
}

function getLegacyGlobalConfigPath(home: string = process.env["HOME"] ?? homedir()): string {
  return join(getGlobalConfigDir(home), LEGACY_CONFIG_FILENAME);
}

function getLegacyGlobalTomlPath(home: string = process.env["HOME"] ?? homedir()): string {
  return join(home, LEGACY_GLOBAL_TOML_FILENAME);
}

export function migrateLegacyGlobalConfigIfNeeded(
  notify?: (message: string) => void,
): GlobalConfigMigrationResult {
  const configPath = getGlobalConfigPath();
  const legacyPath = getLegacyGlobalConfigPath();
  if (existsSync(configPath) || !existsSync(legacyPath)) {
    return { migrated: false };
  }

  const raw = readFileSync(legacyPath, "utf-8");
  const parsed = JSON.parse(raw) as Record<string, unknown>;
  const next = {} as Record<string, unknown>;
  for (const [path, value] of flattenObject(parsed)) {
    try {
      const resolved = resolveConfigAssignment(path, value);
      setNestedValue(next, resolved.path, resolved.value);
    } catch {
      // Preserve unknown keys in the backup only.
    }
  }

  writeGlobalConfigObject(next);
  const backupPath = join(
    getGlobalConfigDir(),
    `config.json.bak.${new Date().toISOString().replace(/[:]/g, "-")}`,
  );
  writeFileSync(backupPath, raw);
  rmSync(legacyPath);

  if (!MIGRATION_NOTICE_ONCE.has(configPath)) {
    MIGRATION_NOTICE_ONCE.add(configPath);
    notify?.(`Migrated legacy config.json to ${configPath} (backup: ${backupPath})`);
  }

  return { migrated: true, backupPath };
}

interface LegacyTomlMigrationResult {
  readonly migrated: boolean;
  readonly sourcePath?: string;
  readonly backupPath?: string;
}

interface GlobalConfigNormalizationResult {
  readonly normalized: boolean;
}

export function migrateLegacyGlobalTomlIfNeeded(
  notify?: (message: string) => void,
): LegacyTomlMigrationResult {
  const configPath = getGlobalConfigPath();
  const legacyPath = getLegacyGlobalTomlPath();
  if (existsSync(configPath) || !existsSync(legacyPath)) {
    return { migrated: false };
  }

  const configDir = getGlobalConfigDir();
  mkdirSync(configDir, { recursive: true });
  const raw = readFileSync(legacyPath, "utf-8");
  const parsed = parseToml(raw) as Record<string, unknown>;
  writeGlobalConfigObject(parsed);

  const backupPath = join(
    configDir,
    `.legacy-global-config.bak.${new Date().toISOString().replace(/[:]/g, "-")}.toml`,
  );
  writeFileSync(backupPath, raw);
  rmSync(legacyPath);

  if (!MIGRATION_NOTICE_ONCE.has(legacyPath)) {
    MIGRATION_NOTICE_ONCE.add(legacyPath);
    notify?.(`Migrated legacy ${legacyPath} to ${configPath} (backup: ${backupPath})`);
  }

  return { migrated: true, sourcePath: legacyPath, backupPath };
}

export function loadGlobalConfigObject(): Record<string, unknown> {
  const configPath = getGlobalConfigPath();
  if (!existsSync(configPath)) {
    return {};
  }
  const raw = readFileSync(configPath, "utf-8");
  const parsed = parseToml(raw) as Record<string, unknown>;
  const { config, changed } = normalizeGlobalConfigObject(parsed);
  if (changed) {
    writeGlobalConfigObject(config);
  }
  return config;
}

export function writeGlobalConfigObject(config: Record<string, unknown>): void {
  const configDir = getGlobalConfigDir();
  mkdirSync(configDir, { recursive: true });
  const configPath = getGlobalConfigPath();
  const tempPath = `${configPath}.tmp`;
  const { config: normalized } = normalizeGlobalConfigObject(config);
  writeFileSync(tempPath, `${stringifyToml(normalized)}\n`);
  renameSync(tempPath, configPath);
}

export function normalizeGlobalConfigIfNeeded(): GlobalConfigNormalizationResult {
  const configPath = getGlobalConfigPath();
  if (!existsSync(configPath)) {
    return { normalized: false };
  }

  const raw = readFileSync(configPath, "utf-8");
  const parsed = parseToml(raw) as Record<string, unknown>;
  const { config, changed } = normalizeGlobalConfigObject(parsed);
  if (!changed) {
    return { normalized: false };
  }

  writeGlobalConfigObject(config);
  return { normalized: true };
}

export function getGlobalConfigValue(path: string): string | undefined {
  const config = loadGlobalConfigObject();
  const value = getNestedValue(config, resolveConfigPath(path).path);
  return value === undefined ? undefined : formatValue(value);
}

export function listGlobalConfigEntries(): Array<[string, string]> {
  return flattenObject(loadGlobalConfigObject()).map(([path, value]) => [
    path,
    formatValue(value),
  ]);
}

export function setGlobalConfigValue(path: string, rawValue: string): void {
  const config = loadGlobalConfigObject();
  const resolved = resolveConfigAssignment(path, parseValue(rawValue));
  setNestedValue(config, resolved.path, resolved.value);
  writeGlobalConfigObject(config);
}

function resolveConfigPath(
  path: string,
): { readonly path: string[]; readonly legacyApprovalAlias: boolean } {
  const rawParts = path.split(".").map((part) => part.trim()).filter(Boolean);
  if (rawParts.length === 0) {
    throw new Error(`Unsupported config key: "${path}"`);
  }

  const topLevel = toSnakeCase(rawParts[0]!);
  if (ROOT_KEYS.has(topLevel)) {
    if (rawParts.length !== 1) {
      throw new Error(`Unsupported config key: "${path}"`);
    }
    return { path: [topLevel], legacyApprovalAlias: false };
  }

  const second = rawParts[1] ? toSnakeCase(rawParts[1]) : undefined;
  if (topLevel === "approval") {
    if (rawParts.length !== 2 || second !== "mode") {
      throw new Error(`Unsupported config key: "${path}"`);
    }
    return { path: ["safety", "mode"], legacyApprovalAlias: true };
  }

  if (second && SECTION_FIELDS.has(topLevel)) {
    if (rawParts.length !== 2 || !SECTION_FIELDS.get(topLevel)!.has(second)) {
      throw new Error(`Unsupported config key: "${path}"`);
    }
    return { path: [topLevel, second], legacyApprovalAlias: false };
  }

  if (topLevel === "providers") {
    const providerId = rawParts[1];
    const field = rawParts[2] ? toSnakeCase(rawParts[2]) : undefined;
    if (!providerId || rawParts.length !== 3 || !field || !PROVIDER_FIELDS.has(field)) {
      throw new Error(`Unsupported config key: "${path}"`);
    }
    return { path: [topLevel, providerId, field], legacyApprovalAlias: false };
  }

  if (topLevel === "subagents") {
    if (!second) {
      throw new Error(`Unsupported config key: "${path}"`);
    }
    if (second === "subagent_timeout_ms" && rawParts.length === 2) {
      return { path: [topLevel, second], legacyApprovalAlias: false };
    }
    if (
      ["agent_model_overrides", "agent_reasoning_overrides", "agent_iteration_caps", "allowed_child_agents"].includes(second)
    ) {
      const agentType = rawParts[2] ? toSnakeCase(rawParts[2]) : undefined;
      if (!agentType || rawParts.length !== 3 || !AGENT_TYPES.has(agentType)) {
        throw new Error(`Unsupported config key: "${path}"`);
      }
      return { path: [topLevel, second, agentType], legacyApprovalAlias: false };
    }
    if (second === "agent_permission_overrides") {
      const agentType = rawParts[2] ? toSnakeCase(rawParts[2]) : undefined;
      const category = rawParts[3] ? toSnakeCase(rawParts[3]) : undefined;
      if (!agentType || !category || rawParts.length !== 4 || !AGENT_TYPES.has(agentType) || !TOOL_CATEGORIES.has(category)) {
        throw new Error(`Unsupported config key: "${path}"`);
      }
      return { path: [topLevel, second, agentType, category], legacyApprovalAlias: false };
    }
  }

  throw new Error(`Unsupported config key: "${path}"`);
}

function resolveConfigAssignment(
  path: string,
  value: unknown,
): { readonly path: string[]; readonly value: unknown } {
  const resolved = resolveConfigPath(path);
  return {
    path: resolved.path,
    value: normalizeAssignedValue(resolved.path, value, {
      originalPath: path,
      legacyApprovalAlias: resolved.legacyApprovalAlias,
    }),
  };
}

function toSnakeCase(value: string): string {
  return value.replace(/([a-z0-9])([A-Z])/g, "$1_$2").replace(/-/g, "_").toLowerCase();
}

function parseValue(raw: string): unknown {
  const trimmed = raw.trim();
  if (trimmed === "true") return true;
  if (trimmed === "false") return false;
  if (/^-?\d+(?:\.\d+)?$/.test(trimmed)) return Number(trimmed);
  if ((trimmed.startsWith("[") && trimmed.endsWith("]")) || (trimmed.startsWith("{") && trimmed.endsWith("}"))) {
    return JSON.parse(trimmed) as unknown;
  }
  return raw;
}

function formatValue(value: unknown): string {
  if (typeof value === "string") return value;
  if (typeof value === "object" && value !== null) return JSON.stringify(value);
  return String(value);
}

function getNestedValue(obj: Record<string, unknown>, path: ReadonlyArray<string>): unknown {
  let current: unknown = obj;
  for (const part of path) {
    if (!current || typeof current !== "object" || Array.isArray(current)) {
      return undefined;
    }
    current = (current as Record<string, unknown>)[part];
  }
  return current;
}

function setNestedValue(obj: Record<string, unknown>, path: ReadonlyArray<string>, value: unknown): void {
  let current = obj;
  for (let index = 0; index < path.length - 1; index++) {
    const part = path[index]!;
    const existing = current[part];
    if (!existing || typeof existing !== "object" || Array.isArray(existing)) {
      current[part] = {};
    }
    current = current[part] as Record<string, unknown>;
  }
  current[path[path.length - 1]!] = value;
}

function normalizeAssignedValue(
  path: ReadonlyArray<string>,
  value: unknown,
  options: { readonly originalPath: string; readonly legacyApprovalAlias: boolean },
): unknown {
  if (path.length === 2 && path[0] === "safety" && path[1] === "mode") {
    if (typeof value !== "string") {
      throw new Error(`Unsupported config value for "${options.originalPath}": ${String(value)}`);
    }

    if (options.legacyApprovalAlias) {
      const mapped = LEGACY_APPROVAL_MODE_MAP.get(value);
      if (mapped) {
        return mapped;
      }
      throw new Error(
        `Unsupported legacy approval.mode: "${value}". Use approval.mode = "suggest" | "auto-edit" | "full-auto", or set safety.mode = "default" | "autopilot".`,
      );
    }

    if (!SAFETY_MODES.has(value)) {
      throw new Error(
        `Unsupported safety.mode: "${value}". Expected "default" or "autopilot".`,
      );
    }
  }

  return value;
}

function flattenObject(
  obj: Record<string, unknown>,
  prefix = "",
): Array<[string, unknown]> {
  const entries: Array<[string, unknown]> = [];
  for (const [key, value] of Object.entries(obj)) {
    const nextKey = prefix ? `${prefix}.${key}` : key;
    if (isPlainObject(value)) {
      entries.push(...flattenObject(value, nextKey));
      continue;
    }
    entries.push([nextKey, value]);
  }
  return entries;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function normalizeGlobalConfigObject(
  config: Record<string, unknown>,
): { readonly config: Record<string, unknown>; readonly changed: boolean } {
  const next = { ...config };
  let changed = false;

  const safety = isPlainObject(next["safety"])
    ? (next["safety"])
    : undefined;
  const approval = isPlainObject(next["approval"])
    ? { ...(next["approval"]) }
    : undefined;

  let nextSafety = safety ? { ...safety } : undefined;
  let safetyChanged = false;
  if (approval) {
    if (nextSafety?.["mode"] === undefined && approval["mode"] !== undefined) {
      nextSafety = {
        ...(nextSafety ?? {}),
        mode: normalizeAssignedValue(["safety", "mode"], approval["mode"], {
          originalPath: "approval.mode",
          legacyApprovalAlias: true,
        }),
      };
      safetyChanged = true;
    }
  }

  if (safetyChanged && nextSafety) {
    next["safety"] = nextSafety;
    changed = true;
  }

  if (next["approval"] !== undefined) {
    delete next["approval"];
    changed = true;
  }

  return { config: next, changed };
}
