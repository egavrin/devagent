import { isModelRegisteredForProvider, loadModelRegistry } from "@devagent/runtime";

import { getGlobalConfigPath, getGlobalConfigValue, listGlobalConfigEntries, loadGlobalConfigObject, migrateLegacyGlobalConfigIfNeeded, migrateLegacyGlobalTomlIfNeeded, normalizeGlobalConfigIfNeeded, setGlobalConfigValue, writeGlobalConfigObject } from "../global-config.js";
import { formatProviderModelCompatibilityError, formatProviderModelCompatibilityHint, getProviderModelCompatibilityIssue } from "../provider-model-compat.js";
import { hasHelpFlag, writeStderr, writeStdout } from "./shared.js";

function renderConfigHelpText(): string {
  return `Usage:
  devagent config path
  devagent config get [key]
  devagent config set <key> <value>

Inspect or edit the global DevAgent config directly at ~/.config/devagent/config.toml.`;
}
const SETUP_PROVIDERS = [
  { id: "anthropic", name: "Anthropic", envVar: "ANTHROPIC_API_KEY", defaultModel: "claude-sonnet-4-20250514", hint: "Get key at https://console.anthropic.com/settings/keys" },
  { id: "openai", name: "OpenAI", envVar: "OPENAI_API_KEY", defaultModel: "gpt-5.4", hint: "Get key at https://platform.openai.com/api-keys" },
  { id: "devagent-api", name: "Devagent API", envVar: "DEVAGENT_API_KEY", defaultModel: "cortex", hint: "Use a gateway virtual key starting with ilg_" },
  { id: "deepseek", name: "DeepSeek", envVar: "DEEPSEEK_API_KEY", defaultModel: "deepseek-chat", hint: "Get key at https://platform.deepseek.com/api_keys" },
  { id: "openrouter", name: "OpenRouter", envVar: "OPENROUTER_API_KEY", defaultModel: "anthropic/claude-sonnet-4-20250514", hint: "Get key at https://openrouter.ai/keys" },
  { id: "ollama", name: "Ollama (local)", envVar: "", defaultModel: "qwen3:32b", hint: "No API key needed — ollama must be running locally" },
  { id: "chatgpt", name: "ChatGPT (Pro/Plus)", envVar: "", defaultModel: "gpt-5.4", hint: "Use 'devagent auth login' after configuration" },
  { id: "github-copilot", name: "GitHub Copilot", envVar: "", defaultModel: "gpt-4o", hint: "Use 'devagent auth login' after configuration" },
];

function getSetupProvider(providerId: string): (typeof SETUP_PROVIDERS)[number] | undefined {
  return SETUP_PROVIDERS.find((provider) => provider.id === providerId);
}

function getDefaultModelForProvider(providerId: string): string | undefined {
  return getSetupProvider(providerId)?.defaultModel;
}
function setValidatedGlobalConfigValue(path: string, rawValue: string): Array<[string, string]> {
  const canonicalPath = path.trim().toLowerCase();
  const config = loadGlobalConfigObject();
  if (canonicalPath === "provider") return setProviderConfigValue(config, rawValue);
  if (canonicalPath === "model") return setModelConfigValue(config, rawValue);
  if (canonicalPath === "safety.mode" || canonicalPath === "approval.mode") {
    setGlobalConfigValue(path, rawValue);
    return [["safety.mode", getGlobalConfigValue("safety.mode") ?? rawValue]];
  }

  setGlobalConfigValue(path, rawValue);
  return [[path, rawValue]];
}

function setProviderConfigValue(config: Record<string, unknown>, rawValue: string): Array<[string, string]> {
  const provider = rawValue.trim();
  const currentModel = typeof config["model"] === "string" ? config["model"] : undefined;
  loadModelRegistry();
  const currentModelSupported = currentModel ? isModelRegisteredForProvider(provider, currentModel) : false;
  const defaultModel = getDefaultModelForProvider(provider);
  config["provider"] = provider;

  if ((!currentModel || !currentModelSupported) && defaultModel) {
    config["model"] = defaultModel;
    writeGlobalConfigObject(config);
    return [["provider", provider], ["model", defaultModel]];
  }

  writeGlobalConfigObject(config);
  return [["provider", provider]];
}

function setModelConfigValue(config: Record<string, unknown>, rawValue: string): Array<[string, string]> {
  const provider = typeof config["provider"] === "string" ? config["provider"] : undefined;
  if (provider) {
    validateProviderModel(provider, rawValue);
  }
  setGlobalConfigValue("model", rawValue);
  return [["model", rawValue]];
}

function validateProviderModel(provider: string, model: string): void {
  loadModelRegistry();
  const providerModelIssue = getProviderModelCompatibilityIssue(provider, model);
  if (!providerModelIssue) return;
  writeStderr(formatProviderModelCompatibilityError(providerModelIssue));
  const hint = formatProviderModelCompatibilityHint(providerModelIssue);
  if (hint) writeStderr(hint);
  process.exit(2);
}
export function runConfig(args: string[]): void {
  if (hasHelpFlag(args)) {
    writeStdout(renderConfigHelpText());
    return;
  }

  migrateLegacyGlobalConfigIfNeeded();
  migrateLegacyGlobalTomlIfNeeded();
  normalizeGlobalConfigIfNeeded();
  const sub = args[0];

  if (!sub || sub === "path") {
    writeStdout(getGlobalConfigPath());
    return;
  }

  if (sub === "get") {
    runConfigGet(args[1]);
    return;
  }

  if (sub === "set") {
    runConfigSet(args[1], args[2]);
    return;
  }

  writeStderr(`Unknown config subcommand: ${sub}`);
  writeStderr("Usage: devagent config {get|set|path}");
  process.exit(2);
}

function runConfigGet(key: string | undefined): void {
  if (!key) {
    writeConfigEntries();
    return;
  }
  const value = getGlobalConfigValue(key);
  writeStdout(value === undefined ? "(not set)" : value);
}

function writeConfigEntries(): void {
  const entries = listGlobalConfigEntries();
  if (entries.length === 0) {
    writeStdout("(no config set)");
    return;
  }
  for (const [entryKey, entryValue] of entries) {
    writeStdout(`${entryKey} = ${entryValue}`);
  }
}

function runConfigSet(key: string | undefined, value: string | undefined): void {
  if (!key || value === undefined) {
    writeStderr("Usage: devagent config set <key> <value>");
    process.exit(2);
  }
  for (const [updatedKey, updatedValue] of setValidatedGlobalConfigValue(key, value)) {
    writeStdout(`${updatedKey} = ${updatedValue}`);
  }
}
