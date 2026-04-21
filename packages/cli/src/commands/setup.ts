import { existsSync } from "node:fs";
import { createInterface } from "node:readline";

import { getGlobalConfigPath, loadGlobalConfigObject, migrateLegacyGlobalConfigIfNeeded, migrateLegacyGlobalTomlIfNeeded, normalizeGlobalConfigIfNeeded, writeGlobalConfigObject } from "../global-config.js";
import { hasHelpFlag, writeStderr, writeStdout } from "./shared.js";

type SetupAsk = (prompt: string) => Promise<string>;

interface SetupProvider {
  readonly id: string;
  readonly name: string;
  readonly envVar: string;
  readonly defaultModel: string;
  readonly hint: string;
}

interface AgentType {
  readonly id: string;
  readonly label: string;
  readonly desc: string;
}

interface SubagentDefaults {
  readonly models: Record<string, string>;
  readonly reasoning: Record<string, string>;
}

interface SubagentSelection {
  readonly agentModels: Record<string, string>;
  readonly agentReasoning: Record<string, string>;
}

interface SetupSelection {
  readonly provider: SetupProvider;
  readonly apiKey: string;
  readonly model: string;
  readonly safetyMode: string;
  readonly maxIterations: number;
  readonly subagents: SubagentSelection;
}

function renderSetupHelpText(): string {
  return `Usage:
  devagent setup

Guided onboarding for global DevAgent defaults. Writes provider, model, safety,
budget, and subagent settings to ~/.config/devagent/config.toml.`;
}
function renderInitHelpText(): string {
  return `Usage:
  devagent init

This command has been removed from the public CLI.
DevAgent no longer scaffolds project instruction files automatically.
Create AGENTS.md manually when you want repository-specific guidance.`;
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
] satisfies ReadonlyArray<SetupProvider>;

const AGENT_TYPES = [
  { id: "general", label: "General", desc: "default agent for code tasks" },
  { id: "explore", label: "Explore", desc: "fast codebase search, read-only" },
  { id: "reviewer", label: "Reviewer", desc: "code review, read-only" },
  { id: "architect", label: "Architect", desc: "design and planning, read-only" },
] satisfies ReadonlyArray<AgentType>;

export async function runSetup(args: ReadonlyArray<string> = []): Promise<void> {
  if (hasHelpFlag(args)) {
    writeStdout(renderSetupHelpText());
    return;
  }

  await runConfigure(args);
}

function createSetupAsk(): { readonly ask: SetupAsk; readonly close: () => void } {
  const rl = createInterface({ input: process.stdin, output: process.stderr });
  return {
    ask: (prompt: string) => new Promise((resolve) => rl.question(prompt, resolve)),
    close: () => rl.close(),
  };
}

function writeSetupIntro(configPath: string): void {
  writeStdout("DevAgent Setup\n");
  if (existsSync(configPath)) writeStdout(`(Existing config at ${configPath} will be updated)\n`);
}

async function selectSetupProvider(ask: SetupAsk): Promise<SetupProvider> {
  writeStdout("Select your LLM provider:\n");
  for (let i = 0; i < SETUP_PROVIDERS.length; i++) {
    const provider = SETUP_PROVIDERS[i]!;
    writeStdout(`  ${i + 1}. ${provider.name}`);
    writeStdout(`     ${provider.hint}`);
  }
  writeStdout("");

  const providerChoice = await ask(`> Provider (1-${SETUP_PROVIDERS.length}) [1]: `);
  const providerIdx = (parseInt(providerChoice.trim(), 10) || 1) - 1;
  const provider = SETUP_PROVIDERS[Math.max(0, Math.min(providerIdx, SETUP_PROVIDERS.length - 1))]!;
  writeStdout(`\n  ✓ Provider: ${provider.name}\n`);
  return provider;
}

async function promptApiKey(provider: SetupProvider, ask: SetupAsk): Promise<string> {
  if (provider.envVar) {
    return promptProviderApiKey(provider, ask);
  }
  if (provider.id === "ollama") {
    writeStdout("  No API key needed for Ollama.\n");
  } else {
    writeStdout("  Run 'devagent auth login' after configuration to authenticate.\n");
  }
  return "";
}

async function promptProviderApiKey(provider: SetupProvider, ask: SetupAsk): Promise<string> {
  const existing = process.env[provider.envVar];
  if (existing) {
    writeStdout(`  ${provider.envVar} already set in environment.`);
    writeStdout(`  (Will use environment variable at runtime)\n`);
    return "";
  }

  const apiKey = (await ask(`> ${provider.envVar}: `)).trim();
  if (apiKey) {
    writeStdout(`  ✓ API key stored\n`);
  } else {
    writeStdout(`  (Skipped — set ${provider.envVar} in your shell profile later)\n`);
  }
  return apiKey;
}

async function promptModel(provider: SetupProvider, ask: SetupAsk): Promise<string> {
  const modelChoice = await ask(`> Model [${provider.defaultModel}]: `);
  const model = modelChoice.trim() || provider.defaultModel;
  writeStdout(`  ✓ Model: ${model}\n`);
  return model;
}

async function promptSafetyMode(ask: SetupAsk): Promise<string> {
  writeStdout("Safety mode:");
  writeStdout("  1. autopilot — allow everything without prompts (recommended)");
  writeStdout("  2. default — auto-allow workspace edits and safe repo commands");
  writeStdout("");
  const approvalChoice = await ask("> Safety mode (1-2) [1]: ");
  const safetyModes = ["autopilot", "default"];
  const approvalIdx = (parseInt(approvalChoice.trim(), 10) || 1) - 1;
  const safetyMode = safetyModes[Math.max(0, Math.min(approvalIdx, 1))]!;
  writeStdout(`  ✓ Safety mode: ${safetyMode}\n`);
  return safetyMode;
}

async function promptMaxIterations(ask: SetupAsk): Promise<number> {
  const iterChoice = await ask("> Max iterations per query [0]: ");
  const parsedMaxIterations = Number.parseInt(iterChoice.trim(), 10);
  const maxIterations = Number.isNaN(parsedMaxIterations) ? 0 : parsedMaxIterations;
  writeStdout(`  ✓ Max iterations: ${maxIterations}\n`);
  return maxIterations;
}

function getSubagentDefaults(providerId: string, model: string): SubagentDefaults {
  const subagentDefaults: Record<string, SubagentDefaults> = {
    anthropic: {
      models: { general: model, explore: "claude-haiku-4-20250414", reviewer: model, architect: model },
      reasoning: { general: "medium", explore: "low", reviewer: "high", architect: "high" },
    },
    openai: {
      models: { general: model, explore: "gpt-5.4-mini", reviewer: "gpt-5.4", architect: "gpt-5.4" },
      reasoning: { general: "medium", explore: "low", reviewer: "high", architect: "high" },
    },
    "devagent-api": {
      models: { general: model, explore: model, reviewer: model, architect: model },
      reasoning: { general: "high", explore: "low", reviewer: "high", architect: "high" },
    },
    deepseek: {
      models: { general: model, explore: model, reviewer: model, architect: model },
      reasoning: { general: "medium", explore: "low", reviewer: "high", architect: "high" },
    },
  };
  return subagentDefaults[providerId] ?? {
    models: { general: model, explore: model, reviewer: model, architect: model },
    reasoning: { general: "medium", explore: "low", reviewer: "high", architect: "high" },
  };
}

function writeSubagentDefaults(defaults: SubagentDefaults, model: string): void {
  writeStdout("Subagent configuration:");
  writeStdout("  DevAgent spawns specialized subagents for different tasks.");
  writeStdout("  You can use cheaper/faster models for simple tasks like exploration.\n");
  writeStdout("  Defaults for your provider:");
  for (const agent of AGENT_TYPES) {
    const m = defaults.models[agent.id] ?? model;
    const r = defaults.reasoning[agent.id] ?? "medium";
    writeStdout(`    ${agent.label.padEnd(10)} model=${m}  reasoning=${r}`);
  }
  writeStdout("");
}

async function promptSubagentConfig(provider: SetupProvider, model: string, ask: SetupAsk): Promise<SubagentSelection> {
  const defaults = getSubagentDefaults(provider.id, model);
  writeSubagentDefaults(defaults, model);

  const agentModels: Record<string, string> = { ...defaults.models };
  const agentReasoning: Record<string, string> = { ...defaults.reasoning };
  const customizeSub = await ask("> Customize subagent models? (y/N) [N]: ");
  if (customizeSub.trim().toLowerCase() !== "y") {
    writeStdout("  ✓ Using defaults\n");
    return { agentModels, agentReasoning };
  }

  await promptCustomSubagents(ask, defaults, model, agentModels, agentReasoning);
  return { agentModels, agentReasoning };
}

async function promptCustomSubagents(
  ask: SetupAsk,
  defaults: SubagentDefaults,
  model: string,
  agentModels: Record<string, string>,
  agentReasoning: Record<string, string>,
): Promise<void> {
  writeStdout("");
  for (const agent of AGENT_TYPES) {
    const defModel = defaults.models[agent.id] ?? model;
    const defReasoning = defaults.reasoning[agent.id] ?? "medium";
    const mChoice = await ask(`  > ${agent.label} model [${defModel}]: `);
    const rChoice = await ask(`  > ${agent.label} reasoning (low/medium/high) [${defReasoning}]: `);
    agentModels[agent.id] = mChoice.trim() || defModel;
    agentReasoning[agent.id] = normalizeReasoningChoice(rChoice, defReasoning);
    writeStdout(`    ✓ ${agent.label}: ${agentModels[agent.id]} (${agentReasoning[agent.id]})\n`);
  }
}

function normalizeReasoningChoice(choice: string, fallback: string): string {
  const reasoning = choice.trim().toLowerCase();
  return reasoning === "low" || reasoning === "medium" || reasoning === "high" ? reasoning : fallback;
}

async function collectSetupSelection(ask: SetupAsk): Promise<SetupSelection> {
  const provider = await selectSetupProvider(ask);
  const apiKey = await promptApiKey(provider, ask);
  const model = await promptModel(provider, ask);
  const safetyMode = await promptSafetyMode(ask);
  const maxIterations = await promptMaxIterations(ask);
  const subagents = await promptSubagentConfig(provider, model, ask);
  return { provider, apiKey, model, safetyMode, maxIterations, subagents };
}

function writeSetupConfig(selection: SetupSelection): void {
  const nextConfig = loadGlobalConfigObject();
  nextConfig["provider"] = selection.provider.id;
  nextConfig["model"] = selection.model;
  nextConfig["safety"] = {
    ...((nextConfig["safety"] as Record<string, unknown> | undefined) ?? {}),
    mode: selection.safetyMode,
  };
  nextConfig["budget"] = {
    ...((nextConfig["budget"] as Record<string, unknown> | undefined) ?? {}),
    max_iterations: selection.maxIterations,
  };

  writeSubagentConfig(nextConfig, selection);
  writeProviderConfig(nextConfig, selection);
  writeGlobalConfigObject(nextConfig);
}

function writeSubagentConfig(nextConfig: Record<string, unknown>, selection: SetupSelection): void {
  const hasCustomModels = Object.entries(selection.subagents.agentModels).some(([, value]) => value !== selection.model);
  const subagents = { ...((nextConfig["subagents"] as Record<string, unknown> | undefined) ?? {}) };

  if (hasCustomModels) {
    subagents["agent_model_overrides"] = buildModelOverrides(selection);
  }

  subagents["agent_reasoning_overrides"] = buildReasoningOverrides(selection.subagents.agentReasoning);
  nextConfig["subagents"] = subagents;
}

function buildModelOverrides(selection: SetupSelection): Record<string, string> {
  const modelOverrides: Record<string, string> = {};
  for (const agent of AGENT_TYPES) {
    const model = selection.subagents.agentModels[agent.id];
    if (model && model !== selection.model) {
      modelOverrides[agent.id] = model;
    }
  }
  return modelOverrides;
}

function buildReasoningOverrides(agentReasoning: Record<string, string>): Record<string, string> {
  const reasoningOverrides: Record<string, string> = {};
  for (const agent of AGENT_TYPES) {
    reasoningOverrides[agent.id] = agentReasoning[agent.id] ?? "medium";
  }
  return reasoningOverrides;
}

function writeProviderConfig(nextConfig: Record<string, unknown>, selection: SetupSelection): void {
  if (selection.apiKey) {
    writeProviderApiKey(nextConfig, selection.provider.id, selection.apiKey);
  }
  if (selection.provider.id === "ollama") {
    writeOllamaBaseUrl(nextConfig);
  }
}

function getProviderConfig(nextConfig: Record<string, unknown>): Record<string, unknown> {
  return { ...((nextConfig["providers"] as Record<string, unknown> | undefined) ?? {}) };
}

function writeProviderApiKey(nextConfig: Record<string, unknown>, providerId: string, apiKey: string): void {
  const providers = getProviderConfig(nextConfig);
  providers[providerId] = {
    ...((providers[providerId] as Record<string, unknown> | undefined) ?? {}),
    api_key: apiKey,
  };
  nextConfig["providers"] = providers;
}

function writeOllamaBaseUrl(nextConfig: Record<string, unknown>): void {
  const providers = getProviderConfig(nextConfig);
  providers["ollama"] = {
    ...((providers["ollama"] as Record<string, unknown> | undefined) ?? {}),
    base_url: "http://localhost:11434/v1",
  };
  nextConfig["providers"] = providers;
}

function writeSetupNextSteps(configPath: string, provider: SetupProvider, apiKey: string): void {
  writeStdout(`Config written to ${configPath}\n`);
  writeStdout("Next steps:");
  if (!apiKey && provider.envVar) {
    writeStdout(`  1. Set ${provider.envVar} in your shell profile`);
    writeStdout(`  2. Run 'devagent doctor' to verify`);
    writeStdout(`  3. Run 'devagent "hello"' to test`);
  } else if (provider.id === "chatgpt" || provider.id === "github-copilot") {
    writeStdout(`  1. Run 'devagent auth login' to authenticate`);
    writeStdout(`  2. Run 'devagent doctor' to verify`);
    writeStdout(`  3. Run 'devagent "hello"' to test`);
  } else {
    writeStdout(`  1. Run 'devagent doctor' to verify`);
    writeStdout(`  2. Run 'devagent "hello"' to test`);
  }
}
export async function runConfigure(args: ReadonlyArray<string> = []): Promise<void> {
  if (hasHelpFlag(args)) {
    writeStdout(renderSetupHelpText());
    return;
  }

  migrateLegacyGlobalConfigIfNeeded();
  migrateLegacyGlobalTomlIfNeeded();
  normalizeGlobalConfigIfNeeded();
  const configPath = getGlobalConfigPath();
  const prompt = createSetupAsk();
  writeSetupIntro(configPath);
  const selection = await collectSetupSelection(prompt.ask);
  prompt.close();
  writeSetupConfig(selection);
  writeSetupNextSteps(configPath, selection.provider, selection.apiKey);
}
export function runInit(args: ReadonlyArray<string> = []): void {
  if (hasHelpFlag(args)) {
    writeStdout(renderInitHelpText());
    return;
  }

  writeStderr("devagent init has been removed from the public CLI.");
  writeStderr("DevAgent no longer scaffolds project instruction files automatically.");
  writeStderr("Create AGENTS.md manually when you want repository-specific guidance.");
  process.exit(2);
}
