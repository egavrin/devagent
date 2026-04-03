/**
 * CLI subcommands: doctor, config, init.
 */

import { existsSync, readFileSync, writeFileSync, mkdirSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";
import { execSync } from "node:child_process";
import { createInterface } from "node:readline";
import {
  CredentialStore,
  findProjectRoot,
  loadConfig,
  loadModelRegistry,
  getRegisteredModels,
  lookupModelEntry,
} from "@devagent/runtime";
import type { CredentialInfo, DevAgentConfig } from "@devagent/runtime";
import type { ProviderModelCompatibilityIssue } from "./provider-model-compat.js";
import {
  formatProviderModelCompatibilityError,
  formatProviderModelCompatibilityHint,
  getProviderModelCompatibilityIssue,
} from "./provider-model-compat.js";

type DoctorCheckStatus = "pass" | "blocking" | "advisory";

export interface DoctorCheck {
  readonly label: string;
  readonly status: DoctorCheckStatus;
  readonly detail?: string;
}

export interface DoctorIssue {
  readonly title: string;
  readonly detail: string;
  readonly nextSteps: ReadonlyArray<string>;
}

export interface DoctorProviderStatus {
  readonly id: string;
  readonly hint: string;
  readonly active: boolean;
  readonly hasCredential: boolean;
}

export interface DoctorLspStatus {
  readonly label: string;
  readonly found: boolean;
  readonly install: string;
}

export interface DoctorProviderCredentialIssue {
  readonly status: "blocking" | "advisory";
  readonly detail: string;
}

export interface DoctorReportInput {
  readonly version: string;
  readonly runtimeLabel: string;
  readonly runtimeError?: string;
  readonly gitError?: string;
  readonly configPath?: string;
  readonly configSearchPaths: ReadonlyArray<string>;
  readonly config: DevAgentConfig;
  readonly providerStatuses: ReadonlyArray<DoctorProviderStatus>;
  readonly providerCredentialIssue?: DoctorProviderCredentialIssue;
  readonly modelRegistryError?: string;
  readonly modelRegistryCount?: number;
  readonly modelRegistered: boolean;
  readonly modelOwner?: string;
  readonly providerModelIssue?: ProviderModelCompatibilityIssue;
  readonly lspStatuses: ReadonlyArray<DoctorLspStatus>;
  readonly platformLabel: string;
  readonly providerSource: "cli" | "env" | "config" | "default";
  readonly modelSource: "cli" | "env" | "config" | "default";
  readonly credentialSource: string;
}

export interface DoctorReport {
  readonly version: string;
  readonly blockingIssues: ReadonlyArray<DoctorIssue>;
  readonly runtimeCheck: DoctorCheck;
  readonly gitCheck: DoctorCheck;
  readonly configCheck: DoctorCheck;
  readonly providerCheck: DoctorCheck;
  readonly providerStatuses: ReadonlyArray<DoctorProviderStatus>;
  readonly modelRegistryCheck: DoctorCheck;
  readonly modelCheck: DoctorCheck;
  readonly providerModelCheck: DoctorCheck;
  readonly effectiveConfig: {
    readonly provider: string;
    readonly providerSource: "cli" | "env" | "config" | "default";
    readonly model: string;
    readonly modelSource: "cli" | "env" | "config" | "default";
    readonly credentialSource: string;
    readonly modelOwner?: string;
  };
  readonly lspStatuses: ReadonlyArray<DoctorLspStatus>;
  readonly platformCheck: DoctorCheck;
  readonly ok: boolean;
}

interface ProviderDescriptor {
  readonly id: string;
  readonly env: string;
  readonly hint: string;
  readonly credentialMode: "api" | "oauth" | "none";
}

interface DoctorConfigFile {
  readonly text: string;
  readonly topLevelProvider?: string;
  readonly topLevelModel?: string;
}

function makeCheck(
  label: string,
  status: DoctorCheckStatus,
  detail?: string,
): DoctorCheck {
  return { label, status, ...(detail ? { detail } : {}) };
}

function statusIcon(status: DoctorCheckStatus): string {
  switch (status) {
    case "pass":
      return "✓";
    case "advisory":
      return "!";
    case "blocking":
      return "✗";
  }
}

function formatCheck(check: DoctorCheck): string {
  return check.detail
    ? `  ${statusIcon(check.status)} ${check.label}: ${check.detail}`
    : `  ${statusIcon(check.status)} ${check.label}`;
}

function getProviderDescriptor(providerId: string): ProviderDescriptor | undefined {
  return PROVIDERS.find((provider) => provider.id === providerId);
}

function hasProviderCredential(
  providerId: string,
  config: DevAgentConfig,
  storedCredentials: Readonly<Record<string, CredentialInfo>>,
): boolean {
  const descriptor = getProviderDescriptor(providerId);
  if (!descriptor || descriptor.credentialMode === "none") {
    return true;
  }

  const providerConfig = config.providers[providerId];
  const stored = storedCredentials[providerId];
  if (descriptor.credentialMode === "oauth") {
    return Boolean(providerConfig?.oauthToken || stored?.type === "oauth");
  }
  return Boolean(
    providerConfig?.apiKey ||
    (descriptor.env && process.env[descriptor.env]) ||
    stored?.type === "api",
  );
}

function buildProviderStatuses(
  config: DevAgentConfig,
  storedCredentials: Readonly<Record<string, CredentialInfo>>,
): DoctorProviderStatus[] {
  return PROVIDERS.map((provider) => ({
    id: provider.id,
    hint: provider.hint,
    active: provider.id === config.provider,
    hasCredential: hasProviderCredential(provider.id, config, storedCredentials),
  }));
}

function loadDoctorConfigFile(configPath: string | undefined): DoctorConfigFile {
  if (!configPath) {
    return { text: "" };
  }
  try {
    const text = readFileSync(configPath, "utf-8");
    return {
      text,
      ...(matchTopLevelTomlString(text, "provider") ? { topLevelProvider: matchTopLevelTomlString(text, "provider")! } : {}),
      ...(matchTopLevelTomlString(text, "model") ? { topLevelModel: matchTopLevelTomlString(text, "model")! } : {}),
    };
  } catch {
    return { text: "" };
  }
}

function resolveConfigValueSource(
  envKey: string,
  fileValue: string | undefined,
): "cli" | "env" | "config" | "default" {
  if (process.env[envKey]) {
    return "env";
  }
  if (typeof fileValue === "string" && fileValue.length > 0) {
    return "config";
  }
  return "default";
}

function matchTopLevelTomlString(text: string, key: string): string | undefined {
  const match = text.match(new RegExp(`^${key}\\s*=\\s*"([^"]+)"`, "m"));
  return match?.[1];
}

function providerHasConfigCredential(
  fileConfig: DoctorConfigFile,
  providerId: string,
): boolean {
  if (/(^|\n)api_key\s*=\s*"[^"]+"/m.test(fileConfig.text)) {
    return true;
  }
  const sectionPattern = new RegExp(`\\[providers\\.${escapeRegExp(providerId)}\\]([\\s\\S]*?)(?=\\n\\[|$)`, "m");
  const sectionMatch = fileConfig.text.match(sectionPattern);
  if (!sectionMatch) {
    return false;
  }
  return /(api_key|apiKey|oauthToken|oauth_token)\s*=\s*"[^"]+"/m.test(sectionMatch[1] ?? "");
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function resolveCredentialSource(
  providerId: string,
  config: DevAgentConfig,
  storedCredentials: Readonly<Record<string, CredentialInfo>>,
  fileConfig: DoctorConfigFile,
): string {
  const descriptor = getProviderDescriptor(providerId);
  if (!descriptor || descriptor.credentialMode === "none") {
    return `missing (not required for ${providerId})`;
  }

  const stored = storedCredentials[providerId];
  const providerConfig = config.providers[providerId];
  if (descriptor.credentialMode === "oauth") {
    if (stored?.type === "oauth") {
      return "stored oauth";
    }
    if (providerConfig?.oauthToken) {
      return "config";
    }
    return "missing";
  }

  if (descriptor.env && process.env[descriptor.env]) {
    return `env (${descriptor.env})`;
  }
  if (providerId === "devagent-api" && process.env["DEVAGENT_API_KEY"]) {
    return "env (DEVAGENT_API_KEY)";
  }
  if (providerHasConfigCredential(fileConfig, providerId)) {
    return "config";
  }
  if (stored?.type === "api") {
    return "stored api key";
  }
  return "missing";
}

function buildLspStatuses(): DoctorLspStatus[] {
  return LSP_SERVERS.map((lsp) => ({
    label: lsp.label,
    found: commandExists(lsp.command),
    install: lsp.install,
  }));
}

function buildProviderCredentialIssue(
  provider: string,
  hasCredential: boolean,
  hasProviderModelMismatch: boolean,
): DoctorProviderCredentialIssue | undefined {
  if (hasCredential) {
    return undefined;
  }

  const envKey = getProviderEnvKey(provider);
  const descriptor = getProviderDescriptor(provider);
  if (!descriptor || descriptor.credentialMode === "none") {
    return undefined;
  }

  const detail = descriptor.credentialMode === "oauth"
    ? `no stored login (run devagent auth login)${hasProviderModelMismatch ? ". Secondary until provider/model pairing is fixed." : ""}`
    : `no API key (set ${envKey ?? "DEVAGENT_API_KEY"} or run devagent auth login)${hasProviderModelMismatch ? ". Secondary until provider/model pairing is fixed." : ""}`;

  return {
    status: hasProviderModelMismatch ? "advisory" : "blocking",
    detail,
  };
}

function buildProviderModelIssueSteps(
  issue: ProviderModelCompatibilityIssue,
): string[] {
  if (issue.model === "cortex" || issue.expectedProvider === "devagent-api") {
    return [
      'Run now: devagent --provider devagent-api --model cortex "<your prompt>"',
      "Set in ~/.config/devagent/config.toml:",
      'provider = "devagent-api"',
      'model = "cortex"',
      "Export credentials: export DEVAGENT_API_KEY=ilg_...",
      "Or store credentials: devagent auth login",
    ];
  }

  return [
    `Run now: devagent --provider ${issue.expectedProvider} --model ${issue.model} "<your prompt>"`,
    `Or switch to a model registered for "${issue.configuredProvider}".`,
  ];
}

function buildProviderCredentialIssueSteps(providerId: string): string[] {
  const descriptor = getProviderDescriptor(providerId);
  if (!descriptor) {
    return ['Run: devagent auth login'];
  }
  if (descriptor.credentialMode === "oauth") {
    return [
      "Run: devagent auth login",
      'Then retry: devagent "<your prompt>"',
    ];
  }
  const placeholder = providerId === "devagent-api" ? "ilg_..." : "<your_api_key>";
  return [
    `Export credentials: export ${descriptor.env}=${placeholder}`,
    "Or store credentials: devagent auth login",
    'Then retry: devagent "<your prompt>"',
  ];
}

function buildBlockingIssues(
  input: DoctorReportInput,
  checks: {
    readonly runtimeCheck: DoctorCheck;
    readonly gitCheck: DoctorCheck;
    readonly configCheck: DoctorCheck;
    readonly providerCheck: DoctorCheck;
    readonly modelRegistryCheck: DoctorCheck;
    readonly modelCheck: DoctorCheck;
    readonly providerModelCheck: DoctorCheck;
  },
): DoctorIssue[] {
  const issues: DoctorIssue[] = [];

  if (checks.runtimeCheck.status === "blocking") {
    issues.push({
      title: "Runtime",
      detail: checks.runtimeCheck.detail ?? "runtime check failed",
      nextSteps: ["Install Node.js >= 20 or Bun >= 1.3 and retry."],
    });
  }
  if (checks.gitCheck.status === "blocking") {
    issues.push({
      title: "Git",
      detail: checks.gitCheck.detail ?? "git not found in PATH",
      nextSteps: ["Install Git and retry devagent doctor."],
    });
  }
  if (checks.configCheck.status === "blocking") {
    issues.push({
      title: "Config file",
      detail: checks.configCheck.detail ?? "config file not found",
      nextSteps: [
        "Run: devagent setup",
        "Or create ~/.config/devagent/config.toml with your provider and model.",
      ],
    });
  }
  if (input.providerModelIssue && checks.providerModelCheck.status === "blocking") {
    issues.push({
      title: "Provider/model pairing",
      detail: formatProviderModelCompatibilityError(input.providerModelIssue),
      nextSteps: buildProviderModelIssueSteps(input.providerModelIssue),
    });
  }
  if (checks.providerCheck.status === "blocking") {
    issues.push({
      title: "Provider credentials",
      detail: checks.providerCheck.detail ?? `provider "${input.config.provider}" is missing credentials`,
      nextSteps: buildProviderCredentialIssueSteps(input.config.provider),
    });
  }
  if (checks.modelRegistryCheck.status === "blocking") {
    issues.push({
      title: "Model registry",
      detail: checks.modelRegistryCheck.detail ?? "model registry failed to load",
      nextSteps: [
        "Rebuild or reinstall DevAgent so bundled model definitions are available.",
        "Then rerun: devagent doctor",
      ],
    });
  }
  if (checks.modelCheck.status === "blocking") {
    issues.push({
      title: "Model",
      detail: checks.modelCheck.detail ?? `model "${input.config.model}" is not registered`,
      nextSteps: [
        "Run: devagent setup",
        `Or choose a registered model for provider "${input.config.provider}".`,
      ],
    });
  }

  const foundLsp = input.lspStatuses.some((lsp) => lsp.found);
  if (!foundLsp) {
    issues.push({
      title: "LSP servers",
      detail: "none found",
      nextSteps: ["Run: devagent install-lsp"],
    });
  }

  return issues;
}

export function buildDoctorReport(input: DoctorReportInput): DoctorReport {
  const runtimeCheck = makeCheck(
    `Runtime: ${input.runtimeLabel}`,
    input.runtimeError ? "blocking" : "pass",
    input.runtimeError,
  );
  const gitCheck = makeCheck(
    "Git",
    input.gitError ? "blocking" : "pass",
    input.gitError,
  );
  const configCheck = makeCheck(
    "Config file",
    input.configPath ? "pass" : "blocking",
    input.configPath
      ? undefined
      : `not found (searched: ${input.configSearchPaths.join(", ")})`,
  );
  const providerCheck = makeCheck(
    `Provider: ${input.config.provider}`,
    input.providerCredentialIssue?.status ?? "pass",
    input.providerCredentialIssue?.detail,
  );
  const modelRegistryCheck = input.modelRegistryError
    ? makeCheck("Model registry", "blocking", input.modelRegistryError)
    : makeCheck(`Model registry: ${input.modelRegistryCount ?? 0} models loaded`, "pass");
  const modelCheck = input.modelRegistryError
    ? makeCheck("Model", "advisory", "skipped until model registry loads")
    : makeCheck(
        `Model: ${input.config.model}`,
        input.modelRegistered ? "pass" : "blocking",
        input.modelRegistered ? undefined : `model "${input.config.model}" not in registry`,
      );
  const providerModelCheck = input.modelRegistryError || !input.modelRegistered
    ? makeCheck("Provider/model pairing", "advisory", "skipped until the configured model is known")
    : input.providerModelIssue
      ? makeCheck(
          "Provider/model pairing",
          "blocking",
          [
            formatProviderModelCompatibilityError(input.providerModelIssue),
            formatProviderModelCompatibilityHint(input.providerModelIssue),
          ].filter(Boolean).join(" "),
        )
      : makeCheck("Provider/model pairing", "pass");
  const platformCheck = makeCheck(`Platform: ${input.platformLabel}`, "pass");

  const blockingIssues = buildBlockingIssues(input, {
    runtimeCheck,
    gitCheck,
    configCheck,
    providerCheck,
    modelRegistryCheck,
    modelCheck,
    providerModelCheck,
  });

  const foundLsp = input.lspStatuses.some((lsp) => lsp.found);
  const ok = blockingIssues.length === 0 && foundLsp;

  return {
    version: input.version,
    blockingIssues,
    runtimeCheck,
    gitCheck,
    configCheck,
    providerCheck,
    providerStatuses: input.providerStatuses,
    modelRegistryCheck,
    modelCheck,
    providerModelCheck,
    effectiveConfig: {
      provider: input.config.provider,
      providerSource: input.providerSource,
      model: input.config.model,
      modelSource: input.modelSource,
      credentialSource: input.credentialSource,
      ...(input.modelOwner ? { modelOwner: input.modelOwner } : {}),
    },
    lspStatuses: input.lspStatuses,
    platformCheck,
    ok,
  };
}

export function renderDoctorReport(report: DoctorReport): string {
  const lines: string[] = [`devagent v${report.version}`, ""];

  if (report.blockingIssues.length > 0) {
    lines.push("Blocking issues:", "");
    for (const issue of report.blockingIssues) {
      lines.push(`  - ${issue.title}: ${issue.detail}`);
    }
    lines.push("", "What to do next:", "");
    for (const issue of report.blockingIssues) {
      lines.push(`  ${issue.title}:`);
      for (const step of issue.nextSteps) {
        lines.push(`    ${step}`);
      }
    }
    lines.push("");
  }

  lines.push("Effective config:", "");
  lines.push(`  Provider: ${report.effectiveConfig.provider} (${report.effectiveConfig.providerSource})`);
  lines.push(`  Model: ${report.effectiveConfig.model} (${report.effectiveConfig.modelSource})`);
  lines.push(`  Credential: ${report.effectiveConfig.credentialSource}`);
  if (report.effectiveConfig.modelOwner) {
    lines.push(`  Model owner: ${report.effectiveConfig.modelOwner}`);
  }
  lines.push("");

  lines.push("Checks:", "");
  lines.push(formatCheck(report.runtimeCheck));
  lines.push(formatCheck(report.gitCheck));
  lines.push(formatCheck(report.configCheck));
  lines.push(formatCheck(report.providerCheck));
  lines.push("");
  lines.push("  Available providers:");
  for (const provider of report.providerStatuses) {
    const status = provider.hasCredential ? "✓" : "·";
    const active = provider.active ? " (active)" : "";
    lines.push(`    ${status} ${provider.id}${active} — ${provider.hint}`);
  }
  lines.push("");
  lines.push(formatCheck(report.modelRegistryCheck));
  lines.push(formatCheck(report.modelCheck));
  lines.push(formatCheck(report.providerModelCheck));
  lines.push("  LSP servers:");
  for (const lsp of report.lspStatuses) {
    const status = lsp.found ? "✓" : "·";
    const install = lsp.found ? "" : ` — install: ${lsp.install}`;
    lines.push(`    ${status} ${lsp.label}${install}`);
  }
  if (!report.lspStatuses.some((lsp) => lsp.found)) {
    lines.push("    (none found — run 'devagent install-lsp' to install)");
  }
  lines.push("");
  lines.push(formatCheck(report.platformCheck));
  lines.push("");
  lines.push(report.ok ? "All checks passed." : "Some checks failed.");

  return lines.join("\n");
}

// ─── doctor ─────────────────────────────────────────────────

export async function runDoctor(version: string): Promise<void> {
  const runtime = typeof Bun !== "undefined" ? `Bun ${Bun.version}` : `Node ${process.version}`;
  const runtimeMajor = parseInt(process.version.replace("v", ""), 10);
  const runtimeError =
    runtimeMajor < 20 && typeof Bun === "undefined"
      ? "Node.js >= 20 required"
      : undefined;
  let gitError: string | undefined;
  try {
    execSync("git --version", { encoding: "utf-8", timeout: 5000 });
  } catch {
    gitError = "git not found in PATH";
  }

  const projectRoot = findProjectRoot() ?? process.cwd();
  const configPaths = [
    join(projectRoot, ".devagent.toml"),
    join(projectRoot, "devagent.toml"),
    join(homedir(), ".config", "devagent", "config.toml"),
    join(homedir(), ".devagent.toml"),
  ];
  const foundConfig = configPaths.find((p) => existsSync(p));
  const fileConfig = loadDoctorConfigFile(foundConfig);
  const config = loadConfig(projectRoot);
  const storedCredentials = new CredentialStore().all();
  const providerStatuses = buildProviderStatuses(config, storedCredentials);
  const lspStatuses = buildLspStatuses();

  let modelRegistryError: string | undefined;
  let modelRegistryCount = 0;
  let modelRegistered = false;
  try {
    loadModelRegistry();
    const models = getRegisteredModels();
    modelRegistryCount = models.length;
    modelRegistered = models.includes(config.model);
  } catch (err) {
    modelRegistryError = String(err);
  }

  const providerModelIssue =
    modelRegistryError || !modelRegistered
      ? undefined
      : getProviderModelCompatibilityIssue(config.provider, config.model);
  const modelOwner =
    modelRegistryError || !modelRegistered
      ? undefined
      : lookupModelEntry(config.model)?.provider;
  const activeProviderHasCredential = hasProviderCredential(
    config.provider,
    config,
    storedCredentials,
  );
  const providerCredentialIssue = buildProviderCredentialIssue(
    config.provider,
    activeProviderHasCredential,
    Boolean(providerModelIssue),
  );

  const report = buildDoctorReport({
    version,
    runtimeLabel: runtime,
    ...(runtimeError ? { runtimeError } : {}),
    ...(gitError ? { gitError } : {}),
    ...(foundConfig ? { configPath: foundConfig } : {}),
    configSearchPaths: configPaths,
    config,
    providerStatuses,
    ...(providerCredentialIssue ? { providerCredentialIssue } : {}),
    ...(modelRegistryError ? { modelRegistryError } : {}),
    modelRegistryCount,
    modelRegistered,
    ...(modelOwner ? { modelOwner } : {}),
    ...(providerModelIssue ? { providerModelIssue } : {}),
    lspStatuses,
    platformLabel: `${process.platform} ${process.arch}`,
    providerSource: resolveConfigValueSource("DEVAGENT_PROVIDER", fileConfig.topLevelProvider),
    modelSource: resolveConfigValueSource("DEVAGENT_MODEL", fileConfig.topLevelModel),
    credentialSource: resolveCredentialSource(config.provider, config, storedCredentials, fileConfig),
  });

  process.stdout.write(renderDoctorReport(report) + "\n");
  process.exit(report.ok ? 0 : 1);
}

const PROVIDERS = [
  { id: "anthropic", env: "ANTHROPIC_API_KEY", hint: "set ANTHROPIC_API_KEY or devagent auth login", credentialMode: "api" },
  { id: "openai", env: "OPENAI_API_KEY", hint: "set OPENAI_API_KEY or devagent auth login", credentialMode: "api" },
  { id: "devagent-api", env: "DEVAGENT_API_KEY", hint: "set DEVAGENT_API_KEY or devagent auth login", credentialMode: "api" },
  { id: "deepseek", env: "DEEPSEEK_API_KEY", hint: "set DEEPSEEK_API_KEY or devagent auth login", credentialMode: "api" },
  { id: "openrouter", env: "OPENROUTER_API_KEY", hint: "set OPENROUTER_API_KEY or devagent auth login", credentialMode: "api" },
  { id: "chatgpt", env: "", hint: "devagent auth login (ChatGPT Plus/Pro)", credentialMode: "oauth" },
  { id: "github-copilot", env: "", hint: "devagent auth login (GitHub device flow)", credentialMode: "oauth" },
  { id: "ollama", env: "", hint: "local — no API key needed (ollama must be running)", credentialMode: "none" },
] as const satisfies ReadonlyArray<ProviderDescriptor>;

const LSP_SERVERS = [
  { command: "typescript-language-server", label: "TypeScript/JavaScript", install: "npm i -g typescript-language-server typescript", npmPackages: ["typescript-language-server", "typescript"] },
  { command: "pyright-langserver", label: "Python (Pyright)", install: "npm i -g pyright", npmPackages: ["pyright"] },
  { command: "clangd", label: "C/C++ (clangd)", install: "apt install clangd / brew install llvm", npmPackages: null },
  { command: "rust-analyzer", label: "Rust", install: "rustup component add rust-analyzer", npmPackages: null },
  { command: "bash-language-server", label: "Bash/Shell", install: "npm i -g bash-language-server", npmPackages: ["bash-language-server"] },
];

function getProviderEnvKey(provider: string): string | null {
  const p = PROVIDERS.find((x) => x.id === provider);
  return p?.env || null;
}

function commandExists(cmd: string): boolean {
  try {
    execSync(`which ${cmd}`, { encoding: "utf-8", timeout: 3000, stdio: "pipe" });
    return true;
  } catch {
    return false;
  }
}

// ─── install-lsp ────────────────────────────────────────────

export function runInstallLsp(): void {
  const isBun = typeof globalThis.Bun !== "undefined";
  const pm = isBun ? "bun" : "npm";

  const toInstall: typeof LSP_SERVERS = [];
  const skipped: typeof LSP_SERVERS = [];
  const alreadyInstalled: typeof LSP_SERVERS = [];

  for (const lsp of LSP_SERVERS) {
    if (commandExists(lsp.command)) {
      alreadyInstalled.push(lsp);
    } else if (lsp.npmPackages) {
      toInstall.push(lsp);
    } else {
      skipped.push(lsp);
    }
  }

  if (alreadyInstalled.length > 0) {
    console.log("Already installed:");
    for (const lsp of alreadyInstalled) {
      console.log(`  ✓ ${lsp.label}`);
    }
    console.log("");
  }

  if (toInstall.length === 0) {
    console.log("All npm-installable LSP servers are already available.");
    if (skipped.length > 0) {
      console.log("\nSystem-level servers (install manually):");
      for (const lsp of skipped) {
        console.log(`  · ${lsp.label} — ${lsp.install}`);
      }
    }
    return;
  }

  // Collect all packages
  const packages = toInstall.flatMap((lsp) => lsp.npmPackages!);
  const cmd = `${pm} install -g ${packages.join(" ")}`;

  console.log(`Installing: ${toInstall.map((l) => l.label).join(", ")}`);
  console.log(`$ ${cmd}\n`);

  try {
    execSync(cmd, { stdio: "inherit", timeout: 120000 });
    console.log("\n✓ LSP servers installed");
  } catch {
    console.error("\n✗ Installation failed. Try running manually:");
    console.error(`  ${cmd}`);
    process.exit(1);
  }

  if (skipped.length > 0) {
    console.log("\nSystem-level servers (install manually):");
    for (const lsp of skipped) {
      console.log(`  · ${lsp.label} — ${lsp.install}`);
    }
  }
}

// ─── config ─────────────────────────────────────────────────

const GLOBAL_CONFIG_DIR = join(homedir(), ".config", "devagent");
const GLOBAL_CONFIG_PATH = join(GLOBAL_CONFIG_DIR, "config.json");

export function runConfig(args: string[]): void {
  const sub = args[0];

  if (!sub || sub === "path") {
    console.log(GLOBAL_CONFIG_PATH);
    return;
  }

  if (sub === "get") {
    const key = args[1];
    const data = loadConfigJson();
    if (!key) {
      // Dump all
      if (Object.keys(data).length === 0) {
        console.log("(no config set)");
      } else {
        for (const [k, v] of flatEntries(data)) {
          console.log(`${k} = ${formatValue(v)}`);
        }
      }
      return;
    }
    const value = getNestedValue(data, key);
    if (value === undefined) {
      console.log(`(not set)`);
    } else {
      console.log(formatValue(value));
    }
    return;
  }

  if (sub === "set") {
    const key = args[1];
    const value = args[2];
    if (!key || value === undefined) {
      console.error("Usage: devagent config set <key> <value>");
      process.exit(2);
    }
    const data = loadConfigJson();
    setNestedValue(data, key, parseValue(value));
    saveConfigJson(data);
    console.log(`${key} = ${value}`);
    return;
  }

  console.error(`Unknown config subcommand: ${sub}`);
  console.error("Usage: devagent config {get|set|path}");
  process.exit(2);
}

function loadConfigJson(): Record<string, unknown> {
  if (!existsSync(GLOBAL_CONFIG_PATH)) return {};
  try {
    return JSON.parse(readFileSync(GLOBAL_CONFIG_PATH, "utf-8")) as Record<string, unknown>;
  } catch {
    return {};
  }
}

function saveConfigJson(data: Record<string, unknown>): void {
  mkdirSync(GLOBAL_CONFIG_DIR, { recursive: true });
  writeFileSync(GLOBAL_CONFIG_PATH, JSON.stringify(data, null, 2) + "\n");
}

function getNestedValue(obj: Record<string, unknown>, path: string): unknown {
  const parts = path.split(".");
  let current: unknown = obj;
  for (const part of parts) {
    if (current == null || typeof current !== "object") return undefined;
    current = (current as Record<string, unknown>)[part];
  }
  return current;
}

function setNestedValue(obj: Record<string, unknown>, path: string, value: unknown): void {
  const parts = path.split(".");
  let current = obj;
  for (let i = 0; i < parts.length - 1; i++) {
    const part = parts[i]!;
    if (!(part in current) || typeof current[part] !== "object" || current[part] === null) {
      current[part] = {};
    }
    current = current[part] as Record<string, unknown>;
  }
  current[parts[parts.length - 1]!] = value;
}

function parseValue(s: string): unknown {
  if (s === "true") return true;
  if (s === "false") return false;
  const n = Number(s);
  if (!isNaN(n) && s.trim() !== "") return n;
  return s;
}

function formatValue(v: unknown): string {
  if (typeof v === "string") return v;
  if (typeof v === "object" && v !== null) return JSON.stringify(v);
  return String(v);
}

function flatEntries(obj: Record<string, unknown>, prefix = ""): Array<[string, unknown]> {
  const entries: Array<[string, unknown]> = [];
  for (const [k, v] of Object.entries(obj)) {
    const key = prefix ? `${prefix}.${k}` : k;
    if (typeof v === "object" && v !== null && !Array.isArray(v)) {
      entries.push(...flatEntries(v as Record<string, unknown>, key));
    } else {
      entries.push([key, v]);
    }
  }
  return entries;
}

// ─── init ───────────────────────────────────────────────────

export function runInit(): void {
  const cwd = process.cwd();
  const devagentDir = join(cwd, ".devagent");

  if (existsSync(devagentDir)) {
    console.log(".devagent/ already exists.");
  } else {
    mkdirSync(devagentDir, { recursive: true });
    console.log("Created .devagent/");
  }

  // Instructions file
  const instructionsPath = join(devagentDir, "instructions.md");
  if (!existsSync(instructionsPath)) {
    writeFileSync(instructionsPath, INSTRUCTIONS_TEMPLATE);
    console.log("Created .devagent/instructions.md");
  } else {
    console.log(".devagent/instructions.md already exists.");
  }

  // AGENTS.md
  const agentsPath = join(cwd, "AGENTS.md");
  if (!existsSync(agentsPath)) {
    const projectType = detectProjectType(cwd);
    writeFileSync(agentsPath, generateAgentsMd(projectType));
    console.log(`Created AGENTS.md (detected: ${projectType})`);
  } else {
    console.log("AGENTS.md already exists.");
  }

  console.log("\nDone. Edit these files to customize agent behavior.");
}

function detectProjectType(dir: string): string {
  if (existsSync(join(dir, "package.json"))) return "node";
  if (existsSync(join(dir, "Cargo.toml"))) return "rust";
  if (existsSync(join(dir, "go.mod"))) return "go";
  if (existsSync(join(dir, "pyproject.toml")) || existsSync(join(dir, "setup.py"))) return "python";
  if (existsSync(join(dir, "pom.xml")) || existsSync(join(dir, "build.gradle"))) return "java";
  if (existsSync(join(dir, "*.sln"))) return "dotnet";
  return "generic";
}

function generateAgentsMd(projectType: string): string {
  const buildCmds: Record<string, string> = {
    node: "npm install\nnpm run build\nnpm test",
    rust: "cargo build\ncargo test",
    go: "go build ./...\ngo test ./...",
    python: "pip install -e .\npytest",
    java: "mvn compile\nmvn test",
    dotnet: "dotnet build\ndotnet test",
    generic: "# Add your build/test commands here",
  };

  return `# Project Agent Instructions

## Build and Test

\`\`\`bash
${buildCmds[projectType] ?? buildCmds.generic}
\`\`\`

## Conventions

- Follow existing code style and patterns
- Keep changes minimal and focused
- Write tests for new functionality
- Run tests before considering work complete

## Architecture

<!-- Describe your project's architecture, key directories, and design decisions -->
`;
}

// ─── setup ──────────────────────────────────────────────────

const SETUP_PROVIDERS = [
  { id: "anthropic", name: "Anthropic", envVar: "ANTHROPIC_API_KEY", defaultModel: "claude-sonnet-4-20250514", hint: "Get key at https://console.anthropic.com/settings/keys" },
  { id: "openai", name: "OpenAI", envVar: "OPENAI_API_KEY", defaultModel: "gpt-4.1", hint: "Get key at https://platform.openai.com/api-keys" },
  { id: "devagent-api", name: "Devagent API", envVar: "DEVAGENT_API_KEY", defaultModel: "cortex", hint: "Use a gateway virtual key starting with ilg_" },
  { id: "deepseek", name: "DeepSeek", envVar: "DEEPSEEK_API_KEY", defaultModel: "deepseek-chat", hint: "Get key at https://platform.deepseek.com/api_keys" },
  { id: "openrouter", name: "OpenRouter", envVar: "OPENROUTER_API_KEY", defaultModel: "anthropic/claude-sonnet-4-20250514", hint: "Get key at https://openrouter.ai/keys" },
  { id: "ollama", name: "Ollama (local)", envVar: "", defaultModel: "qwen3:32b", hint: "No API key needed — ollama must be running locally" },
  { id: "chatgpt", name: "ChatGPT (Pro/Plus)", envVar: "", defaultModel: "gpt-4.1", hint: "Use 'devagent auth login' after setup" },
  { id: "github-copilot", name: "GitHub Copilot", envVar: "", defaultModel: "gpt-4.1", hint: "Use 'devagent auth login' after setup" },
];

export async function runSetup(): Promise<void> {
  const rl = createInterface({ input: process.stdin, output: process.stderr });
  const ask = (prompt: string): Promise<string> =>
    new Promise((resolve) => rl.question(prompt, resolve));

  const configDir = join(homedir(), ".config", "devagent");
  const configPath = join(configDir, "config.toml");
  const isUpdate = existsSync(configPath);

  console.log("DevAgent Setup\n");
  if (isUpdate) console.log(`(Existing config at ${configPath} will be updated)\n`);

  // 1. Provider selection
  console.log("Select your LLM provider:\n");
  for (let i = 0; i < SETUP_PROVIDERS.length; i++) {
    const p = SETUP_PROVIDERS[i]!;
    console.log(`  ${i + 1}. ${p.name}`);
    console.log(`     ${p.hint}`);
  }
  console.log("");

  const providerChoice = await ask(`> Provider (1-${SETUP_PROVIDERS.length}) [1]: `);
  const providerIdx = (parseInt(providerChoice.trim(), 10) || 1) - 1;
  const provider = SETUP_PROVIDERS[Math.max(0, Math.min(providerIdx, SETUP_PROVIDERS.length - 1))]!;
  console.log(`\n  ✓ Provider: ${provider.name}\n`);

  // 2. API key (for key-based providers)
  let apiKey = "";
  if (provider.envVar) {
    const existing = process.env[provider.envVar];
    if (existing) {
      console.log(`  ${provider.envVar} already set in environment.`);
      console.log(`  (Will use environment variable at runtime)\n`);
    } else {
      apiKey = await ask(`> ${provider.envVar}: `);
      apiKey = apiKey.trim();
      if (apiKey) {
        console.log(`  ✓ API key stored\n`);
      } else {
        console.log(`  (Skipped — set ${provider.envVar} in your shell profile later)\n`);
      }
    }
  } else if (provider.id === "ollama") {
    console.log("  No API key needed for Ollama.\n");
  } else {
    console.log(`  Run 'devagent auth login' after setup to authenticate.\n`);
  }

  // 3. Model selection
  const modelPrompt = `> Model [${provider.defaultModel}]: `;
  const modelChoice = await ask(modelPrompt);
  const model = modelChoice.trim() || provider.defaultModel;
  console.log(`  ✓ Model: ${model}\n`);

  // 4. Approval mode
  console.log("Approval mode:");
  console.log("  1. suggest — ask before writing files (recommended)");
  console.log("  2. auto-edit — auto-approve file writes, ask for commands");
  console.log("  3. full-auto — auto-approve everything");
  console.log("");
  const approvalChoice = await ask("> Approval mode (1-3) [1]: ");
  const approvalModes = ["suggest", "auto-edit", "full-auto"];
  const approvalIdx = (parseInt(approvalChoice.trim(), 10) || 1) - 1;
  const approvalMode = approvalModes[Math.max(0, Math.min(approvalIdx, 2))]!;
  console.log(`  ✓ Approval mode: ${approvalMode}\n`);

  // 5. Max iterations
  const iterChoice = await ask("> Max iterations per query [30]: ");
  const maxIterations = parseInt(iterChoice.trim(), 10) || 30;
  console.log(`  ✓ Max iterations: ${maxIterations}\n`);

  // 6. Subagent configuration
  const AGENT_TYPES = [
    { id: "general", label: "General", desc: "default agent for code tasks" },
    { id: "explore", label: "Explore", desc: "fast codebase search, read-only" },
    { id: "reviewer", label: "Reviewer", desc: "code review, read-only" },
    { id: "architect", label: "Architect", desc: "design and planning, read-only" },
  ];

  // Sensible defaults per provider
  const subagentDefaults: Record<string, { models: Record<string, string>; reasoning: Record<string, string> }> = {
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
  const defaults = subagentDefaults[provider.id] ?? {
    models: { general: model, explore: model, reviewer: model, architect: model },
    reasoning: { general: "medium", explore: "low", reviewer: "high", architect: "high" },
  };

  console.log("Subagent configuration:");
  console.log("  DevAgent spawns specialized subagents for different tasks.");
  console.log("  You can use cheaper/faster models for simple tasks like exploration.\n");

  console.log("  Defaults for your provider:");
  for (const agent of AGENT_TYPES) {
    const m = defaults.models[agent.id] ?? model;
    const r = defaults.reasoning[agent.id] ?? "medium";
    console.log(`    ${agent.label.padEnd(10)} model=${m}  reasoning=${r}`);
  }
  console.log("");

  const customizeSub = await ask("> Customize subagent models? (y/N) [N]: ");
  const agentModels: Record<string, string> = { ...defaults.models };
  const agentReasoning: Record<string, string> = { ...defaults.reasoning };

  if (customizeSub.trim().toLowerCase() === "y") {
    console.log("");
    for (const agent of AGENT_TYPES) {
      const defModel = defaults.models[agent.id] ?? model;
      const defReasoning = defaults.reasoning[agent.id] ?? "medium";

      const mChoice = await ask(`  > ${agent.label} model [${defModel}]: `);
      agentModels[agent.id] = mChoice.trim() || defModel;

      const rChoice = await ask(`  > ${agent.label} reasoning (low/medium/high) [${defReasoning}]: `);
      const r = rChoice.trim().toLowerCase();
      agentReasoning[agent.id] = (r === "low" || r === "medium" || r === "high") ? r : defReasoning;

      console.log(`    ✓ ${agent.label}: ${agentModels[agent.id]} (${agentReasoning[agent.id]})\n`);
    }
  } else {
    console.log("  ✓ Using defaults\n");
  }

  rl.close();

  // Write config.toml
  const lines: string[] = [
    "# DevAgent global configuration",
    `# Generated by 'devagent setup' on ${new Date().toISOString().split("T")[0]}`,
    "",
    `provider = "${provider.id}"`,
    `model = "${model}"`,
    "",
    "[approval]",
    `mode = "${approvalMode}"`,
    "",
    "[budget]",
    `max_iterations = ${maxIterations}`,
  ];

  // Subagent config
  const hasCustomModels = Object.entries(agentModels).some(([k, v]) => v !== model);
  const hasCustomReasoning = true; // Always write reasoning defaults
  if (hasCustomModels || hasCustomReasoning) {
    lines.push("", "[subagents]", "# Per-agent model and reasoning overrides");
    lines.push("# Agent types: general, explore, reviewer, architect");

    if (hasCustomModels) {
      lines.push("", "[subagents.agent_model_overrides]");
      for (const agent of AGENT_TYPES) {
        const m = agentModels[agent.id];
        if (m && m !== model) {
          lines.push(`${agent.id} = "${m}"`);
        }
      }
    }

    lines.push("", "[subagents.agent_reasoning_overrides]");
    for (const agent of AGENT_TYPES) {
      lines.push(`${agent.id} = "${agentReasoning[agent.id] ?? "medium"}"`);
    }
  }

  // Provider-specific config
  if (apiKey) {
    lines.push("", `[providers.${provider.id}]`, `api_key = "${apiKey}"`);
  }
  if (provider.id === "ollama") {
    lines.push("", "[providers.ollama]", 'base_url = "http://localhost:11434/v1"');
  }

  mkdirSync(configDir, { recursive: true });
  writeFileSync(configPath, lines.join("\n") + "\n");

  console.log(`Config written to ${configPath}\n`);
  console.log("Next steps:");
  if (!apiKey && provider.envVar) {
    console.log(`  1. Set ${provider.envVar} in your shell profile`);
    console.log(`  2. Run 'devagent doctor' to verify`);
    console.log(`  3. Run 'devagent "hello"' to test`);
  } else if (provider.id === "chatgpt" || provider.id === "github-copilot") {
    console.log(`  1. Run 'devagent auth login' to authenticate`);
    console.log(`  2. Run 'devagent doctor' to verify`);
    console.log(`  3. Run 'devagent "hello"' to test`);
  } else {
    console.log(`  1. Run 'devagent doctor' to verify`);
    console.log(`  2. Run 'devagent "hello"' to test`);
  }
  console.log(`  - Run 'devagent init' in a project to add project-level config`);
}

// ─── update ─────────────────────────────────────────────────

export async function runUpdate(): Promise<void> {
  const PACKAGE = "@egavrin/devagent";

  console.log("Checking for updates...");

  try {
    const res = await fetch(`https://registry.npmjs.org/${PACKAGE}/latest`, {
      signal: AbortSignal.timeout(5000),
    });
    const data = (await res.json()) as { version?: string };
    const latest = data.version;

    if (!latest) {
      console.error("Could not determine latest version.");
      process.exit(1);
    }

    const current = getCurrentVersion();
    if (latest === current) {
      console.log(`Already up to date (v${current}).`);
      return;
    }

    console.log(`Updating: v${current} → v${latest}\n`);

    // Detect package manager
    const isBun = typeof globalThis.Bun !== "undefined";
    const cmd = isBun
      ? `bun install -g ${PACKAGE}@latest`
      : `npm install -g ${PACKAGE}@latest`;

    console.log(`$ ${cmd}\n`);
    execSync(cmd, { stdio: "inherit" });
    console.log(`\n✓ Updated to v${latest}`);
  } catch (err) {
    console.error(`Update failed: ${err instanceof Error ? err.message : String(err)}`);
    process.exit(1);
  }
}

function getCurrentVersion(): string {
  try {
    const dir = new URL(".", import.meta.url).pathname;
    const pkgPath = join(dir, "package.json");
    if (existsSync(pkgPath)) {
      return JSON.parse(readFileSync(pkgPath, "utf-8")).version ?? "0.0.0";
    }
  } catch { /* ignore */ }
  return "0.0.0";
}

// ─── completions ────────────────────────────────────────────

const COMMANDS = [
  "setup", "init", "doctor", "config", "update", "completions",
  "version", "sessions", "review", "auth", "execute",
];
const FLAGS = [
  "--help", "--version", "--provider", "--model", "--max-iterations",
  "--reasoning", "--resume", "--continue", "--suggest", "--auto-edit",
  "--full-auto", "--verbose", "--quiet", "--file",
];

export function runCompletions(shell: string): void {
  switch (shell) {
    case "bash":
      console.log(bashCompletions());
      console.log("\n# Add to ~/.bashrc:\n#   eval \"$(devagent completions bash)\"");
      break;
    case "zsh":
      console.log(zshCompletions());
      console.log("\n# Add to ~/.zshrc:\n#   eval \"$(devagent completions zsh)\"");
      break;
    case "fish":
      console.log(fishCompletions());
      console.log("\n# Save to ~/.config/fish/completions/devagent.fish:\n#   devagent completions fish > ~/.config/fish/completions/devagent.fish");
      break;
    default:
      console.log("Usage: devagent completions <bash|zsh|fish>");
      console.log("\nExamples:");
      console.log("  eval \"$(devagent completions bash)\"   # Add to ~/.bashrc");
      console.log("  eval \"$(devagent completions zsh)\"    # Add to ~/.zshrc");
      console.log("  devagent completions fish > ~/.config/fish/completions/devagent.fish");
      break;
  }
}

function bashCompletions(): string {
  return `_devagent_completions() {
  local cur="\${COMP_WORDS[COMP_CWORD]}"
  local prev="\${COMP_WORDS[COMP_CWORD-1]}"

  case "\${prev}" in
    devagent)
      COMPREPLY=( $(compgen -W "${COMMANDS.join(" ")} ${FLAGS.join(" ")}" -- "\${cur}") )
      return 0
      ;;
    config)
      COMPREPLY=( $(compgen -W "get set path" -- "\${cur}") )
      return 0
      ;;
    auth)
      COMPREPLY=( $(compgen -W "login status logout" -- "\${cur}") )
      return 0
      ;;
    completions)
      COMPREPLY=( $(compgen -W "bash zsh fish" -- "\${cur}") )
      return 0
      ;;
    --provider)
      COMPREPLY=( $(compgen -W "anthropic openai devagent-api deepseek openrouter ollama chatgpt github-copilot" -- "\${cur}") )
      return 0
      ;;
    --reasoning)
      COMPREPLY=( $(compgen -W "low medium high" -- "\${cur}") )
      return 0
      ;;
  esac

  if [[ "\${cur}" == -* ]]; then
    COMPREPLY=( $(compgen -W "${FLAGS.join(" ")}" -- "\${cur}") )
  else
    COMPREPLY=( $(compgen -W "${COMMANDS.join(" ")}" -- "\${cur}") )
  fi
}
complete -F _devagent_completions devagent`;
}

function zshCompletions(): string {
  return `#compdef devagent

_devagent() {
  local -a commands flags

  commands=(
${COMMANDS.map((c) => `    '${c}:${c} command'`).join("\n")}
  )

  flags=(
    '--help[Show help]'
    '--version[Show version]'
    '--provider[LLM provider]:provider:(anthropic openai devagent-api deepseek openrouter ollama chatgpt github-copilot)'
    '--model[Model ID]:model:'
    '--max-iterations[Max iterations]:number:'
    '--reasoning[Reasoning effort]:level:(low medium high)'
    '--resume[Resume session]:session_id:'
    '--continue[Resume most recent session]'
    '--suggest[Suggest mode]'
    '--auto-edit[Auto-edit mode]'
    '--full-auto[Full-auto mode]'
    '--verbose[Verbose output]'
    '--quiet[Quiet output]'
    '--file[Read query from file]:file:_files'
  )

  _arguments -s \\
    '1:command:->command' \\
    '*::arg:->args' \\
    \${flags}

  case \$state in
    command)
      _describe 'command' commands
      ;;
    args)
      case \$words[1] in
        config) _values 'subcommand' get set path ;;
        auth) _values 'subcommand' login status logout ;;
        completions) _values 'shell' bash zsh fish ;;
      esac
      ;;
  esac
}

_devagent`;
}

function fishCompletions(): string {
  const lines = [
    "# devagent completions for fish",
    "complete -c devagent -e",
    "",
    "# Commands",
    ...COMMANDS.map((c) => `complete -c devagent -n '__fish_use_subcommand' -a '${c}' -d '${c}'`),
    "",
    "# Flags",
    "complete -c devagent -l help -s h -d 'Show help'",
    "complete -c devagent -l version -s V -d 'Show version'",
    "complete -c devagent -l provider -x -a 'anthropic openai devagent-api deepseek openrouter ollama chatgpt github-copilot' -d 'LLM provider'",
    "complete -c devagent -l model -x -d 'Model ID'",
    "complete -c devagent -l max-iterations -x -d 'Max iterations'",
    "complete -c devagent -l reasoning -x -a 'low medium high' -d 'Reasoning effort'",
    "complete -c devagent -l resume -x -d 'Resume session by ID'",
    "complete -c devagent -l continue -d 'Resume most recent session'",
    "complete -c devagent -l suggest -d 'Suggest mode'",
    "complete -c devagent -l auto-edit -d 'Auto-edit mode'",
    "complete -c devagent -l full-auto -d 'Full-auto mode'",
    "complete -c devagent -l verbose -s v -d 'Verbose output'",
    "complete -c devagent -l quiet -s q -d 'Quiet output'",
    "complete -c devagent -l file -s f -r -d 'Read query from file'",
    "",
    "# Subcommands",
    "complete -c devagent -n '__fish_seen_subcommand_from config' -a 'get set path'",
    "complete -c devagent -n '__fish_seen_subcommand_from auth' -a 'login status logout'",
    "complete -c devagent -n '__fish_seen_subcommand_from completions' -a 'bash zsh fish'",
  ];
  return lines.join("\n");
}

const INSTRUCTIONS_TEMPLATE = `# DevAgent Instructions

<!--
  This file provides project-specific instructions to devagent.
  It is loaded automatically when devagent runs in this directory.
-->

## Guidelines

- Follow the project's existing coding conventions
- Prefer editing existing files over creating new ones
- Run tests after making changes
`;
