/**
 * CLI subcommands: doctor, config, setup.
 */

import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";
import { execSync } from "node:child_process";
import { createInterface } from "node:readline";
import {
  CredentialStore,
  formatResolvedCredentialSource,
  findProjectRoot,
  getProvidersForModel,
  getProviderCredentialDescriptor,
  loadConfig,
  loadModelRegistry,
  getRegisteredModels,
  isModelRegisteredForProvider,
  listProviderCredentialDescriptors,
  resolveProviderCredentialStatus,
} from "@devagent/runtime";
import type { CredentialInfo, DevAgentConfig, ProviderCredentialDescriptor as ProviderDescriptor } from "@devagent/runtime";
import type { ProviderModelCompatibilityIssue } from "./provider-model-compat.js";
import {
  formatProviderModelCompatibilityError,
  formatProviderModelCompatibilityHint,
  getProviderModelCompatibilityIssue,
} from "./provider-model-compat.js";
import {
  getGlobalConfigPath,
  getGlobalConfigValue,
  listGlobalConfigEntries,
  loadGlobalConfigObject,
  migrateLegacyGlobalConfigIfNeeded,
  migrateLegacyGlobalTomlIfNeeded,
  normalizeGlobalConfigIfNeeded,
  setGlobalConfigValue,
  writeGlobalConfigObject,
} from "./global-config.js";

type DoctorCheckStatus = "pass" | "blocking" | "advisory";

function hasHelpFlag(args: ReadonlyArray<string>): boolean {
  return args.includes("--help") || args.includes("-h");
}

function renderDoctorHelpText(): string {
  return `Usage:
  devagent doctor

Check the current environment, global config, provider credentials, model registry,
and LSP availability.`;
}

function renderConfigHelpText(): string {
  return `Usage:
  devagent config path
  devagent config get [key]
  devagent config set <key> <value>

Inspect or edit the global DevAgent config directly at ~/.config/devagent/config.toml.`;
}

function renderConfigureHelpText(): string {
  return `Usage:
  devagent configure

Alias for "devagent setup".`;
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

function renderInstallLspHelpText(): string {
  return `Usage:
  devagent install-lsp

Install npm-managed LSP servers that power DevAgent code intelligence.`;
}

function renderUpdateHelpText(): string {
  return `Usage:
  devagent update

Check npm for the latest published version and upgrade the installed CLI.`;
}

function renderCompletionsHelpText(): string {
  return `Usage:
  devagent completions <bash|zsh|fish>

Generate shell completions for the public CLI surface.`;
}

interface DoctorCheck {
  readonly label: string;
  readonly status: DoctorCheckStatus;
  readonly detail?: string;
}

interface DoctorIssue {
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
  readonly modelProviders?: ReadonlyArray<string>;
  readonly providerModelIssue?: ProviderModelCompatibilityIssue;
  readonly lspStatuses: ReadonlyArray<DoctorLspStatus>;
  readonly platformLabel: string;
  readonly providerSource: "cli" | "env" | "config" | "default";
  readonly modelSource: "cli" | "env" | "config" | "default";
  readonly credentialSource: string;
}

interface DoctorReport {
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
    readonly modelProviders?: ReadonlyArray<string>;
  };
  readonly lspStatuses: ReadonlyArray<DoctorLspStatus>;
  readonly platformCheck: DoctorCheck;
  readonly ok: boolean;
  readonly summaryStatus: "pass" | "advisory" | "blocking";
}

interface DoctorConfigFile {
  readonly topLevelProvider?: string;
  readonly topLevelApiKey?: string;
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
  return getProviderCredentialDescriptor(providerId);
}

function hasProviderCredential(
  providerId: string,
  config: DevAgentConfig,
  storedCredentials: Readonly<Record<string, CredentialInfo>>,
  topLevelApiKey?: string,
): boolean {
  return resolveProviderCredentialStatus({
    providerId,
    providerConfig: config.providers[providerId],
    topLevelApiKey,
    storedCredential: storedCredentials[providerId],
  }).hasCredential;
}

function buildProviderStatuses(
  config: DevAgentConfig,
  storedCredentials: Readonly<Record<string, CredentialInfo>>,
  topLevelApiKey?: string,
): DoctorProviderStatus[] {
  return listProviderCredentialDescriptors().map((provider) => ({
    id: provider.id,
    hint: provider.hint,
    active: provider.id === config.provider,
    hasCredential: hasProviderCredential(
      provider.id,
      config,
      storedCredentials,
      provider.id === config.provider ? topLevelApiKey : undefined,
    ),
  }));
}

function loadDoctorConfigFile(configPath: string | undefined): DoctorConfigFile {
  if (!configPath) {
    return {};
  }
  try {
    const text = readFileSync(configPath, "utf-8");
    const topLevelApiKey = matchTopLevelTomlString(text, "api_key");
    const topLevelProvider = matchTopLevelTomlString(text, "provider");
    const topLevelModel = matchTopLevelTomlString(text, "model");
    return {
      ...(topLevelApiKey ? { topLevelApiKey } : {}),
      ...(topLevelProvider ? { topLevelProvider } : {}),
      ...(topLevelModel ? { topLevelModel } : {}),
    };
  } catch {
    return {};
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

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function resolveCredentialSource(
  providerId: string,
  config: DevAgentConfig,
  storedCredentials: Readonly<Record<string, CredentialInfo>>,
  fileConfig: DoctorConfigFile,
): string {
  return formatResolvedCredentialSource(resolveProviderCredentialStatus({
    providerId,
    providerConfig: config.providers[providerId],
    topLevelApiKey: fileConfig.topLevelApiKey,
    storedCredential: storedCredentials[providerId],
  }));
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

  const descriptor = getProviderDescriptor(provider);
  if (!descriptor || descriptor.credentialMode === "none") {
    return undefined;
  }

  const detail = descriptor.credentialMode === "oauth"
    ? `no stored login (run devagent auth login)${hasProviderModelMismatch ? ". Secondary until provider/model pairing is fixed." : ""}`
    : `no API key (set ${descriptor.envVar ?? "DEVAGENT_API_KEY"} or run devagent auth login)${hasProviderModelMismatch ? ". Secondary until provider/model pairing is fixed." : ""}`;

  return {
    status: hasProviderModelMismatch ? "advisory" : "blocking",
    detail,
  };
}

function buildProviderModelIssueSteps(
  issue: ProviderModelCompatibilityIssue,
): string[] {
  if (issue.model === "cortex" || issue.supportedProviders.includes("devagent-api")) {
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
    `Run now: devagent --provider ${issue.supportedProviders[0]} --model ${issue.model} "<your prompt>"`,
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
    `Export credentials: export ${descriptor.envVar}=${placeholder}`,
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
      nextSteps: [
        "Install Node.js >= 20 (recommended on Ubuntu: nvm install 20 && nvm use 20).",
        "Or use Bun >= 1.3 if you prefer a Bun-first setup.",
        "Then retry: devagent doctor",
      ],
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
  const hasAdvisories =
    providerCheck.status === "advisory" ||
    modelCheck.status === "advisory" ||
    providerModelCheck.status === "advisory" ||
    !foundLsp;
  const ok = blockingIssues.length === 0;
  const summaryStatus = ok ? (hasAdvisories ? "advisory" : "pass") : "blocking";

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
      ...(input.modelProviders && input.modelProviders.length > 0 ? { modelProviders: input.modelProviders } : {}),
    },
    lspStatuses: input.lspStatuses,
    platformCheck,
    ok,
    summaryStatus,
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
  if (report.effectiveConfig.modelProviders && report.effectiveConfig.modelProviders.length > 0) {
    lines.push(`  Registered providers: ${report.effectiveConfig.modelProviders.join(", ")}`);
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
  lines.push(
    report.summaryStatus === "blocking"
      ? "Blocking issues found."
      : report.summaryStatus === "advisory"
        ? "Checks passed with advisories."
        : "All checks passed.",
  );

  return lines.join("\n");
}

// ─── doctor ─────────────────────────────────────────────────

export async function runDoctor(version: string, args: ReadonlyArray<string> = []): Promise<void> {
  if (hasHelpFlag(args)) {
    console.log(renderDoctorHelpText());
    return;
  }

  migrateLegacyGlobalConfigIfNeeded();
  migrateLegacyGlobalTomlIfNeeded();
  normalizeGlobalConfigIfNeeded();
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
  const globalHome = process.env["HOME"] ?? homedir();
  const configPaths = [getGlobalConfigPath(globalHome)];
  const foundConfig = configPaths.find((p) => existsSync(p));
  const fileConfig = loadDoctorConfigFile(foundConfig);
  const config = loadConfig(projectRoot);
  const storedCredentials = new CredentialStore().all();
  const providerStatuses = buildProviderStatuses(config, storedCredentials, fileConfig.topLevelApiKey);
  const lspStatuses = buildLspStatuses();

  let modelRegistryError: string | undefined;
  let modelRegistryCount = 0;
  let modelRegistered = false;
  let modelProviders: ReadonlyArray<string> | undefined;
  try {
    loadModelRegistry();
    const models = getRegisteredModels();
    modelRegistryCount = models.length;
    modelRegistered = models.includes(config.model);
    modelProviders = modelRegistered ? getProvidersForModel(config.model) : undefined;
  } catch (err) {
    modelRegistryError = String(err);
  }

  const providerModelIssue =
    modelRegistryError || !modelRegistered
      ? undefined
      : getProviderModelCompatibilityIssue(config.provider, config.model);
  const activeProviderHasCredential = hasProviderCredential(
    config.provider,
    config,
    storedCredentials,
    fileConfig.topLevelApiKey,
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
    ...(modelProviders ? { modelProviders } : {}),
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

const LSP_SERVERS = [
  { command: "typescript-language-server", label: "TypeScript/JavaScript", install: "npm i -g typescript-language-server typescript", npmPackages: ["typescript-language-server", "typescript"] },
  { command: "pyright-langserver", label: "Python (Pyright)", install: "npm i -g pyright", npmPackages: ["pyright"] },
  { command: "clangd", label: "C/C++ (clangd)", install: "apt install clangd / brew install llvm", npmPackages: null },
  { command: "rust-analyzer", label: "Rust", install: "rustup component add rust-analyzer", npmPackages: null },
  { command: "bash-language-server", label: "Bash/Shell", install: "npm i -g bash-language-server", npmPackages: ["bash-language-server"] },
];

function commandExists(cmd: string): boolean {
  try {
    execSync(`which ${cmd}`, { encoding: "utf-8", timeout: 3000, stdio: "pipe" });
    return true;
  } catch {
    return false;
  }
}

// ─── install-lsp ────────────────────────────────────────────

export function runInstallLsp(args: ReadonlyArray<string> = []): void {
  if (hasHelpFlag(args)) {
    console.log(renderInstallLspHelpText());
    return;
  }

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

export function runConfig(args: string[]): void {
  if (hasHelpFlag(args)) {
    console.log(renderConfigHelpText());
    return;
  }

  migrateLegacyGlobalConfigIfNeeded();
  migrateLegacyGlobalTomlIfNeeded();
  normalizeGlobalConfigIfNeeded();
  const sub = args[0];

  if (!sub || sub === "path") {
    console.log(getGlobalConfigPath());
    return;
  }

  if (sub === "get") {
    const key = args[1];
    if (!key) {
      const entries = listGlobalConfigEntries();
      if (entries.length === 0) {
        console.log("(no config set)");
      } else {
        for (const [entryKey, entryValue] of entries) {
          console.log(`${entryKey} = ${entryValue}`);
        }
      }
      return;
    }
    const value = getGlobalConfigValue(key);
    if (value === undefined) {
      console.log(`(not set)`);
    } else {
      console.log(value);
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
    const updates = setValidatedGlobalConfigValue(key, value);
    for (const [updatedKey, updatedValue] of updates) {
      console.log(`${updatedKey} = ${updatedValue}`);
    }
    return;
  }

  console.error(`Unknown config subcommand: ${sub}`);
  console.error("Usage: devagent config {get|set|path}");
  process.exit(2);
}

// ─── init (retired) ────────────────────────────────────────

export function runInit(args: ReadonlyArray<string> = []): void {
  if (hasHelpFlag(args)) {
    console.log(renderInitHelpText());
    return;
  }

  console.error("devagent init has been removed from the public CLI.");
  console.error("DevAgent no longer scaffolds project instruction files automatically.");
  console.error("Create AGENTS.md manually when you want repository-specific guidance.");
  process.exit(2);
}

export async function runSetup(args: ReadonlyArray<string> = []): Promise<void> {
  if (hasHelpFlag(args)) {
    console.log(renderSetupHelpText());
    return;
  }

  await runConfigure(args);
}

// ─── configure ──────────────────────────────────────────────

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

  if (canonicalPath === "provider") {
    const provider = rawValue.trim();
    const currentModel = typeof config["model"] === "string" ? config["model"] : undefined;
    loadModelRegistry();
    const currentModelSupported = currentModel ? isModelRegisteredForProvider(provider, currentModel) : false;
    const defaultModel = getDefaultModelForProvider(provider);
    config["provider"] = provider;

    if (!currentModel || !currentModelSupported) {
      if (defaultModel) {
        config["model"] = defaultModel;
        writeGlobalConfigObject(config);
        return [["provider", provider], ["model", defaultModel]];
      }
    }

    writeGlobalConfigObject(config);
    return [["provider", provider]];
  }

  if (canonicalPath === "model") {
    const provider = typeof config["provider"] === "string" ? config["provider"] : undefined;
    if (provider) {
      loadModelRegistry();
      const providerModelIssue = getProviderModelCompatibilityIssue(provider, rawValue);
      if (providerModelIssue) {
        console.error(formatProviderModelCompatibilityError(providerModelIssue));
        const hint = formatProviderModelCompatibilityHint(providerModelIssue);
        if (hint) {
          console.error(hint);
        }
        process.exit(2);
      }
    }
    setGlobalConfigValue("model", rawValue);
    return [["model", rawValue]];
  }

  if (canonicalPath === "safety.mode" || canonicalPath === "approval.mode") {
    setGlobalConfigValue(path, rawValue);
    return [["safety.mode", getGlobalConfigValue("safety.mode") ?? rawValue]];
  }

  setGlobalConfigValue(path, rawValue);
  return [[path, rawValue]];
}

export async function runConfigure(args: ReadonlyArray<string> = []): Promise<void> {
  if (hasHelpFlag(args)) {
    console.log(renderSetupHelpText());
    return;
  }

  migrateLegacyGlobalConfigIfNeeded();
  migrateLegacyGlobalTomlIfNeeded();
  normalizeGlobalConfigIfNeeded();
  const rl = createInterface({ input: process.stdin, output: process.stderr });
  const ask = (prompt: string): Promise<string> =>
    new Promise((resolve) => rl.question(prompt, resolve));

  const configPath = getGlobalConfigPath();
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
    console.log("  Run 'devagent auth login' after configuration to authenticate.\n");
  }

  // 3. Model selection
  const modelPrompt = `> Model [${provider.defaultModel}]: `;
  const modelChoice = await ask(modelPrompt);
  const model = modelChoice.trim() || provider.defaultModel;
  console.log(`  ✓ Model: ${model}\n`);

  // 4. Safety mode
  console.log("Safety mode:");
  console.log("  1. autopilot — allow everything without prompts (recommended)");
  console.log("  2. default — auto-allow workspace edits and safe repo commands");
  console.log("");
  const approvalChoice = await ask("> Safety mode (1-2) [1]: ");
  const safetyModes = ["autopilot", "default"];
  const approvalIdx = (parseInt(approvalChoice.trim(), 10) || 1) - 1;
  const safetyMode = safetyModes[Math.max(0, Math.min(approvalIdx, 1))]!;
  console.log(`  ✓ Safety mode: ${safetyMode}\n`);

  // 5. Max iterations
  const iterChoice = await ask("> Max iterations per query [0]: ");
  const parsedMaxIterations = Number.parseInt(iterChoice.trim(), 10);
  const maxIterations = Number.isNaN(parsedMaxIterations) ? 0 : parsedMaxIterations;
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
  const nextConfig = loadGlobalConfigObject();
  nextConfig["provider"] = provider.id;
  nextConfig["model"] = model;
  nextConfig["safety"] = {
    ...((nextConfig["safety"] as Record<string, unknown> | undefined) ?? {}),
    mode: safetyMode,
  };
  nextConfig["budget"] = {
    ...((nextConfig["budget"] as Record<string, unknown> | undefined) ?? {}),
    max_iterations: maxIterations,
  };

  // Subagent config
  const hasCustomModels = Object.entries(agentModels).some(([k, v]) => v !== model);
  const hasCustomReasoning = true; // Always write reasoning defaults
  if (hasCustomModels || hasCustomReasoning) {
    const subagents = { ...((nextConfig["subagents"] as Record<string, unknown> | undefined) ?? {}) };

    if (hasCustomModels) {
      const modelOverrides: Record<string, string> = {};
      for (const agent of AGENT_TYPES) {
        const m = agentModels[agent.id];
        if (m && m !== model) {
          modelOverrides[agent.id] = m;
        }
      }
      subagents["agent_model_overrides"] = modelOverrides;
    }

    const reasoningOverrides: Record<string, string> = {};
    for (const agent of AGENT_TYPES) {
      reasoningOverrides[agent.id] = agentReasoning[agent.id] ?? "medium";
    }
    subagents["agent_reasoning_overrides"] = reasoningOverrides;
    nextConfig["subagents"] = subagents;
  }

  // Provider-specific config
  if (apiKey) {
    const providers = { ...((nextConfig["providers"] as Record<string, unknown> | undefined) ?? {}) };
    providers[provider.id] = {
      ...((providers[provider.id] as Record<string, unknown> | undefined) ?? {}),
      api_key: apiKey,
    };
    nextConfig["providers"] = providers;
  }
  if (provider.id === "ollama") {
    const providers = { ...((nextConfig["providers"] as Record<string, unknown> | undefined) ?? {}) };
    providers["ollama"] = {
      ...((providers["ollama"] as Record<string, unknown> | undefined) ?? {}),
      base_url: "http://localhost:11434/v1",
    };
    nextConfig["providers"] = providers;
  }

  writeGlobalConfigObject(nextConfig);

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
}

// ─── update ─────────────────────────────────────────────────

export async function runUpdate(args: ReadonlyArray<string> = []): Promise<void> {
  if (hasHelpFlag(args)) {
    console.log(renderUpdateHelpText());
    return;
  }

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
  "setup", "doctor", "config", "update", "completions",
  "version", "sessions", "review", "auth", "execute",
];
const FLAGS = [
  "--help", "--version", "--provider", "--model", "--max-iterations",
  "--reasoning", "--resume", "--continue", "--mode", "--verbose", "--quiet", "--file",
];

export function runCompletions(args: ReadonlyArray<string> = []): void {
  if (hasHelpFlag(args)) {
    console.log(renderCompletionsHelpText());
    return;
  }

  const shell = args[0] ?? "";
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
    '--mode[Interactive safety mode]:mode:(default autopilot)'
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
    "complete -c devagent -l mode -d 'Interactive safety mode' -a 'default autopilot'",
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
