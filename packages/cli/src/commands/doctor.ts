import {
  CredentialStore,
  findProjectRoot,
  formatResolvedCredentialSource,
  getProviderCredentialDescriptor,
  getProvidersForModel,
  getRegisteredModels,
  listProviderCredentialDescriptors,
  loadConfig,
  loadModelRegistry,
  resolveProviderCredentialStatus,
} from "@devagent/runtime";
import { execSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";

import { getGlobalConfigPath, migrateLegacyGlobalConfigIfNeeded, migrateLegacyGlobalTomlIfNeeded, normalizeGlobalConfigIfNeeded } from "../global-config.js";
import type { ProviderModelCompatibilityIssue } from "../provider-model-compat.js";
import { formatProviderModelCompatibilityError, formatProviderModelCompatibilityHint, getProviderModelCompatibilityIssue } from "../provider-model-compat.js";
import { renderDoctorReport } from "./doctor-render.js";
import type { DoctorCheck, DoctorCheckStatus, DoctorIssue, DoctorLspStatus, DoctorProviderCredentialIssue, DoctorProviderStatus, DoctorReport, DoctorReportInput, DoctorStoredCredentials } from "./doctor-types.js";
import { commandExists, LSP_SERVERS } from "./lsp-servers.js";
import { hasHelpFlag, writeStdout } from "./shared.js";
import type { CredentialInfo, DevAgentConfig, ProviderCredentialDescriptor as ProviderDescriptor } from "@devagent/runtime";

function renderDoctorHelpText(): string {
  return `Usage:
  devagent doctor

Check the current environment, global config, provider credentials, model registry,
and LSP availability.`;
}

interface DoctorConfigFile {
  readonly topLevelProvider?: string;
  readonly topLevelApiKey?: string;
  readonly topLevelModel?: string;
  readonly rawText?: string;
}

interface DoctorRuntimeStatus {
  readonly label: string;
  readonly error?: string;
}

interface DoctorConfigContext {
  readonly configPaths: ReadonlyArray<string>;
  readonly foundConfig?: string;
  readonly fileConfig: DoctorConfigFile;
  readonly config: DevAgentConfig;
  readonly storedCredentials: DoctorStoredCredentials;
}

interface DoctorModelRegistryStatus {
  readonly error?: string;
  readonly count: number;
  readonly registered: boolean;
  readonly providers?: ReadonlyArray<string>;
}

interface DoctorReportInputContext {
  readonly version: string;
  readonly runtime: DoctorRuntimeStatus;
  readonly gitError?: string;
  readonly configContext: DoctorConfigContext;
  readonly modelRegistry: DoctorModelRegistryStatus;
  readonly providerCredentialIssue?: DoctorProviderCredentialIssue;
  readonly providerModelIssue?: ProviderModelCompatibilityIssue;
}

type Mutable<T> = {
  -readonly [Property in keyof T]: T[Property];
};

function makeCheck(
  label: string,
  status: DoctorCheckStatus,
  detail?: string,
): DoctorCheck {
  return { label, status, ...(detail ? { detail } : {}) };
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
      rawText: text,
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

function matchProviderTomlString(
  text: string,
  providerId: string,
  key: string,
): string | undefined {
  const sectionMatch = text.match(
    new RegExp(
      `^\\[providers\\.${escapeRegExp(providerId)}\\]\\s*$([\\s\\S]*?)(?=^\\[|\\Z)`,
      "m",
    ),
  );
  if (!sectionMatch?.[1]) {
    return undefined;
  }
  return matchTopLevelTomlString(sectionMatch[1], key);
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
    providerConfigApiKey: fileConfig.rawText
      ? (matchProviderTomlString(fileConfig.rawText, providerId, "api_key") ?? null)
      : undefined,
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

function getDoctorRuntimeStatus(): DoctorRuntimeStatus {
  const label = typeof Bun !== "undefined" ? `Bun ${Bun.version}` : `Node ${process.version}`;
  const runtimeMajor = parseInt(process.version.replace("v", ""), 10);
  const error =
    runtimeMajor < 20 && typeof Bun === "undefined"
      ? "Node.js >= 20 required"
      : undefined;
  return { label, ...(error ? { error } : {}) };
}

function getDoctorGitError(): string | undefined {
  try {
    execSync("git --version", { encoding: "utf-8", timeout: 5000 });
    return undefined;
  } catch {
    return "git not found in PATH";
  }
}

function loadDoctorConfigContext(): DoctorConfigContext {
  const projectRoot = findProjectRoot() ?? process.cwd();
  const globalHome = process.env["HOME"] ?? homedir();
  const configPaths = [getGlobalConfigPath(globalHome)];
  const foundConfig = configPaths.find((p) => existsSync(p));
  const fileConfig = loadDoctorConfigFile(foundConfig);
  return {
    configPaths,
    ...(foundConfig ? { foundConfig } : {}),
    fileConfig,
    config: loadConfig(projectRoot),
    storedCredentials: new CredentialStore().all(),
  };
}

function loadDoctorModelRegistryStatus(model: string): DoctorModelRegistryStatus {
  try {
    loadModelRegistry();
    const models = getRegisteredModels();
    const registered = models.includes(model);
    return {
      count: models.length,
      registered,
      ...(registered ? { providers: getProvidersForModel(model) } : {}),
    };
  } catch (err) {
    return {
      error: String(err),
      count: 0,
      registered: false,
    };
  }
}

function resolveDoctorProviderModelIssue(
  config: DevAgentConfig,
  modelRegistry: DoctorModelRegistryStatus,
): ProviderModelCompatibilityIssue | undefined {
  if (modelRegistry.error || !modelRegistry.registered) {
    return undefined;
  }
  return getProviderModelCompatibilityIssue(config.provider, config.model);
}

function buildDoctorReportInput(context: DoctorReportInputContext): DoctorReportInput {
  const input: Partial<Mutable<DoctorReportInput>> = {
    version: context.version,
    runtimeLabel: context.runtime.label,
    configSearchPaths: context.configContext.configPaths,
    config: context.configContext.config,
    providerStatuses: buildProviderStatuses(
      context.configContext.config,
      context.configContext.storedCredentials,
      context.configContext.fileConfig.topLevelApiKey,
    ),
    modelRegistryCount: context.modelRegistry.count,
    modelRegistered: context.modelRegistry.registered,
    lspStatuses: buildLspStatuses(),
    platformLabel: `${process.platform} ${process.arch}`,
    providerSource: resolveConfigValueSource("DEVAGENT_PROVIDER", context.configContext.fileConfig.topLevelProvider),
    modelSource: resolveConfigValueSource("DEVAGENT_MODEL", context.configContext.fileConfig.topLevelModel),
    credentialSource: resolveCredentialSource(
      context.configContext.config.provider,
      context.configContext.config,
      context.configContext.storedCredentials,
      context.configContext.fileConfig,
    ),
  };

  appendOptionalDoctorReportFields(input, context);
  return input as DoctorReportInput;
}

function appendOptionalDoctorReportFields(
  input: Partial<Mutable<DoctorReportInput>>,
  context: DoctorReportInputContext,
): void {
  if (context.runtime.error) input.runtimeError = context.runtime.error;
  if (context.gitError) input.gitError = context.gitError;
  if (context.configContext.foundConfig) input.configPath = context.configContext.foundConfig;
  if (context.providerCredentialIssue) input.providerCredentialIssue = context.providerCredentialIssue;
  if (context.modelRegistry.error) input.modelRegistryError = context.modelRegistry.error;
  if (context.modelRegistry.providers) input.modelProviders = context.modelRegistry.providers;
  if (context.providerModelIssue) input.providerModelIssue = context.providerModelIssue;
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
interface DoctorChecksForIssues {
  readonly runtimeCheck: DoctorCheck;
  readonly gitCheck: DoctorCheck;
  readonly configCheck: DoctorCheck;
  readonly providerCheck: DoctorCheck;
  readonly modelRegistryCheck: DoctorCheck;
  readonly modelCheck: DoctorCheck;
  readonly providerModelCheck: DoctorCheck;
}

type DoctorIssueBuilder = (
  input: DoctorReportInput,
  checks: DoctorChecksForIssues,
) => DoctorIssue | undefined;

const DOCTOR_ISSUE_BUILDERS: ReadonlyArray<DoctorIssueBuilder> = [
  buildRuntimeIssue,
  buildGitIssue,
  buildConfigIssue,
  buildProviderModelIssue,
  buildProviderCredentialBlockingIssue,
  buildModelRegistryIssue,
  buildModelIssue,
];

function buildBlockingIssues(input: DoctorReportInput, checks: DoctorChecksForIssues): DoctorIssue[] {
  return DOCTOR_ISSUE_BUILDERS
    .map((builder) => builder(input, checks))
    .filter((issue): issue is DoctorIssue => issue !== undefined);
}

function buildRuntimeIssue(_: DoctorReportInput, checks: DoctorChecksForIssues): DoctorIssue | undefined {
  if (checks.runtimeCheck.status !== "blocking") return undefined;
  return {
    title: "Runtime",
    detail: checks.runtimeCheck.detail ?? "runtime check failed",
    nextSteps: [
      "Install Node.js >= 20 (recommended on Ubuntu: nvm install 20 && nvm use 20).",
      "Or use Bun >= 1.3 if you prefer a Bun-first setup.",
      "Then retry: devagent doctor",
    ],
  };
}

function buildGitIssue(_: DoctorReportInput, checks: DoctorChecksForIssues): DoctorIssue | undefined {
  if (checks.gitCheck.status !== "blocking") return undefined;
  return {
    title: "Git",
    detail: checks.gitCheck.detail ?? "git not found in PATH",
    nextSteps: ["Install Git and retry devagent doctor."],
  };
}

function buildConfigIssue(_: DoctorReportInput, checks: DoctorChecksForIssues): DoctorIssue | undefined {
  if (checks.configCheck.status !== "blocking") return undefined;
  return {
    title: "Config file",
    detail: checks.configCheck.detail ?? "config file not found",
    nextSteps: [
      "Run: devagent setup",
      "Or create ~/.config/devagent/config.toml with your provider and model.",
    ],
  };
}

function buildProviderModelIssue(
  input: DoctorReportInput,
  checks: DoctorChecksForIssues,
): DoctorIssue | undefined {
  if (!input.providerModelIssue || checks.providerModelCheck.status !== "blocking") return undefined;
  return {
    title: "Provider/model pairing",
    detail: formatProviderModelCompatibilityError(input.providerModelIssue),
    nextSteps: buildProviderModelIssueSteps(input.providerModelIssue),
  };
}

function buildProviderCredentialBlockingIssue(
  input: DoctorReportInput,
  checks: DoctorChecksForIssues,
): DoctorIssue | undefined {
  if (checks.providerCheck.status !== "blocking") return undefined;
  return {
    title: "Provider credentials",
    detail: checks.providerCheck.detail ?? `provider "${input.config.provider}" is missing credentials`,
    nextSteps: buildProviderCredentialIssueSteps(input.config.provider),
  };
}

function buildModelRegistryIssue(_: DoctorReportInput, checks: DoctorChecksForIssues): DoctorIssue | undefined {
  if (checks.modelRegistryCheck.status !== "blocking") return undefined;
  return {
    title: "Model registry",
    detail: checks.modelRegistryCheck.detail ?? "model registry failed to load",
    nextSteps: [
      "Rebuild or reinstall DevAgent so bundled model definitions are available.",
      "Then rerun: devagent doctor",
    ],
  };
}

function buildModelIssue(input: DoctorReportInput, checks: DoctorChecksForIssues): DoctorIssue | undefined {
  if (checks.modelCheck.status !== "blocking") return undefined;
  return {
    title: "Model",
    detail: checks.modelCheck.detail ?? `model "${input.config.model}" is not registered`,
    nextSteps: [
      "Run: devagent setup",
      `Or choose a registered model for provider "${input.config.provider}".`,
    ],
  };
}
export function buildDoctorReport(input: DoctorReportInput): DoctorReport {
  const checks = buildDoctorChecks(input);
  const platformCheck = makeCheck(`Platform: ${input.platformLabel}`, "pass");
  const blockingIssues = buildBlockingIssues(input, checks);
  const hasAdvisories = doctorHasAdvisories(input.lspStatuses, checks);
  const ok = blockingIssues.length === 0;
  const summaryStatus = ok ? (hasAdvisories ? "advisory" : "pass") : "blocking";

  return {
    version: input.version,
    blockingIssues,
    runtimeCheck: checks.runtimeCheck,
    gitCheck: checks.gitCheck,
    configCheck: checks.configCheck,
    providerCheck: checks.providerCheck,
    providerStatuses: input.providerStatuses,
    modelRegistryCheck: checks.modelRegistryCheck,
    modelCheck: checks.modelCheck,
    providerModelCheck: checks.providerModelCheck,
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

function buildDoctorChecks(input: DoctorReportInput): DoctorChecksForIssues {
  return {
    runtimeCheck: makeCheck(
      `Runtime: ${input.runtimeLabel}`,
      input.runtimeError ? "blocking" : "pass",
      input.runtimeError,
    ),
    gitCheck: makeCheck("Git", input.gitError ? "blocking" : "pass", input.gitError),
    configCheck: buildConfigFileCheck(input),
    providerCheck: makeCheck(
      `Provider: ${input.config.provider}`,
      input.providerCredentialIssue?.status ?? "pass",
      input.providerCredentialIssue?.detail,
    ),
    modelRegistryCheck: buildModelRegistryCheck(input),
    modelCheck: buildModelCheck(input),
    providerModelCheck: buildProviderModelCheck(input),
  };
}

function buildConfigFileCheck(input: DoctorReportInput): DoctorCheck {
  return makeCheck(
    "Config file",
    input.configPath ? "pass" : "blocking",
    input.configPath ? undefined : `not found (searched: ${input.configSearchPaths.join(", ")})`,
  );
}

function buildModelRegistryCheck(input: DoctorReportInput): DoctorCheck {
  return input.modelRegistryError
    ? makeCheck("Model registry", "blocking", input.modelRegistryError)
    : makeCheck(`Model registry: ${input.modelRegistryCount ?? 0} models loaded`, "pass");
}

function buildModelCheck(input: DoctorReportInput): DoctorCheck {
  if (input.modelRegistryError) {
    return makeCheck("Model", "advisory", "skipped until model registry loads");
  }
  return makeCheck(
    `Model: ${input.config.model}`,
    input.modelRegistered ? "pass" : "blocking",
    input.modelRegistered ? undefined : `model "${input.config.model}" not in registry`,
  );
}

function buildProviderModelCheck(input: DoctorReportInput): DoctorCheck {
  if (input.modelRegistryError || !input.modelRegistered) {
    return makeCheck("Provider/model pairing", "advisory", "skipped until the configured model is known");
  }
  if (!input.providerModelIssue) {
    return makeCheck("Provider/model pairing", "pass");
  }
  return makeCheck(
    "Provider/model pairing",
    "blocking",
    [
      formatProviderModelCompatibilityError(input.providerModelIssue),
      formatProviderModelCompatibilityHint(input.providerModelIssue),
    ].filter(Boolean).join(" "),
  );
}

function doctorHasAdvisories(
  lspStatuses: ReadonlyArray<DoctorLspStatus>,
  checks: DoctorChecksForIssues,
): boolean {
  return [
    checks.providerCheck,
    checks.modelCheck,
    checks.providerModelCheck,
  ].some((check) => check.status === "advisory") || !lspStatuses.some((lsp) => lsp.found);
}
export { renderDoctorReport } from "./doctor-render.js";

// ─── doctor ─────────────────────────────────────────────────
export async function runDoctor(version: string, args: ReadonlyArray<string> = []): Promise<void> {
  if (hasHelpFlag(args)) {
    writeStdout(renderDoctorHelpText());
    return;
  }

  migrateLegacyGlobalConfigIfNeeded();
  migrateLegacyGlobalTomlIfNeeded();
  normalizeGlobalConfigIfNeeded();
  const runtime = getDoctorRuntimeStatus();
  const gitError = getDoctorGitError();
  const configContext = loadDoctorConfigContext();
  const modelRegistry = loadDoctorModelRegistryStatus(configContext.config.model);
  const providerModelIssue = resolveDoctorProviderModelIssue(configContext.config, modelRegistry);
  const activeProviderHasCredential = hasProviderCredential(
    configContext.config.provider,
    configContext.config,
    configContext.storedCredentials,
    configContext.fileConfig.topLevelApiKey,
  );
  const providerCredentialIssue = buildProviderCredentialIssue(
    configContext.config.provider,
    activeProviderHasCredential,
    Boolean(providerModelIssue),
  );

  const report = buildDoctorReport(buildDoctorReportInput({
    version,
    runtime,
    ...(gitError ? { gitError } : {}),
    configContext,
    modelRegistry,
    ...(providerCredentialIssue ? { providerCredentialIssue } : {}),
    ...(providerModelIssue ? { providerModelIssue } : {}),
  }));

  process.stdout.write(renderDoctorReport(report) + "\n");
  process.exit(report.ok ? 0 : 1);
}
