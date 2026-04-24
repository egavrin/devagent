#!/usr/bin/env bun

import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { mkdir, mkdtemp, readFile, readdir, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { buildPrompts, createFixture } from "./tool-script-adoption-fixture.js";
import {
  CredentialStore,
  getProviderCredentialDescriptor,
  type CredentialInfo,
} from "../../packages/runtime/src/index.ts";

type ProviderId =
  | "anthropic"
  | "devagent-api"
  | "openai"
  | "openrouter"
  | "deepseek"
  | "chatgpt"
  | "github-copilot"
  | "ollama";

type CheckKind = "natural" | "explicit";
type CheckStatus = "passed" | "failed" | "blocked";
type FailureReason =
  | "command_timeout"
  | "command_failed"
  | "no_batching"
  | "late_batching"
  | "post_batch_direct_reads"
  | "script_error"
  | "oversized_stdout"
  | "line_dump_stdout";

interface CliOptions {
  readonly provider: ProviderId;
  readonly model: string;
  readonly outputRoot?: string;
}

interface CommandResult {
  readonly exitCode: number;
  readonly stdout: string;
  readonly stderr: string;
  readonly timedOut: boolean;
  readonly durationMs: number;
}

export interface AdoptionPrompt {
  readonly label: string;
  readonly kind: CheckKind;
  readonly prompt: string;
}

interface AdoptionCheck {
  readonly label: string;
  readonly kind: CheckKind;
  readonly status: CheckStatus;
  readonly firstTool?: string;
  readonly firstInspectionTool?: string;
  readonly usedExecuteToolScript: boolean;
  readonly toolScriptTelemetry?: Record<string, unknown>;
  readonly directReadonlyBeforeExecute: number;
  readonly directReadonlyAfterExecute: number;
  readonly compactFinalOutput: boolean;
  readonly lineDumpOutput: boolean;
  readonly innerOutputChars?: number;
  readonly finalOutputChars?: number;
  readonly hiddenToFinalRatio?: number;
  readonly failedBecause?: FailureReason;
  readonly exitCode?: number;
  readonly durationMs: number;
  readonly blockedReason?: string;
  readonly stdoutPath: string;
  readonly stderrPath: string;
  readonly logPath?: string;
}

interface LogEntry {
  readonly event: string;
  readonly data: unknown;
}

type ClassificationSummary = Pick<
  AdoptionCheck,
  | "firstTool"
  | "firstInspectionTool"
  | "usedExecuteToolScript"
  | "toolScriptTelemetry"
  | "directReadonlyBeforeExecute"
  | "directReadonlyAfterExecute"
  | "compactFinalOutput"
  | "lineDumpOutput"
  | "innerOutputChars"
  | "finalOutputChars"
  | "hiddenToFinalRatio"
>;
type ClassificationResult = ClassificationSummary & Pick<AdoptionCheck, "status" | "failedBecause" | "blockedReason">;

const DEFAULT_MODELS: Readonly<Record<ProviderId, string>> = {
  anthropic: "claude-sonnet-4-20250514",
  "devagent-api": "cortex",
  openai: "gpt-5.4-mini",
  openrouter: "openai/gpt-4o-mini",
  deepseek: "deepseek-chat",
  chatgpt: "gpt-5.4-mini",
  "github-copilot": "gpt-4o",
  ollama: "llama3.2",
};

const NATURAL_ACCEPTANCE_MIN = 3;
const DIRECT_READONLY_TOOLS = new Set([
  "read_file",
  "search_files",
  "find_files",
  "git_status",
  "git_diff",
  "symbols",
  "definitions",
  "references",
  "lsp",
]);
const INSPECTION_TOOLS = new Set([...DIRECT_READONLY_TOOLS, "execute_tool_script"]);
const COMPACT_STDOUT_LIMIT = 4096;

function parseArgs(argv: string[]): CliOptions {
  let provider: ProviderId = "deepseek";
  let model: string | undefined;
  let outputRoot: string | undefined;

  for (let index = 2; index < argv.length; index++) {
    const arg = argv[index]!;
    if (arg === "--help" || arg === "-h") {
      process.stdout.write([
        "Tool Script Adoption Validation",
        "",
        "Usage:",
        "  bun run scripts/live-validation/tool-script-adoption.ts --provider deepseek --model deepseek-v4-pro",
        "  bun run scripts/live-validation/tool-script-adoption.ts --output-dir <path>",
        "",
      ].join("\n"));
      process.exit(0);
    }
    const value = readFlagValue(argv, index, arg);
    if (arg === "--provider") provider = value as ProviderId;
    else if (arg === "--model") model = value;
    else if (arg === "--output-dir") outputRoot = value;
    else throw new Error(`Unknown argument: ${arg}`);
    index++;
  }

  return { provider, model: model ?? DEFAULT_MODELS[provider], outputRoot };
}

function readFlagValue(argv: string[], index: number, arg: string): string {
  const value = argv[index + 1];
  if (!value) throw new Error(`Missing value for ${arg}`);
  return value;
}

function getDevagentCommand(devagentRoot: string): { executable: string; baseArgs: string[] } {
  return {
    executable: "bun",
    baseArgs: [join(devagentRoot, "packages", "cli", "src", "index.ts")],
  };
}

async function runCommand(
  executable: string,
  args: ReadonlyArray<string>,
  cwd: string,
  env: NodeJS.ProcessEnv,
  timeoutMs: number,
): Promise<CommandResult> {
  const startedAt = Date.now();
  return await new Promise((resolve) => {
    const child = spawn(executable, [...args], {
      cwd,
      env,
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      child.kill("SIGKILL");
    }, timeoutMs);
    child.stdout.on("data", (chunk: Buffer) => {
      stdout += chunk.toString();
    });
    child.stderr.on("data", (chunk: Buffer) => {
      stderr += chunk.toString();
    });
    child.once("close", (code) => {
      clearTimeout(timer);
      resolve({
        exitCode: code ?? (timedOut ? 124 : 1),
        stdout,
        stderr,
        timedOut,
        durationMs: Date.now() - startedAt,
      });
    });
  });
}

function loadStoredCredential(provider: ProviderId): CredentialInfo | undefined {
  return new CredentialStore().all()[provider];
}

function isCredentialExpired(credential: CredentialInfo | undefined): boolean {
  return credential?.type === "oauth"
    && credential.expiresAt !== undefined
    && credential.expiresAt <= Date.now();
}

async function seedCredential(homeDir: string, provider: ProviderId): Promise<void> {
  const credential = loadStoredCredential(provider);
  if (!credential || isCredentialExpired(credential)) return;

  const configDir = join(homeDir, ".config", "devagent");
  await mkdir(configDir, { recursive: true });
  new CredentialStore({ filePath: join(configDir, "credentials.json") }).set(provider, credential);
}

function providerBlockedReason(provider: ProviderId): string | null {
  if (provider === "ollama") return null;
  const credential = loadStoredCredential(provider);
  if (!credential) {
    const descriptor = getProviderCredentialDescriptor(provider);
    return `${provider} credential is not stored locally (${descriptor.envVar}).`;
  }
  if (isCredentialExpired(credential)) {
    return `${provider} credential is expired. Refresh with devagent auth login.`;
  }
  return null;
}

async function writeCommandArtifacts(
  outputDir: string,
  label: string,
  result: CommandResult,
): Promise<{ stdoutPath: string; stderrPath: string }> {
  const stdoutPath = join(outputDir, `${label}.stdout.txt`);
  const stderrPath = join(outputDir, `${label}.stderr.txt`);
  await writeFile(stdoutPath, result.stdout);
  await writeFile(stderrPath, result.stderr);
  return { stdoutPath, stderrPath };
}

async function readLogEntries(logPath: string | null): Promise<LogEntry[]> {
  if (!logPath) return [];
  const text = await readFile(logPath, "utf-8");
  return text.trim().length === 0
    ? []
    : text.trim().split("\n").map((line) => JSON.parse(line) as LogEntry);
}

async function newestLogPath(logDir: string): Promise<string | null> {
  if (!existsSync(logDir)) return null;
  const files = (await readdir(logDir)).filter((file) => file.endsWith(".jsonl"));
  if (files.length === 0) return null;
  const withStats = await Promise.all(files.map(async (file) => {
    const path = join(logDir, file);
    return { path, mtimeMs: (await stat(path)).mtimeMs };
  }));
  withStats.sort((a, b) => a.mtimeMs - b.mtimeMs);
  return withStats[withStats.length - 1]!.path;
}

function getToolName(entry: LogEntry): string | null {
  if (entry.event !== "tool:before" || !entry.data || typeof entry.data !== "object") return null;
  const name = (entry.data as Record<string, unknown>)["name"];
  return typeof name === "string" ? name : null;
}

function getToolCallId(entry: LogEntry): string | null {
  if (!entry.data || typeof entry.data !== "object") return null;
  const callId = (entry.data as Record<string, unknown>)["callId"];
  return typeof callId === "string" ? callId : null;
}

function isNestedScriptToolCall(entry: LogEntry): boolean {
  return getToolCallId(entry)?.includes("_script_") ?? false;
}

function getDirectToolNames(entries: ReadonlyArray<LogEntry>): string[] {
  return entries
    .filter((entry) => entry.event === "tool:before" && !isNestedScriptToolCall(entry))
    .map(getToolName)
    .filter((name): name is string => name !== null);
}

function getToolAfterTelemetry(entries: ReadonlyArray<LogEntry>): Record<string, unknown> | undefined {
  const result = getToolScriptResult(entries);
  const metadata = result?.["metadata"];
  if (!metadata || typeof metadata !== "object") return undefined;
  const toolScript = (metadata as Record<string, unknown>)["toolScript"];
  return toolScript && typeof toolScript === "object" ? toolScript as Record<string, unknown> : undefined;
}

function getToolScriptOutput(entries: ReadonlyArray<LogEntry>): string {
  const output = getToolScriptResult(entries)?.["output"];
  return typeof output === "string" ? output : "";
}

function getToolScriptFailure(entries: ReadonlyArray<LogEntry>): string | undefined {
  const result = getToolScriptResult(entries);
  if (!result) return undefined;
  if (result["success"] !== false) return undefined;
  const error = result["error"];
  return typeof error === "string" ? error : "execute_tool_script failed";
}

function getToolScriptResult(entries: ReadonlyArray<LogEntry>): Record<string, unknown> | undefined {
  for (const entry of entries) {
    if (entry.event !== "tool:after" || !entry.data || typeof entry.data !== "object") continue;
    if (isNestedScriptToolCall(entry)) continue;
    const data = entry.data as Record<string, unknown>;
    if (data["name"] !== "execute_tool_script") continue;
    const result = data["result"];
    if (!result || typeof result !== "object") continue;
    return result as Record<string, unknown>;
  }
  return undefined;
}

function numberFromTelemetry(telemetry: Record<string, unknown> | undefined, key: string): number | undefined {
  const value = telemetry?.[key];
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function looksLikeLineDump(output: string): boolean {
  const numberedHits = output.match(/(?:^|\n)\s*\d{1,5}:\s+\S/g)?.length ?? 0;
  const headingBlocks = output.match(/(?:^|\n)#{1,3}\s+\S/g)?.length ?? 0;
  return numberedHits >= 5 || (headingBlocks >= 2 && numberedHits >= 3);
}

function classifyCheck(
  prompt: AdoptionPrompt,
  result: CommandResult,
  entries: ReadonlyArray<LogEntry>,
): ClassificationResult {
  const summary = buildClassificationSummary(entries);
  if (result.timedOut) return blockedClassification(summary);
  if (result.exitCode !== 0) return failedCommandClassification(summary);
  const failedBecause = classifyFailureReason(summary, getToolScriptFailure(entries));
  return { ...summary, status: failedBecause === undefined ? "passed" : "failed", failedBecause };
}

function buildClassificationSummary(entries: ReadonlyArray<LogEntry>): ClassificationSummary {
  const toolNames = getDirectToolNames(entries);
  const firstTool = toolNames[0];
  const firstInspectionTool = toolNames.find((name) => INSPECTION_TOOLS.has(name));
  const usedExecuteToolScript = toolNames.includes("execute_tool_script");
  const toolScriptTelemetry = getToolAfterTelemetry(entries);
  const executeIndex = toolNames.indexOf("execute_tool_script");
  const directReadonlyBeforeExecute = executeIndex >= 0
    ? toolNames.slice(0, executeIndex).filter((name) => DIRECT_READONLY_TOOLS.has(name)).length
    : toolNames.filter((name) => DIRECT_READONLY_TOOLS.has(name)).length;
  const directReadonlyAfterExecute = executeIndex >= 0
    ? toolNames.slice(executeIndex + 1).filter((name) => DIRECT_READONLY_TOOLS.has(name)).length
    : 0;
  const innerOutputChars = numberFromTelemetry(toolScriptTelemetry, "innerOutputChars");
  const finalOutputChars = numberFromTelemetry(toolScriptTelemetry, "finalOutputChars");
  const hiddenToFinalRatio = innerOutputChars !== undefined && finalOutputChars !== undefined && finalOutputChars > 0
    ? Number((innerOutputChars / finalOutputChars).toFixed(2))
    : undefined;
  const toolScriptOutput = getToolScriptOutput(entries);
  const compactFinalOutput = finalOutputChars === undefined ? true : finalOutputChars <= COMPACT_STDOUT_LIMIT;
  const lineDumpOutput = looksLikeLineDump(toolScriptOutput);
  return {
    firstTool,
    firstInspectionTool,
    usedExecuteToolScript,
    toolScriptTelemetry,
    directReadonlyBeforeExecute,
    directReadonlyAfterExecute,
    compactFinalOutput,
    lineDumpOutput,
    innerOutputChars,
    finalOutputChars,
    hiddenToFinalRatio,
  };
}

function blockedClassification(summary: ClassificationSummary): ClassificationResult {
  return {
    ...summary,
    status: "blocked",
    failedBecause: "command_timeout",
    blockedReason: "command timed out",
  };
}

function failedCommandClassification(summary: ClassificationSummary): ClassificationResult {
  return {
    ...summary,
    status: "failed",
    failedBecause: "command_failed",
  };
}

function classifyFailureReason(
  summary: ClassificationSummary,
  scriptFailure: string | undefined,
): FailureReason | undefined {
  if (!summary.usedExecuteToolScript) return "no_batching";
  if (summary.firstInspectionTool !== "execute_tool_script") return "late_batching";
  if (summary.directReadonlyBeforeExecute > 0) return "late_batching";
  if (scriptFailure) return "script_error";
  if (summary.directReadonlyAfterExecute > 0) return "post_batch_direct_reads";
  if (!summary.compactFinalOutput) return "oversized_stdout";
  if (summary.lineDumpOutput) return "line_dump_stdout";
  return undefined;
}

async function runCheck(input: {
  readonly prompt: AdoptionPrompt;
  readonly command: ReturnType<typeof getDevagentCommand>;
  readonly env: NodeJS.ProcessEnv;
  readonly outputDir: string;
  readonly repoDir: string;
  readonly logDir: string;
  readonly provider: ProviderId;
  readonly model: string;
}): Promise<AdoptionCheck> {
  const result = await runCommand(
    input.command.executable,
    [
      ...input.command.baseArgs,
      "--provider",
      input.provider,
      "--model",
      input.model,
      "--quiet",
      input.prompt.prompt,
    ],
    input.repoDir,
    input.env,
    180_000,
  );
  const artifacts = await writeCommandArtifacts(input.outputDir, input.prompt.label, result);
  const logPath = await newestLogPath(input.logDir);
  const entries = await readLogEntries(logPath);
  const classified = classifyCheck(input.prompt, result, entries);
  return {
    label: input.prompt.label,
    kind: input.prompt.kind,
    durationMs: result.durationMs,
    exitCode: result.exitCode,
    stdoutPath: artifacts.stdoutPath,
    stderrPath: artifacts.stderrPath,
    ...(logPath ? { logPath } : {}),
    ...classified,
  };
}

function summarizeStatus(checks: ReadonlyArray<AdoptionCheck>): CheckStatus {
  const natural = checks.filter((check) => check.kind === "natural");
  const explicit = checks.filter((check) => check.kind === "explicit");
  const naturalPassed = natural.filter((check) => check.status === "passed").length;
  const explicitPassed = explicit.every((check) => check.status === "passed");
  if (checks.some((check) => check.status === "blocked")) return "blocked";
  return naturalPassed >= NATURAL_ACCEPTANCE_MIN && explicitPassed ? "passed" : "failed";
}

function renderMarkdown(options: {
  readonly provider: ProviderId;
  readonly model: string;
  readonly status: CheckStatus;
  readonly checks: ReadonlyArray<AdoptionCheck>;
}): string {
  const lines = [
    "# Tool Script Adoption Summary",
    "",
    `- Provider: ${options.provider}`,
    `- Model: ${options.model}`,
    `- Status: ${options.status}`,
    "",
  ];
  for (const check of options.checks) {
    lines.push(...renderCheckMarkdown(check));
  }
  return lines.join("\n").trim() + "\n";
}

function renderCheckMarkdown(check: AdoptionCheck): string[] {
  const lines = [
    `## ${check.label}`,
    `- Kind: ${check.kind}`,
    `- Status: ${check.status}`,
    `- First tool: ${check.firstTool ?? "(none)"}`,
    `- First inspection tool: ${check.firstInspectionTool ?? "(none)"}`,
    `- Used execute_tool_script: ${yesNo(check.usedExecuteToolScript)}`,
    `- Direct readonly calls before execute_tool_script: ${check.directReadonlyBeforeExecute}`,
    `- Direct readonly calls after execute_tool_script: ${check.directReadonlyAfterExecute}`,
    `- Compact final stdout: ${yesNo(check.compactFinalOutput)}`,
    `- Line-dump style stdout: ${yesNo(check.lineDumpOutput)}`,
  ];
  return [...lines, ...renderOptionalCheckMarkdown(check), ""];
}

function yesNo(value: boolean): "yes" | "no" {
  return value ? "yes" : "no";
}

function renderOptionalCheckMarkdown(check: AdoptionCheck): string[] {
  return [
    check.failedBecause ? `- Failed because: ${check.failedBecause}` : null,
    renderHiddenChars(check),
    check.hiddenToFinalRatio !== undefined ? `- Hidden/final ratio: ${check.hiddenToFinalRatio}` : null,
    check.blockedReason ? `- Blocked: ${check.blockedReason}` : null,
    check.toolScriptTelemetry ? `- Telemetry: ${JSON.stringify(check.toolScriptTelemetry)}` : null,
  ].filter((line): line is string => line !== null);
}

function renderHiddenChars(check: AdoptionCheck): string | null {
  if (check.innerOutputChars === undefined || check.finalOutputChars === undefined) return null;
  return `- Hidden chars: ${check.innerOutputChars} -> ${check.finalOutputChars} stdout chars`;
}

async function main(): Promise<void> {
  const options = parseArgs(process.argv);
  const blockedReason = providerBlockedReason(options.provider);
  if (blockedReason) {
    process.stderr.write(`${blockedReason}\n`);
    process.exit(2);
  }

  const scriptDir = dirname(fileURLToPath(import.meta.url));
  const devagentRoot = dirname(dirname(scriptDir));
  const outputRoot = options.outputRoot
    ? options.outputRoot
    : await mkdtemp(join(tmpdir(), "devagent-tool-script-adoption-"));
  const repoDir = join(outputRoot, "repo");
  const homeDir = join(outputRoot, "home");
  const logDir = join(outputRoot, "logs");
  await mkdir(outputRoot, { recursive: true });
  await mkdir(repoDir, { recursive: true });
  await mkdir(logDir, { recursive: true });
  await createFixture(repoDir);
  await seedCredential(homeDir, options.provider);
  await mkdir(join(homeDir, ".config", "devagent"), { recursive: true });
  await writeFile(join(homeDir, ".config", "devagent", "config.toml"), [
    "[logging]",
    `log_dir = "${logDir.replaceAll("\\", "\\\\").replaceAll("\"", "\\\"")}"`,
    "",
  ].join("\n"));

  const command = getDevagentCommand(devagentRoot);
  const env = {
    ...process.env,
    HOME: homeDir,
  };
  const checks: AdoptionCheck[] = [];
  for (const prompt of buildPrompts()) {
    process.stdout.write(`Running ${prompt.label}...\n`);
    checks.push(await runCheck({
      prompt,
      command,
      env,
      outputDir: outputRoot,
      repoDir,
      logDir,
      provider: options.provider,
      model: options.model,
    }));
  }

  const status = summarizeStatus(checks);
  const report = { provider: options.provider, model: options.model, status, checks };
  await writeFile(join(outputRoot, "tool-script-adoption-summary.json"), JSON.stringify(report, null, 2));
  await writeFile(join(outputRoot, "tool-script-adoption-summary.md"), renderMarkdown(report));
  process.stdout.write(`Tool script adoption summary written to ${outputRoot}\n`);
  if (status === "failed") process.exit(1);
  if (status === "blocked") process.exit(2);
}

if (import.meta.main) {
  main().catch((error) => {
    process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
    process.exit(1);
  });
}
