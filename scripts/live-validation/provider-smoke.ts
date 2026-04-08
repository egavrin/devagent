#!/usr/bin/env bun

import { spawn } from "node:child_process";
import { mkdir, mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { PROTOCOL_VERSION, type TaskExecutionRequest } from "@devagent-sdk/types";
import {
  CredentialStore,
  getProviderCredentialDescriptor,
  type CredentialInfo,
} from "../../packages/runtime/src/index.ts";

type ProviderId =
  | "devagent-api"
  | "openai"
  | "openrouter"
  | "deepseek"
  | "chatgpt"
  | "github-copilot"
  | "ollama";

type SmokeStatus = "passed" | "failed" | "blocked";

interface ProviderSmokeCheck {
  readonly label: string;
  readonly command: string;
  readonly exitCode?: number;
  readonly durationMs: number;
  readonly status: SmokeStatus;
  readonly blockedReason?: string;
  readonly stdoutPath?: string;
  readonly stderrPath?: string;
}

interface ProviderSmokeReport {
  readonly provider: ProviderId;
  readonly model: string;
  readonly status: SmokeStatus;
  readonly checks: ReadonlyArray<ProviderSmokeCheck>;
}

interface CliOptions {
  readonly outputRoot?: string;
}

interface CommandResult {
  readonly exitCode: number;
  readonly stdout: string;
  readonly stderr: string;
  readonly timedOut: boolean;
  readonly durationMs: number;
}

interface OllamaModelSelection {
  readonly model: string | null;
  readonly blockedReason?: string;
}

function extractStatusCode(output: string): number | null {
  const match = output.match(/statusCode:\s*(\d{3})/);
  return match ? Number.parseInt(match[1]!, 10) : null;
}

function extractUrl(output: string): string | null {
  const quoted = output.match(/url:\s*'([^']+)'/);
  if (quoted) return quoted[1]!;
  const plain = output.match(/https?:\/\/[^\s'"]+/);
  return plain ? plain[0]! : null;
}

export function classifyProviderFailure(result: CommandResult): Pick<ProviderSmokeCheck, "status" | "blockedReason"> {
  const output = `${result.stderr}\n${result.stdout}`;
  const statusCode = extractStatusCode(output);
  const url = extractUrl(output);
  const lowered = output.toLowerCase();

  if (
    statusCode === 401
    && (lowered.includes("invalid api key") || lowered.includes("authentication fails") || lowered.includes("authentication_error"))
  ) {
    return {
      status: "blocked",
      blockedReason: "invalid stored credential",
    };
  }

  if (statusCode !== null && statusCode >= 500 && statusCode <= 599) {
    return {
      status: "blocked",
      blockedReason: url
        ? `provider service unavailable (${statusCode}) at ${url}`
        : `provider service unavailable (${statusCode})`,
    };
  }

  return {
    status: "failed",
  };
}

const DEFAULT_MODELS: Readonly<Record<Exclude<ProviderId, "ollama">, string>> = {
  "devagent-api": "cortex",
  openai: "gpt-5.4-mini",
  openrouter: "openai/gpt-4o-mini",
  deepseek: "deepseek-chat",
  chatgpt: "gpt-5.4-mini",
  "github-copilot": "gpt-4o",
};

function parseArgs(argv: string[]): CliOptions {
  let outputRoot: string | undefined;
  for (let index = 2; index < argv.length; index++) {
    const arg = argv[index]!;
    if (arg === "--output-dir" && argv[index + 1]) {
      outputRoot = argv[++index]!;
    } else if (arg === "--help" || arg === "-h") {
      process.stdout.write(
        [
          "Provider smoke matrix",
          "",
          "Usage:",
          "  bun run scripts/live-validation/provider-smoke.ts",
          "  bun run scripts/live-validation/provider-smoke.ts --output-dir <path>",
          "",
        ].join("\n"),
      );
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }
  return { outputRoot };
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
  return await new Promise((resolvePromise) => {
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
      resolvePromise({
        exitCode: code ?? (timedOut ? 124 : 1),
        stdout,
        stderr,
        timedOut,
        durationMs: Date.now() - startedAt,
      });
    });
  });
}

function loadStoredCredentials(): Readonly<Record<string, CredentialInfo>> {
  return new CredentialStore().all();
}

function isCredentialExpired(credential: CredentialInfo | undefined): boolean {
  return credential?.type === "oauth"
    && credential.expiresAt !== undefined
    && credential.expiresAt <= Date.now();
}

async function seedCredential(homeDir: string, provider: ProviderId, credential: CredentialInfo | undefined): Promise<void> {
  if (!credential || isCredentialExpired(credential)) {
    return;
  }
  const store = new CredentialStore({
    filePath: join(homeDir, ".config", "devagent", "credentials.json"),
  });
  await mkdir(join(homeDir, ".config", "devagent"), { recursive: true });
  store.set(provider, credential);
}

async function createIsolatedEnv(outputDir: string, provider: ProviderId): Promise<NodeJS.ProcessEnv> {
  const homeDir = join(outputDir, provider, "home");
  const workDir = join(outputDir, provider, "workspace");
  await mkdir(workDir, { recursive: true });
  await mkdir(join(homeDir, ".config", "devagent"), { recursive: true });
  const storedCredentials = loadStoredCredentials();
  await seedCredential(homeDir, provider, storedCredentials[provider]);
  return {
    ...process.env,
    HOME: homeDir,
    XDG_CONFIG_HOME: join(homeDir, ".config"),
    XDG_CACHE_HOME: join(homeDir, ".cache"),
    DEVAGENT_DISABLE_UPDATE_CHECK: "1",
    NO_COLOR: "1",
    FORCE_COLOR: "0",
  };
}

function providerBlockedReason(provider: ProviderId, credential: CredentialInfo | undefined): string | null {
  if (provider === "ollama") {
    return null;
  }
  const descriptor = getProviderCredentialDescriptor(provider);
  if (!descriptor || descriptor.credentialMode === "none") {
    return null;
  }
  if (!credential) {
    return `${provider} credential is not stored locally.`;
  }
  if (isCredentialExpired(credential)) {
    return `${provider} credential is expired. Refresh with devagent auth login.`;
  }
  return null;
}

const PREFERRED_OLLAMA_MODEL = "qwen3.5:9b";

export function selectPreferredOllamaModel(ollamaListOutput: string): OllamaModelSelection {
  const candidates = ollamaListOutput
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .slice(1)
    .map((line) => line.split(/\s+/)[0] ?? "")
    .filter(Boolean);
  const hasPreferredModel = candidates.includes(PREFERRED_OLLAMA_MODEL);
  if (hasPreferredModel) {
    return { model: PREFERRED_OLLAMA_MODEL };
  }
  return {
    model: null,
    blockedReason: `Required Ollama model "${PREFERRED_OLLAMA_MODEL}" is not installed locally.`,
  };
}

async function selectOllamaModel(): Promise<OllamaModelSelection> {
  const result = await runCommand(
    process.env["SHELL"] ?? "/bin/sh",
    ["-lc", "ollama list"],
    process.cwd(),
    process.env,
    10_000,
  );
  if (result.exitCode !== 0) {
    return {
      model: null,
      blockedReason: "Failed to run `ollama list`.",
    };
  }
  return selectPreferredOllamaModel(result.stdout);
}

async function writeSmokeArtifacts(
  providerDir: string,
  label: string,
  result: CommandResult,
): Promise<Pick<ProviderSmokeCheck, "stdoutPath" | "stderrPath">> {
  const base = label.replace(/[^a-z0-9]+/gi, "-").toLowerCase();
  const stdoutPath = join(providerDir, `${base}.stdout.txt`);
  const stderrPath = join(providerDir, `${base}.stderr.txt`);
  await writeFile(stdoutPath, result.stdout);
  await writeFile(stderrPath, result.stderr);
  return { stdoutPath, stderrPath };
}

function buildExecuteRequest(repoRoot: string, provider: ProviderId, model: string): TaskExecutionRequest {
  return {
    protocolVersion: PROTOCOL_VERSION,
    taskId: `provider-smoke-${provider}`,
    taskType: "plan",
    workspaceRef: {
      id: "workspace-1",
      name: "provider-smoke",
      provider: "local",
      primaryRepositoryId: "repo-1",
    },
    repositories: [{
      id: "repo-1",
      workspaceId: "workspace-1",
      alias: "primary",
      name: "repo",
      repoRoot,
      repoFullName: "local/provider-smoke",
      defaultBranch: "main",
      provider: "local",
    }],
    workItem: {
      id: "item-1",
      kind: "local-task",
      externalId: "provider-smoke",
      title: "Create a tiny plan",
      repositoryId: "repo-1",
    },
    execution: {
      primaryRepositoryId: "repo-1",
      repositories: [{
        repositoryId: "repo-1",
        alias: "primary",
        sourceRepoPath: repoRoot,
        baseRef: "main",
        workBranch: "devagent/provider-smoke",
        isolation: "temp-copy",
      }],
    },
    targetRepositoryIds: ["repo-1"],
    executor: {
      executorId: "devagent",
      provider,
      model,
      approvalMode: "full-auto",
      reasoning: "low",
    },
    constraints: {
      maxIterations: 1,
      timeoutSec: 120,
      allowNetwork: true,
    },
    capabilities: {
      canSyncTasks: true,
      canCreateTask: true,
      canComment: true,
      canReview: true,
      canMerge: true,
      canOpenReviewable: true,
    },
    context: {
      summary: "Return a short plan only.",
      issueBody: "Do not modify files.",
      skills: [],
    },
    expectedArtifacts: ["plan"],
  };
}

async function runProviderChecks(
  devagentRoot: string,
  outputRoot: string,
  provider: ProviderId,
  model: string,
): Promise<ProviderSmokeReport> {
  const providerDir = join(outputRoot, provider);
  await mkdir(providerDir, { recursive: true });
  const storedCredential = loadStoredCredentials()[provider];
  const blockedReason = providerBlockedReason(provider, storedCredential);
  if (blockedReason) {
    return {
      provider,
      model,
      status: "blocked",
      checks: [{
        label: "credential",
        command: "auth status",
        durationMs: 0,
        status: "blocked",
        blockedReason,
      }],
    };
  }

  const env = await createIsolatedEnv(outputRoot, provider);
  const command = getDevagentCommand(devagentRoot);
  const workDir = join(outputRoot, provider, "workspace");
  const checks: ProviderSmokeCheck[] = [];

  const runAndRecord = async (
    label: string,
    args: string[],
    timeoutMs: number,
  ): Promise<void> => {
    const result = await runCommand(command.executable, [...command.baseArgs, ...args], workDir, env, timeoutMs);
    const paths = await writeSmokeArtifacts(providerDir, label, result);
    const classified = result.exitCode === 0
      ? { status: "passed" as const, blockedReason: undefined }
      : classifyProviderFailure(result);
    checks.push({
      label,
      command: [command.executable, ...command.baseArgs, ...args].join(" "),
      exitCode: result.exitCode,
      durationMs: result.durationMs,
      status: classified.status,
      ...(classified.blockedReason ? { blockedReason: classified.blockedReason } : {}),
      stdoutPath: paths.stdoutPath,
      stderrPath: paths.stderrPath,
    });
  };

  if (provider === "devagent-api") {
    await writeFile(join(workDir, "prompt.md"), "Reply with exactly: OK\n");
    await runAndRecord("quiet-query", ["--provider", provider, "--model", model, "--quiet", "Reply with exactly: OK"], 120_000);
    await runAndRecord("file-query", ["--provider", provider, "--model", model, "--quiet", "-f", join(workDir, "prompt.md")], 120_000);
  } else if (provider === "openai" || provider === "deepseek" || provider === "chatgpt" || provider === "github-copilot" || provider === "ollama") {
    await runAndRecord("quiet-query", ["--provider", provider, "--model", model, "--quiet", "Reply with exactly: OK"], provider === "ollama" ? 180_000 : 120_000);
  } else if (provider === "openrouter") {
    await mkdir(join(workDir, "repo"), { recursive: true });
    await runCommand("git", ["init", "-q"], join(workDir, "repo"), env, 10_000);
    await writeFile(join(workDir, "repo", "hello.txt"), "hello\n");
    const requestPath = join(workDir, "request.json");
    await writeFile(requestPath, JSON.stringify(buildExecuteRequest(join(workDir, "repo"), provider, model), null, 2));
    await mkdir(join(workDir, "artifacts"), { recursive: true });
    await runAndRecord("execute", ["execute", "--request", requestPath, "--artifact-dir", join(workDir, "artifacts")], 180_000);
  }

  const status = checks.every((check) => check.status === "passed")
    ? "passed"
    : checks.some((check) => check.status === "failed")
      ? "failed"
      : "blocked";
  return { provider, model, status, checks };
}

function renderMarkdown(reports: ReadonlyArray<ProviderSmokeReport>): string {
  const lines = [
    "# Provider Smoke Summary",
    "",
  ];
  for (const report of reports) {
    lines.push(`## ${report.provider}`);
    lines.push(`- Model: ${report.model}`);
    lines.push(`- Status: ${report.status}`);
    for (const check of report.checks) {
      lines.push(`- ${check.label}: ${check.status}${check.exitCode !== undefined ? ` (exit ${check.exitCode})` : ""}${check.blockedReason ? ` — ${check.blockedReason}` : ""}`);
    }
    lines.push("");
  }
  return lines.join("\n").trim() + "\n";
}

async function main(): Promise<void> {
  const options = parseArgs(process.argv);
  const scriptDir = dirname(fileURLToPath(import.meta.url));
  const devagentRoot = dirname(dirname(scriptDir));
  const outputRoot = options.outputRoot
    ? options.outputRoot
    : await mkdtemp(join(tmpdir(), "devagent-provider-smoke-"));
  await mkdir(outputRoot, { recursive: true });

  const ollamaSelection = await selectOllamaModel();
  const matrix: Array<{ provider: ProviderId; model: string | null }> = [
    { provider: "devagent-api", model: DEFAULT_MODELS["devagent-api"] },
    { provider: "openai", model: DEFAULT_MODELS.openai },
    { provider: "openrouter", model: DEFAULT_MODELS.openrouter },
    { provider: "deepseek", model: DEFAULT_MODELS.deepseek },
    { provider: "chatgpt", model: DEFAULT_MODELS.chatgpt },
    { provider: "github-copilot", model: DEFAULT_MODELS["github-copilot"] },
    { provider: "ollama", model: ollamaSelection.model },
  ];

  const reports: ProviderSmokeReport[] = [];
  for (const entry of matrix) {
    if (!entry.model) {
      reports.push({
        provider: entry.provider,
        model: "(unavailable)",
        status: "blocked",
        checks: [{
          label: "model-selection",
          command: "ollama list",
          durationMs: 0,
          status: "blocked",
          blockedReason: ollamaSelection.blockedReason ?? `Required Ollama model "${PREFERRED_OLLAMA_MODEL}" is not installed locally.`,
        }],
      });
      continue;
    }
    process.stdout.write(`Running provider smoke for ${entry.provider}...\n`);
    reports.push(await runProviderChecks(devagentRoot, outputRoot, entry.provider, entry.model));
  }

  await writeFile(join(outputRoot, "provider-smoke-summary.json"), JSON.stringify(reports, null, 2));
  await writeFile(join(outputRoot, "provider-smoke-summary.md"), renderMarkdown(reports));
  process.stdout.write(`Provider smoke summary written to ${outputRoot}\n`);
  if (reports.some((report) => report.status === "failed")) {
    process.exit(1);
  }
}

if (import.meta.main) {
  main().catch((error) => {
    process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
    process.exit(1);
  });
}
