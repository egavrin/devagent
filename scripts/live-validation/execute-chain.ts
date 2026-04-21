#!/usr/bin/env bun

import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { mkdir, mkdtemp, readFile, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import type { BreakdownDoc, IssueSpecDoc, TaskExecutionRequest, TaskExecutionResult } from "@devagent-sdk/types";
import {
  captureGitOutputs,
  createIsolationWorkspaceWithTimeout,
  destroyIsolationWorkspace,
  ensureGitIdentity,
} from "./isolation";
import {
  artifactFileNamesForChainStage,
  buildExecuteChainRequest,
  EXECUTE_CHAIN_STAGES,
  type ExecuteChainArtifactContext,
  type ExecuteChainStage,
} from "./execute-chain-lib";
import type { IsolationWorkspace } from "./types";

interface CliOptions {
  readonly outputRoot?: string;
  readonly provider: string;
  readonly model: string;
}

interface CommandResult {
  readonly exitCode: number;
  readonly stdout: string;
  readonly stderr: string;
  readonly timedOut: boolean;
  readonly durationMs: number;
}

interface StageArtifactEntry {
  readonly path: string;
  readonly exists: boolean;
  readonly sizeBytes?: number;
}

interface StageReport {
  readonly stage: ExecuteChainStage;
  readonly status: "passed" | "failed";
  readonly failureMessage?: string;
  readonly requestPath: string;
  readonly artifactDir: string;
  readonly command: {
    readonly executable: string;
    readonly args: ReadonlyArray<string>;
    readonly exitCode: number;
  };
  readonly workspaceEffect: {
    readonly expected: "non-mutating" | "mutating";
    readonly passed: boolean;
    readonly baselineStatus: string;
    readonly finalStatus: string;
  };
  readonly changedFiles: ReadonlyArray<string>;
  readonly artifactFiles: ReadonlyArray<StageArtifactEntry>;
  readonly stdoutPath: string;
  readonly stderrPath: string;
  readonly repoStatusPath: string;
  readonly repoDiffPath: string;
  readonly eventsPath?: string;
  readonly summary?: string;
}

interface ChainSummary {
  readonly provider: string;
  readonly model: string;
  readonly repo: string;
  readonly outputRoot: string;
  readonly status: "passed" | "failed";
  readonly implementationChangedFiles: ReadonlyArray<string>;
  readonly finalChangedFiles: ReadonlyArray<string>;
  readonly reports: ReadonlyArray<StageReport>;
}

const SOURCE_REPO = "/Users/egavrin/Documents/arkcompiler_runtime_core_docs";
const README_TARGET = "README.md";

function parseArgs(argv: string[]): CliOptions {
  let outputRoot: string | undefined;
  let provider = "chatgpt";
  let model = "gpt-5.4";

  for (let index = 2; index < argv.length; index++) {
    const arg = argv[index]!;
    if (arg === "--output-dir" && argv[index + 1]) {
      outputRoot = argv[++index]!;
    } else if (arg === "--provider" && argv[index + 1]) {
      provider = argv[++index]!;
    } else if (arg === "--model" && argv[index + 1]) {
      model = argv[++index]!;
    } else if (arg === "--help" || arg === "-h") {
      process.stdout.write([
        "Chained execute live validation",
        "",
        "Usage:",
        "  bun run scripts/live-validation/execute-chain.ts",
        "  bun run scripts/live-validation/execute-chain.ts --output-dir <path>",
        "  bun run scripts/live-validation/execute-chain.ts --provider chatgpt --model gpt-5.4",
        "",
      ].join("\n"));
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }

  return { outputRoot, provider, model };
}

async function runCommand(
  executable: string,
  args: ReadonlyArray<string>,
  cwd: string,
  timeoutMs: number,
): Promise<CommandResult> {
  const startedAt = Date.now();
  return await new Promise((resolvePromise) => {
    const child = spawn(executable, [...args], {
      cwd,
      env: {
        ...process.env,
        DEVAGENT_DISABLE_UPDATE_CHECK: "1",
      },
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

function extractChangedFiles(repoStatus: string): string[] {
  return repoStatus
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => line.replace(/^[A-Z?]+\s+/, ""))
    .filter(Boolean);
}

function stageExpectedWorkspaceEffect(stage: ExecuteChainStage): "non-mutating" | "mutating" {
  return stage === "implement" || stage === "repair" ? "mutating" : "non-mutating";
}

function isNoOpRepairAllowed(priorArtifacts: ExecuteChainArtifactContext): boolean {
  return priorArtifacts.reviewReport?.includes("No defects found.") ?? false;
}

async function writeStageOutputs(
  stageDir: string,
  requestPath: string,
  commandResult: CommandResult,
  repoStatus: string,
  repoDiff: string,
  eventsPath?: string,
): Promise<{
  readonly stdoutPath: string;
  readonly stderrPath: string;
  readonly repoStatusPath: string;
  readonly repoDiffPath: string;
  readonly eventsPath?: string;
}> {
  const stdoutPath = join(stageDir, "stdout.txt");
  const stderrPath = join(stageDir, "stderr.txt");
  const repoStatusPath = join(stageDir, "repo-status.txt");
  const repoDiffPath = join(stageDir, "repo-diff.txt");
  await Promise.all([
    writeFile(stdoutPath, commandResult.stdout),
    writeFile(stderrPath, commandResult.stderr),
    writeFile(repoStatusPath, repoStatus),
    writeFile(repoDiffPath, repoDiff),
  ]);
  return {
    stdoutPath,
    stderrPath,
    repoStatusPath,
    repoDiffPath,
    ...(eventsPath ? { eventsPath } : {}),
  };
}

async function collectArtifactEntries(
  artifactDir: string,
  stage: ExecuteChainStage,
): Promise<StageArtifactEntry[]> {
  const entries: StageArtifactEntry[] = [];
  for (const relativePath of artifactFileNamesForChainStage(stage)) {
    const fullPath = join(artifactDir, relativePath);
    const exists = existsSync(fullPath);
    entries.push({
      path: relativePath,
      exists,
      ...(exists ? { sizeBytes: (await stat(fullPath)).size } : {}),
    });
  }
  return entries;
}

function extractFinalAssistantSummary(result: TaskExecutionResult): string | undefined {
  const messages = result.session?.payload?.messages;
  if (!Array.isArray(messages)) {
    return undefined;
  }

  for (let index = messages.length - 1; index >= 0; index--) {
    const message = messages[index] as { role?: string; content?: unknown };
    if (message.role === "assistant" && typeof message.content === "string" && message.content.trim().length > 0) {
      return message.content.trim();
    }
  }
  return undefined;
}

export async function buildStageFailureMessage(
  commandResult: CommandResult,
  artifactDir: string,
): Promise<string | undefined> {
  const resultPath = join(artifactDir, "result.json");
  const details: string[] = [];

  if (existsSync(resultPath)) {
    try {
      const result = JSON.parse(await readFile(resultPath, "utf-8")) as TaskExecutionResult;
      details.push(`Result: ${result.status}`);
      if (result.error) {
        details.push(`Error: ${result.error.code}: ${result.error.message}`);
      }
      if (result.outcomeReason) {
        details.push(`Outcome reason: ${result.outcomeReason}`);
      }
      const finalSummary = extractFinalAssistantSummary(result);
      if (finalSummary) {
        details.push(`Final assistant summary: ${finalSummary}`);
      }
    } catch (error) {
      details.push(`Failed to parse ${resultPath}: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  const commandOutput = commandResult.stderr.trim() || commandResult.stdout.trim();
  if (commandOutput) {
    details.push(commandOutput);
  }
  if (commandResult.exitCode !== 0 && details.length === 0) {
    details.push(`Stage exited with ${commandResult.exitCode}.`);
  }
  return details.length > 0 ? details.join("\n") : undefined;
}

async function updatePriorArtifacts(
  stage: ExecuteChainStage,
  artifactDir: string,
  priorArtifacts: ExecuteChainArtifactContext,
): Promise<ExecuteChainArtifactContext> {
  if (stage === "design") {
    return {
      ...priorArtifacts,
      designDoc: await readFile(join(artifactDir, "design-doc.md"), "utf-8"),
    };
  }
  if (stage === "breakdown") {
    return {
      ...priorArtifacts,
      breakdownDoc: await readFile(join(artifactDir, "breakdown-doc.md"), "utf-8"),
      breakdownStructured: JSON.parse(await readFile(join(artifactDir, "breakdown-doc.json"), "utf-8")) as BreakdownDoc,
    };
  }
  if (stage === "issue-generation") {
    return {
      ...priorArtifacts,
      issueSpec: await readFile(join(artifactDir, "issue-spec.md"), "utf-8"),
      issueStructured: JSON.parse(await readFile(join(artifactDir, "issue-spec.json"), "utf-8")) as IssueSpecDoc,
    };
  }
  if (stage === "implement") {
    return {
      ...priorArtifacts,
      implementationSummary: await readFile(join(artifactDir, "implementation-summary.md"), "utf-8"),
    };
  }
  if (stage === "review") {
    return {
      ...priorArtifacts,
      reviewReport: await readFile(join(artifactDir, "review-report.md"), "utf-8"),
    };
  }
  return priorArtifacts;
}

function renderStageMarkdown(report: StageReport): string {
  return [
    `# ${report.stage}`,
    "",
    `- Status: ${report.status}`,
    `- Expected workspace effect: ${report.workspaceEffect.expected}`,
    `- Workspace verdict: ${report.workspaceEffect.passed ? "pass" : "fail"}`,
    `- Changed files: ${report.changedFiles.join(", ") || "(none)"}`,
    ...(report.failureMessage ? [`- Failure: ${report.failureMessage}`] : []),
    "",
    "## Artifacts",
    "",
    ...report.artifactFiles.map((artifact) => `- ${artifact.path}: ${artifact.exists ? "present" : "missing"}${artifact.sizeBytes !== undefined ? ` (${artifact.sizeBytes} bytes)` : ""}`),
    "",
    "## Evidence",
    "",
    `- Request: ${report.requestPath}`,
    `- Stdout: ${report.stdoutPath}`,
    `- Stderr: ${report.stderrPath}`,
    `- Repo status: ${report.repoStatusPath}`,
    `- Repo diff: ${report.repoDiffPath}`,
    ...(report.eventsPath ? [`- Events: ${report.eventsPath}`] : []),
    ...(report.summary ? ["", "## Notes", "", report.summary] : []),
    "",
  ].join("\n");
}

function renderChainSummary(summary: ChainSummary): string {
  const lines = [
    "# Execute Chain Summary",
    "",
    `- Provider: ${summary.provider}`,
    `- Model: ${summary.model}`,
    `- Repo: ${summary.repo}`,
    `- Output root: ${summary.outputRoot}`,
    `- Status: ${summary.status}`,
    `- Implementation changed files: ${summary.implementationChangedFiles.join(", ") || "(none)"}`,
    `- Final changed files: ${summary.finalChangedFiles.join(", ") || "(none)"}`,
    "",
  ];

  for (const report of summary.reports) {
    lines.push(`## ${report.stage}`);
    lines.push(`- Status: ${report.status}`);
    lines.push(`- Changed files: ${report.changedFiles.join(", ") || "(none)"}`);
    lines.push(`- Workspace verdict: ${report.workspaceEffect.passed ? "pass" : "fail"}`);
    if (report.failureMessage) {
      lines.push(`- Failure: ${report.failureMessage}`);
    }
    lines.push("");
  }

  return `${lines.join("\n").trim()}\n`;
}

async function main(): Promise<void> {
  const options = parseArgs(process.argv);
  const scriptDir = dirname(fileURLToPath(import.meta.url));
  const devagentRoot = dirname(dirname(scriptDir));
  const outputRoot = options.outputRoot
    ? join(options.outputRoot)
    : await mkdtemp(join(tmpdir(), "devagent-execute-chain-"));
  await mkdir(outputRoot, { recursive: true });

  const workspaceRoot = join(outputRoot, "workspace");
  const workspaceCreation = await createIsolationWorkspaceWithTimeout({
    mode: "worktree",
    sourceRoot: SOURCE_REPO,
    targetRoot: workspaceRoot,
  }, 180_000);
  const workspace = workspaceCreation.workspace;
  await ensureGitIdentity(workspace.path);

  let priorArtifacts: ExecuteChainArtifactContext = {};
  let baselineOutputs = await captureGitOutputs(workspace.path);
  let baselineCombinedDiff = [baselineOutputs.repoDiff, baselineOutputs.repoDiffCached].filter(Boolean).join("\n");
  const reports: StageReport[] = [];
  let implementationChangedFiles: string[] = [];

  try {
    for (const stage of EXECUTE_CHAIN_STAGES) {
      process.stdout.write(`Running chained stage: ${stage}\n`);
      const stageDir = join(outputRoot, stage);
      const artifactDir = join(stageDir, "artifacts");
      await mkdir(artifactDir, { recursive: true });

      const request = buildExecuteChainRequest({
        stage,
        workspaceRoot: workspace.path,
        sourceRepoRoot: SOURCE_REPO,
        provider: options.provider,
        model: options.model,
        taskIdPrefix: "live-execute-chain",
        changedFilesHint: stage === "review" || stage === "repair"
          ? extractChangedFiles(baselineOutputs.repoStatus)
          : undefined,
        priorArtifacts,
      });
      const requestPath = join(stageDir, "request.json");
      await writeFile(requestPath, JSON.stringify(request, null, 2));

      const commandArgs = [
        join(devagentRoot, "packages", "cli", "src", "index.ts"),
        "execute",
        "--request",
        requestPath,
        "--artifact-dir",
        artifactDir,
      ];
      const commandResult = await runCommand("bun", commandArgs, workspace.path, 900_000);
      const gitOutputs = await captureGitOutputs(workspace.path);
      const combinedDiff = [gitOutputs.repoDiff, gitOutputs.repoDiffCached].filter(Boolean).join("\n");
      const eventsPath = join(artifactDir, "engine-events.jsonl");
      const outputPaths = await writeStageOutputs(
        stageDir,
        requestPath,
        commandResult,
        gitOutputs.repoStatus,
        combinedDiff,
        existsSync(eventsPath) ? eventsPath : undefined,
      );
      const artifactFiles = await collectArtifactEntries(artifactDir, stage);
      const changedFiles = extractChangedFiles(gitOutputs.repoStatus);
      const requestedWorkspaceEffect = stageExpectedWorkspaceEffect(stage);
      const effectiveWorkspaceEffect = stage === "repair" && isNoOpRepairAllowed(priorArtifacts)
        ? "non-mutating"
        : requestedWorkspaceEffect;
      const workspacePassed = effectiveWorkspaceEffect === "mutating"
        ? gitOutputs.repoStatus !== baselineOutputs.repoStatus || combinedDiff !== baselineCombinedDiff
        : gitOutputs.repoStatus === baselineOutputs.repoStatus && combinedDiff === baselineCombinedDiff;
      const stagePassed = commandResult.exitCode === 0
        && artifactFiles.every((artifact) => artifact.exists)
        && workspacePassed;
      const failureMessage = commandResult.exitCode !== 0
        ? await buildStageFailureMessage(commandResult, artifactDir)
        : !artifactFiles.every((artifact) => artifact.exists)
          ? "Missing expected stage artifacts."
          : !workspacePassed
            ? `Unexpected workspace effect for ${stage}.`
            : undefined;
      const report: StageReport = {
        stage,
        status: stagePassed ? "passed" : "failed",
        ...(failureMessage ? { failureMessage } : {}),
        requestPath,
        artifactDir,
        command: {
          executable: "bun",
          args: commandArgs,
          exitCode: commandResult.exitCode,
        },
        workspaceEffect: {
          expected: effectiveWorkspaceEffect,
          passed: workspacePassed,
          baselineStatus: baselineOutputs.repoStatus,
          finalStatus: gitOutputs.repoStatus,
        },
        changedFiles,
        artifactFiles,
        ...outputPaths,
        summary: stage === "implement"
          ? "Implementation stage is expected to edit README.md only."
          : stage === "repair"
            ? "Repair may legitimately be a no-op if the review artifact reported no defects."
            : undefined,
      };
      await writeFile(join(stageDir, "report.json"), JSON.stringify(report, null, 2));
      await writeFile(join(stageDir, "summary.md"), renderStageMarkdown(report));
      reports.push(report);
      process.stdout.write(`  -> ${report.status}\n`);
      if (!stagePassed) {
        break;
      }

      priorArtifacts = await updatePriorArtifacts(stage, artifactDir, priorArtifacts);
      if (stage === "implement") {
        implementationChangedFiles = changedFiles;
        if (!implementationChangedFiles.includes(README_TARGET)) {
          throw new Error(`Implement stage completed without modifying ${README_TARGET}.`);
        }
      }
      baselineOutputs = gitOutputs;
      baselineCombinedDiff = combinedDiff;
    }

    const finalChangedFiles = extractChangedFiles(baselineOutputs.repoStatus);
    const summary: ChainSummary = {
      provider: options.provider,
      model: options.model,
      repo: SOURCE_REPO,
      outputRoot,
      status: reports.length === EXECUTE_CHAIN_STAGES.length && reports.every((report) => report.status === "passed")
        ? "passed"
        : "failed",
      implementationChangedFiles,
      finalChangedFiles,
      reports,
    };
    await writeFile(join(outputRoot, "execute-chain-summary.json"), JSON.stringify(summary, null, 2));
    await writeFile(join(outputRoot, "execute-chain-summary.md"), renderChainSummary(summary));

    process.stdout.write(`Execute chain packet written to ${outputRoot}\n`);
    if (summary.status !== "passed") {
      process.exit(1);
    }
  } finally {
    await destroyIsolationWorkspace(workspace);
  }
}

if (import.meta.main) {
  main().catch((error) => {
    process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
    process.exit(1);
  });
}
