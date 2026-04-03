import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, dirname, join, resolve } from "node:path";
import { PROTOCOL_VERSION, type TaskExecutionRequest } from "@devagent-sdk/types";
import {
  captureGitOutputs,
  commitWorkspaceState,
  createIsolationWorkspace,
  destroyIsolationWorkspace,
  ensureGitIdentity,
  initializeTempCopyRepository,
} from "./isolation";
import { evaluateAssertions } from "./reporting";
import type {
  ArtifactValidationCheck,
  AuthStatusSummary,
  FailureClass,
  IsolationWorkspace,
  ObservedToolBatch,
  RunValidationScenarioOptions,
  ToolBatchAssertionResult,
  ToolCallAssertionResult,
  ValidationAssertion,
  ValidationScenario,
  ValidationScenarioReport,
  VerificationCommandResult,
} from "./types";

type CommandResult = {
  readonly exitCode: number;
  readonly stdout: string;
  readonly stderr: string;
  readonly timedOut: boolean;
};

function defaultCommand(devagentRoot: string): { executable: string; baseArgs: string[] } {
  return {
    executable: "bun",
    baseArgs: [join(devagentRoot, "packages", "cli", "src", "index.ts")],
  };
}

function renderTemplate(
  value: string,
  variables: Record<string, string>,
): string {
  return value.replace(/\$\{([^}]+)\}/g, (_, key: string) => variables[key] ?? "");
}

async function readTemplate(
  devagentRoot: string,
  templateFile: string,
): Promise<string> {
  return await readFile(
    join(devagentRoot, "scripts", "live-validation", "templates", templateFile),
    "utf-8",
  );
}

async function runCommand(
  executable: string,
  args: ReadonlyArray<string>,
  cwd: string,
  timeoutMs: number,
  env: NodeJS.ProcessEnv = process.env,
): Promise<CommandResult> {
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
      });
    });
  });
}

function classifyInvocationFailure(stderr: string, timedOut: boolean): FailureClass {
  if (timedOut) return "runtime";
  const lowered = stderr.toLowerCase();
  if (
    lowered.includes("no credentials configured") ||
    lowered.includes("run \"devagent auth login\"") ||
    lowered.includes("oauth") ||
    lowered.includes("chatgpt")
  ) {
    return "provider";
  }
  return "runtime";
}

function classifySetupFailure(message: string): FailureClass {
  const lowered = message.toLowerCase();
  if (
    lowered.includes("chatgpt auth") ||
    lowered.includes("run `devagent auth login`") ||
    lowered.includes("oauth") ||
    lowered.includes("credential")
  ) {
    return "provider";
  }
  return "setup";
}

function resolveSourceRepoRoot(
  scenario: ValidationScenario,
  options: RunValidationScenarioOptions,
): string {
  const explicit = options.sourceRepoRoots?.[scenario.targetRepo];
  if (explicit) return resolve(explicit);
  return resolve(options.devagentRoot, "..", scenario.targetRepo);
}

function resolveLinterPath(sourceRoot: string): string {
  return join(sourceRoot, "ets2panda", "linter");
}

async function loadAuthStatus(options: RunValidationScenarioOptions): Promise<AuthStatusSummary> {
  if (options.authStatusOverride) {
    return options.authStatusOverride;
  }
  const command = options.command ?? defaultCommand(options.devagentRoot);
  const result = await runCommand(
    command.executable,
    [...command.baseArgs, "auth", "status"],
    options.devagentRoot,
    30_000,
  );
  const configuredProviders = result.stderr
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .filter((line) => !line.startsWith("Credential status") && !line.startsWith("Provider"))
    .map((line) => line.split(/\s+/)[0]!)
    .filter((provider) => provider !== "(DEVAGENT_API_KEY)");
  return { configuredProviders };
}

async function applyPreSetup(
  scenario: ValidationScenario,
  workspace: IsolationWorkspace,
  variables: Record<string, string>,
  devagentRoot: string,
  linterPath: string | null,
): Promise<void> {
  for (const step of scenario.preSetup ?? []) {
    if (step.kind === "write-file") {
      const content = step.templateFile
        ? await readTemplate(devagentRoot, step.templateFile)
        : step.content ?? "";
      const rendered = renderTemplate(content, variables);
      const absolutePath = join(workspace.path, renderTemplate(step.path, variables));
      await mkdir(join(absolutePath, ".."), { recursive: true });
      await writeFile(absolutePath, rendered);
      if (step.executable) {
        await runCommand("chmod", ["+x", absolutePath], workspace.path, 10_000);
      }
      continue;
    }

    const cwd = step.cwd === "linter"
      ? (linterPath ?? workspace.path)
      : workspace.path;
    const renderedCommand = renderTemplate(step.command, variables);
    const result = await runCommand(
      process.env["SHELL"] ?? "/bin/sh",
      ["-lc", renderedCommand],
      cwd,
      scenario.timeoutMs ?? 120_000,
    );
    if (result.exitCode !== 0) {
      throw new Error(`Pre-setup command failed: ${renderedCommand}\n${result.stderr || result.stdout}`);
    }
  }
}

async function hydrateArktsLinterAssets(
  sourceRepoRoot: string,
  workspaceRoot: string,
): Promise<void> {
  const sourceLinterRoot = join(sourceRepoRoot, "ets2panda", "linter");
  const workspaceLinterRoot = join(workspaceRoot, "ets2panda", "linter");
  const assetNames = ["dist", "node_modules"];

  for (const assetName of assetNames) {
    const sourcePath = join(sourceLinterRoot, assetName);
    const targetPath = join(workspaceLinterRoot, assetName);
    if (!existsSync(sourcePath) || existsSync(targetPath)) {
      continue;
    }
    await mkdir(dirname(targetPath), { recursive: true });
    await Bun.$`ln -s ${sourcePath} ${targetPath}`.quiet();
  }
}

function upsertTomlSection(
  content: string,
  sectionName: string,
  entries: Record<string, string>,
): string {
  const lines = content.length > 0 ? content.split("\n") : [];
  const header = `[${sectionName}]`;
  const startIndex = lines.findIndex((line) => line.trim() === header);
  const renderedEntries = Object.entries(entries).map(([key, value]) => `${key} = ${value}`);

  if (startIndex === -1) {
    const trimmed = content.trimEnd();
    return `${trimmed}${trimmed.length > 0 ? "\n\n" : ""}${header}\n${renderedEntries.join("\n")}\n`;
  }

  let endIndex = lines.length;
  for (let index = startIndex + 1; index < lines.length; index++) {
    if (lines[index]!.trim().startsWith("[") && lines[index]!.trim().endsWith("]")) {
      endIndex = index;
      break;
    }
  }

  const preserved = lines.slice(startIndex + 1, endIndex).filter((line) => {
    const trimmed = line.trim();
    return !Object.keys(entries).some((key) => trimmed.startsWith(`${key} =`) || trimmed.startsWith(`${key}=`));
  });

  const replacement = [header, ...preserved, ...renderedEntries];
  return `${[...lines.slice(0, startIndex), ...replacement, ...lines.slice(endIndex)].join("\n").trimEnd()}\n`;
}

async function ensureIgnoredFile(repoRoot: string, relativePath: string): Promise<void> {
  const gitDirResult = await runCommand("git", ["rev-parse", "--git-dir"], repoRoot, 10_000);
  if (gitDirResult.exitCode !== 0) {
    throw new Error(`Failed to locate git dir for ${repoRoot}: ${gitDirResult.stderr || gitDirResult.stdout}`);
  }
  const gitDir = resolve(repoRoot, gitDirResult.stdout.trim());
  const excludePath = join(gitDir, "info", "exclude");
  const existing = existsSync(excludePath) ? await readFile(excludePath, "utf-8") : "";
  if (!existing.split("\n").includes(relativePath)) {
    await mkdir(dirname(excludePath), { recursive: true });
    await writeFile(excludePath, `${existing.trimEnd()}${existing.trimEnd().length > 0 ? "\n" : ""}${relativePath}\n`);
  }
}

async function ensureCliLoggingConfig(
  workspaceRoot: string,
  logDir: string,
): Promise<void> {
  const configPath = existsSync(join(workspaceRoot, ".devagent.toml"))
    ? join(workspaceRoot, ".devagent.toml")
    : existsSync(join(workspaceRoot, "devagent.toml"))
      ? join(workspaceRoot, "devagent.toml")
      : join(workspaceRoot, ".devagent.toml");
  const current = existsSync(configPath) ? await readFile(configPath, "utf-8") : "";
  const updated = upsertTomlSection(current, "logging", {
    enabled: "true",
    log_dir: JSON.stringify(logDir),
    retention_days: "1",
  });
  await writeFile(configPath, updated);
  await ensureIgnoredFile(workspaceRoot, configPath.endsWith("devagent.toml") && !configPath.endsWith(".devagent.toml")
    ? "devagent.toml"
    : ".devagent.toml");
}

type ToolCallObservation = {
  readonly eventsText: string;
  readonly eventsSourcePath?: string;
  readonly observedToolCalls: Record<string, number>;
  readonly observedToolBatches: Record<string, ObservedToolBatch>;
  readonly assertionResults: ToolCallAssertionResult[];
  readonly batchAssertionResults: ToolBatchAssertionResult[];
};

function countExecuteToolCalls(eventsText: string): Record<string, number> {
  const directCounts: Record<string, number> = {};
  const fallbackCounts: Record<string, number> = {};
  for (const line of eventsText.split("\n").filter(Boolean)) {
    try {
      const parsed = JSON.parse(line) as { type?: string; tool?: string; event?: string; data?: { toolCalls?: Array<{ name?: string }> } };
      if (parsed.type === "tool_call" && parsed.tool) {
        directCounts[parsed.tool] = (directCounts[parsed.tool] ?? 0) + 1;
      } else if (parsed.event === "message:assistant" && Array.isArray(parsed.data?.toolCalls)) {
        for (const toolCall of parsed.data.toolCalls) {
          if (toolCall?.name) {
            fallbackCounts[toolCall.name] = (fallbackCounts[toolCall.name] ?? 0) + 1;
          }
        }
      }
    } catch {
      // Ignore malformed lines.
    }
  }
  return Object.keys(directCounts).length > 0 ? directCounts : fallbackCounts;
}

function countCliToolCalls(eventsText: string): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const line of eventsText.split("\n").filter(Boolean)) {
    try {
      const parsed = JSON.parse(line) as { event?: string; data?: { name?: string } };
      if (parsed.event === "tool:after" && parsed.data?.name) {
        counts[parsed.data.name] = (counts[parsed.data.name] ?? 0) + 1;
      }
    } catch {
      // Ignore malformed lines.
    }
  }
  return counts;
}

function countToolBatches(eventsText: string): Record<string, ObservedToolBatch> {
  const batches: Record<string, ObservedToolBatch> = {};
  for (const line of eventsText.split("\n").filter(Boolean)) {
    try {
      const parsed = JSON.parse(line) as { event?: string; data?: { toolCalls?: Array<{ name?: string }> } };
      if (parsed.event !== "message:assistant" || !Array.isArray(parsed.data?.toolCalls)) {
        continue;
      }
      const countsForTurn = new Map<string, number>();
      for (const toolCall of parsed.data.toolCalls) {
        if (!toolCall?.name) continue;
        countsForTurn.set(toolCall.name, (countsForTurn.get(toolCall.name) ?? 0) + 1);
      }
      for (const [tool, batchSize] of countsForTurn) {
        if (batchSize < 2) continue;
        const current = batches[tool] ?? { batchCount: 0, maxBatchSize: 0 };
        batches[tool] = {
          batchCount: current.batchCount + 1,
          maxBatchSize: Math.max(current.maxBatchSize, batchSize),
        };
      }
    } catch {
      // Ignore malformed lines.
    }
  }
  return batches;
}

function evaluateToolCallRequirements(
  scenario: ValidationScenario,
  observedToolCalls: Readonly<Record<string, number>>,
): ToolCallAssertionResult[] {
  return (scenario.requiredToolCalls ?? []).map((requirement) => {
    const observedCalls = observedToolCalls[requirement.tool] ?? 0;
    const passed = observedCalls >= requirement.minCalls;
    return {
      tool: requirement.tool,
      minCalls: requirement.minCalls,
      observedCalls,
      passed,
      message: passed
        ? `Observed ${observedCalls} call(s) to ${requirement.tool}.`
        : `Expected at least ${requirement.minCalls} call(s) to ${requirement.tool}, observed ${observedCalls}.`,
    };
  });
}

function evaluateToolBatchRequirements(
  scenario: ValidationScenario,
  observedToolBatches: Readonly<Record<string, ObservedToolBatch>>,
): ToolBatchAssertionResult[] {
  return (scenario.requiredToolBatches ?? []).map((requirement) => {
    const observed = observedToolBatches[requirement.tool] ?? { batchCount: 0, maxBatchSize: 0 };
    const passed = observed.batchCount >= requirement.minBatches
      && observed.maxBatchSize >= requirement.minBatchSize;
    return {
      tool: requirement.tool,
      minBatches: requirement.minBatches,
      minBatchSize: requirement.minBatchSize,
      observedBatches: observed.batchCount,
      observedMaxBatchSize: observed.maxBatchSize,
      passed,
      message: passed
        ? `Observed ${observed.batchCount} batch(es) of ${requirement.tool} with max batch size ${observed.maxBatchSize}.`
        : `Expected at least ${requirement.minBatches} batch(es) of ${requirement.tool} with batch size >= ${requirement.minBatchSize}, observed ${observed.batchCount} batch(es) and max batch size ${observed.maxBatchSize}.`,
    };
  });
}

async function collectToolCallObservations(
  scenario: ValidationScenario,
  artifactDir: string,
  cliLogDir: string,
): Promise<ToolCallObservation> {
  if (scenario.surface === "execute") {
    const eventsSourcePath = join(artifactDir, "engine-events.jsonl");
    const eventsText = existsSync(eventsSourcePath)
      ? await readFile(eventsSourcePath, "utf-8")
      : "";
    const observedToolCalls = countExecuteToolCalls(eventsText);
    const observedToolBatches = countToolBatches(eventsText);
    return {
      eventsText,
      eventsSourcePath: existsSync(eventsSourcePath) ? eventsSourcePath : undefined,
      observedToolCalls,
      observedToolBatches,
      assertionResults: evaluateToolCallRequirements(scenario, observedToolCalls),
      batchAssertionResults: evaluateToolBatchRequirements(scenario, observedToolBatches),
    };
  }

  const entries = existsSync(cliLogDir)
    ? (await readdir(cliLogDir)).filter((entry) => entry.endsWith(".jsonl")).sort()
    : [];
  const eventsSourcePath = entries.length > 0 ? join(cliLogDir, entries[entries.length - 1]!) : undefined;
  const eventsText = eventsSourcePath ? await readFile(eventsSourcePath, "utf-8") : "";
  const observedToolCalls = countCliToolCalls(eventsText);
  const observedToolBatches = countToolBatches(eventsText);
  return {
    eventsText,
    eventsSourcePath,
    observedToolCalls,
    observedToolBatches,
    assertionResults: evaluateToolCallRequirements(scenario, observedToolCalls),
    batchAssertionResults: evaluateToolBatchRequirements(scenario, observedToolBatches),
  };
}

function artifactKindForTaskType(taskType: TaskExecutionRequest["taskType"]): TaskExecutionRequest["expectedArtifacts"][number] {
  switch (taskType) {
    case "triage":
      return "triage-report";
    case "plan":
      return "plan";
    case "implement":
      return "implementation-summary";
    case "review":
      return "review-report";
    case "repair":
      return "final-summary";
    default:
      return "plan";
  }
}

function buildExecuteRequest(
  scenario: ValidationScenario,
  workspaceRoot: string,
  sourceRepoRoot: string,
  provider: string,
  model: string,
  variables: Record<string, string>,
): TaskExecutionRequest {
  if (scenario.invocation.type !== "execute") {
    throw new Error(`Scenario ${scenario.id} is not an execute scenario.`);
  }

  const repositoryId = "repo-1";
  return {
    protocolVersion: PROTOCOL_VERSION,
    taskId: `live-${scenario.id}`,
    taskType: scenario.invocation.taskType,
    workspaceRef: {
      id: "workspace-1",
      name: basename(workspaceRoot),
      provider: "local",
      primaryRepositoryId: repositoryId,
    },
    repositories: [{
      id: repositoryId,
      workspaceId: "workspace-1",
      alias: "primary",
      name: basename(workspaceRoot),
      repoRoot: workspaceRoot,
      repoFullName: scenario.targetRepo,
      defaultBranch: "main",
      provider: "local",
    }],
    workItem: {
      id: `item-${scenario.id}`,
      kind: "local-task",
      externalId: scenario.id,
      title: renderTemplate(scenario.invocation.workItemTitle, variables),
      repositoryId,
    },
    execution: {
      primaryRepositoryId: repositoryId,
      repositories: [{
        repositoryId,
        alias: "primary",
        sourceRepoPath: sourceRepoRoot,
        workBranch: `devagent/live/${scenario.id}`,
        isolation: "temp-copy",
      }],
    },
    targetRepositoryIds: [repositoryId],
    executor: {
      executorId: "devagent",
      provider,
      model,
      approvalMode: "full-auto",
    },
    constraints: {
      allowNetwork: true,
      ...(scenario.invocation.maxIterations ? { maxIterations: scenario.invocation.maxIterations } : {}),
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
      summary: renderTemplate(scenario.invocation.summary, variables),
      ...(scenario.invocation.issueBody
        ? { issueBody: renderTemplate(scenario.invocation.issueBody, variables) }
        : {}),
      ...(scenario.invocation.extraInstructions?.length
        ? {
            extraInstructions: scenario.invocation.extraInstructions.map((entry) => renderTemplate(entry, variables)),
          }
        : {}),
    },
    expectedArtifacts: [artifactKindForTaskType(scenario.invocation.taskType)],
  };
}

async function validateExecuteArtifacts(
  scenario: ValidationScenario,
  artifactDir: string,
  stdout: string,
): Promise<{ passed: boolean; checks: ArtifactValidationCheck[]; artifactContents: Map<string, string>; cost: ValidationScenarioReport["cost"] }> {
  const checks: ArtifactValidationCheck[] = [];
  const artifactContents = new Map<string, string>();
  for (const relativePath of scenario.expectedArtifacts) {
    const fullPath = join(artifactDir, relativePath);
    const exists = existsSync(fullPath);
    checks.push({
      name: `artifact:${relativePath}`,
      passed: exists,
      message: exists ? "Artifact created." : "Artifact missing.",
    });
    if (exists) {
      artifactContents.set(relativePath, await readFile(fullPath, "utf-8"));
    }
  }

  const resultPath = join(artifactDir, "result.json");
  const resultExists = existsSync(resultPath);
  checks.push({
    name: "result.json",
    passed: resultExists,
    message: resultExists ? "Result file created." : "Result file missing.",
  });
  if (resultExists) {
    const parsed = JSON.parse(await readFile(resultPath, "utf-8")) as { status?: string; metrics?: { durationMs?: number } };
    checks.push({
      name: "result.status",
      passed: parsed.status === "success",
      message: parsed.status === "success"
        ? "Result status is success."
        : `Unexpected result status: ${String(parsed.status)}`,
    });
  }

  const eventTypes = stdout
    .trim()
    .split("\n")
    .filter(Boolean)
    .map((line) => {
      try {
        return (JSON.parse(line) as { type?: string }).type ?? "";
      } catch {
        return "";
      }
    })
    .filter(Boolean);
  checks.push({
    name: "events",
    passed: eventTypes.includes("started") && eventTypes.includes("completed"),
    message: eventTypes.length > 0
      ? `Observed events: ${eventTypes.join(", ")}`
      : "No parseable events captured.",
  });

  const engineEventsPath = join(artifactDir, "engine-events.jsonl");
  let cost: ValidationScenarioReport["cost"] = {};
  if (existsSync(engineEventsPath)) {
    const lines = (await readFile(engineEventsPath, "utf-8"))
      .split("\n")
      .filter(Boolean);
    for (const line of lines) {
      try {
        const parsed = JSON.parse(line) as { type?: string; inputTokens?: number; outputTokens?: number; totalCost?: number };
        if (parsed.type === "cost:update") {
          cost = {
            inputTokens: parsed.inputTokens,
            outputTokens: parsed.outputTokens,
            totalCost: parsed.totalCost,
          };
        }
      } catch {
        // Ignore malformed log lines.
      }
    }
  }

  return {
    passed: checks.every((check) => check.passed),
    checks,
    artifactContents,
    cost,
  };
}

async function runVerificationCommands(
  scenario: ValidationScenario,
  variables: Record<string, string>,
  workspaceRoot: string,
  linterPath: string | null,
): Promise<VerificationCommandResult[]> {
  const results: VerificationCommandResult[] = [];
  for (const step of scenario.verificationCommands) {
    const cwd = step.cwd === "linter"
      ? (linterPath ?? workspaceRoot)
      : workspaceRoot;
    const rendered = renderTemplate(step.command, variables);
    const result = await runCommand(
      process.env["SHELL"] ?? "/bin/sh",
      ["-lc", rendered],
      cwd,
      scenario.timeoutMs ?? 120_000,
    );
    results.push({
      command: rendered,
      cwd,
      exitCode: result.exitCode,
      passed: result.exitCode === 0,
      stdout: result.stdout,
      stderr: result.stderr,
    });
    if (result.exitCode !== 0) {
      break;
    }
  }
  return results;
}

async function writeRawOutputs(
  outputDir: string,
  stdout: string,
  stderr: string,
  repoStatus: string,
  repoDiff: string,
  events: string,
): Promise<ValidationScenarioReport["rawOutputs"]> {
  const stdoutPath = join(outputDir, "stdout.txt");
  const stderrPath = join(outputDir, "stderr.txt");
  const repoStatusPath = join(outputDir, "repo-status.txt");
  const repoDiffPath = join(outputDir, "repo-diff.txt");
  const eventsPath = join(outputDir, "events.txt");
  await Promise.all([
    writeFile(stdoutPath, stdout),
    writeFile(stderrPath, stderr),
    writeFile(repoStatusPath, repoStatus),
    writeFile(repoDiffPath, repoDiff),
    writeFile(eventsPath, events),
  ]);
  return {
    stdoutPath,
    stderrPath,
    repoStatusPath,
    repoDiffPath,
    eventsPath,
  };
}

function buildVariables(
  scenario: ValidationScenario,
  sourceRepoRoot: string,
  workspaceRoot: string,
  outputDir: string,
  artifactDir: string,
  linterPath: string | null,
): Record<string, string> {
  return {
    sourceRepoRoot,
    repoRoot: workspaceRoot,
    outputDir,
    artifactDir,
    ...(linterPath ? { linterPath } : {}),
    ...(scenario.variables ? scenario.variables : {}),
  };
}

function buildCommandEnv(
  scenario: ValidationScenario,
  variables: Record<string, string>,
): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = { ...process.env };
  for (const [key, value] of Object.entries(scenario.commandEnv ?? {})) {
    env[key] = renderTemplate(value, variables);
  }
  return env;
}

function buildCliArgs(
  scenario: ValidationScenario,
  provider: string,
  model: string,
  variables: Record<string, string>,
): string[] {
  if (scenario.invocation.type === "cli-command") {
    return scenario.invocation.args.map((entry) => renderTemplate(entry, variables));
  }
  if (scenario.invocation.type !== "cli") {
    throw new Error(`Scenario ${scenario.id} is not a CLI scenario.`);
  }
  const args: string[] = [
    "--provider",
    provider,
    "--model",
    model,
    scenario.invocation.approvalMode === "suggest"
      ? "--suggest"
      : scenario.invocation.approvalMode === "auto-edit"
        ? "--auto-edit"
        : "--full-auto",
  ];
  if (scenario.invocation.maxIterations) {
    args.push("--max-iterations", String(scenario.invocation.maxIterations));
  }
  if (scenario.invocation.reasoning) {
    args.push("--reasoning", scenario.invocation.reasoning);
  }
  if (scenario.invocation.extraArgs) {
    args.push(...scenario.invocation.extraArgs.map((entry) => renderTemplate(entry, variables)));
  }
  args.push(renderTemplate(scenario.invocation.query, variables));
  return args;
}

function renderAssertions(
  assertions: ReadonlyArray<ValidationAssertion>,
  variables: Record<string, string>,
): ValidationAssertion[] {
  return assertions.map((assertion) => {
    if (assertion.type === "contains") {
      return {
        ...assertion,
        ...(assertion.path ? { path: renderTemplate(assertion.path, variables) } : {}),
        value: renderTemplate(assertion.value, variables),
      };
    }
    return {
      ...assertion,
      ...(assertion.path ? { path: renderTemplate(assertion.path, variables) } : {}),
      pattern: renderTemplate(assertion.pattern, variables),
    };
  });
}

async function createFailureReport(
  scenario: ValidationScenario,
  options: RunValidationScenarioOptions,
  sourceRepoPath: string,
  outputDir: string,
  failureClass: FailureClass,
  failureMessage: string,
  isolationPath = "",
): Promise<ValidationScenarioReport> {
  const now = new Date().toISOString();
  const report: ValidationScenarioReport = {
    scenarioId: scenario.id,
    description: scenario.description,
    targetRepo: scenario.targetRepo,
    surface: scenario.surface,
    taskShape: scenario.taskShape,
    provider: options.provider,
    model: options.model,
    status: "failed",
    failureClass,
    failureMessage,
    startedAt: now,
    finishedAt: now,
    durationMs: 0,
    sourceRepoPath,
    isolationPath,
    outputDir,
    command: {
      executable: options.command?.executable ?? defaultCommand(options.devagentRoot).executable,
      args: options.command?.baseArgs ?? defaultCommand(options.devagentRoot).baseArgs,
      exitCode: 1,
    },
    artifactValidation: { passed: false, checks: [] },
    assertionResults: [],
    toolCallAssertionResults: [],
    toolBatchAssertionResults: [],
    verificationResults: [],
    timing: { durationMs: 0 },
    cost: {},
    rawOutputs: {},
  };
  await mkdir(outputDir, { recursive: true });
  await writeFile(join(outputDir, "report.json"), JSON.stringify(report, null, 2));
  return report;
}

export async function runValidationScenario(
  scenario: ValidationScenario,
  options: RunValidationScenarioOptions,
): Promise<ValidationScenarioReport> {
  const startedAt = new Date().toISOString();
  const sourceRepoRoot = resolveSourceRepoRoot(scenario, options);
  const outputDir = join(options.outputRoot, scenario.id);
  await mkdir(outputDir, { recursive: true });

  if (!existsSync(sourceRepoRoot)) {
    return await createFailureReport(
      scenario,
      options,
      sourceRepoRoot,
      outputDir,
      "setup",
      `Missing source repository: ${sourceRepoRoot}`,
    );
  }

  const linterPath = scenario.requiresArktsLinter
    ? resolveLinterPath(sourceRepoRoot)
    : null;
  const workspaceRoot = join(outputDir, "workspace");
  const artifactDir = join(outputDir, "artifacts");
  const cliLogDir = join(outputDir, "cli-logs");
  let workspace: IsolationWorkspace | null = null;
  try {
    if (scenario.requiresChatgptAuth ?? true) {
      const authStatus = await loadAuthStatus(options);
      if (!authStatus.configuredProviders.includes("chatgpt")) {
        return await createFailureReport(
          scenario,
          options,
          sourceRepoRoot,
          outputDir,
          "provider",
          "ChatGPT auth is not configured. Run `devagent auth login` first.",
        );
      }
    }
    if (scenario.requiresArktsLinter && (!linterPath || !existsSync(join(linterPath, "dist", "tslinter.js")))) {
      return await createFailureReport(
        scenario,
        options,
        sourceRepoRoot,
        outputDir,
        "setup",
        `ArkTS linter is unavailable at ${String(linterPath)}.`,
      );
    }

    workspace = await createIsolationWorkspace({
      mode: scenario.isolationMode,
      sourceRoot: sourceRepoRoot,
      targetRoot: workspaceRoot,
    });

    if (workspace.mode === "temp-copy") {
      await initializeTempCopyRepository(workspace.path);
      await commitWorkspaceState(workspace.path, "live validation baseline");
    } else {
      await ensureGitIdentity(workspace.path);
    }

    const variables = buildVariables(
      scenario,
      sourceRepoRoot,
      workspace.path,
      outputDir,
      artifactDir,
      linterPath,
    );
    if (scenario.surface === "execute" && scenario.requiresArktsLinter && linterPath) {
      await hydrateArktsLinterAssets(sourceRepoRoot, workspace.path);
    }
    await applyPreSetup(scenario, workspace, variables, options.devagentRoot, linterPath);
    if (scenario.surface === "cli") {
      await ensureCliLoggingConfig(workspace.path, cliLogDir);
    }
    if (scenario.baselineAfterSetup) {
      await commitWorkspaceState(workspace.path, "seed live validation scenario");
    }

    const command = options.command ?? defaultCommand(options.devagentRoot);
    const commandEnv = buildCommandEnv(scenario, variables);
    let invocationResult: CommandResult;
    let executedArgs: string[];

    if (scenario.invocation.type === "execute") {
      await mkdir(artifactDir, { recursive: true });
      const request = buildExecuteRequest(
        scenario,
        workspace.path,
        sourceRepoRoot,
        options.provider,
        options.model,
        variables,
      );
      const requestPath = join(outputDir, "request.json");
      await writeFile(requestPath, JSON.stringify(request, null, 2));
      executedArgs = [...command.baseArgs, "execute", "--request", requestPath, "--artifact-dir", artifactDir];
      invocationResult = await runCommand(
        command.executable,
        executedArgs,
        workspace.path,
        scenario.timeoutMs ?? 600_000,
        commandEnv,
      );
    } else {
      executedArgs = [...command.baseArgs, ...buildCliArgs(scenario, options.provider, options.model, variables)];
      invocationResult = await runCommand(
        command.executable,
        executedArgs,
        workspace.path,
        scenario.timeoutMs ?? 600_000,
        commandEnv,
      );
    }

    const gitOutputs = await captureGitOutputs(workspace.path);
    const combinedDiff = [gitOutputs.repoDiff, gitOutputs.repoDiffCached].filter(Boolean).join("\n");
    const toolCallObservation = await collectToolCallObservations(scenario, artifactDir, cliLogDir);
    const rawOutputs = await writeRawOutputs(
      outputDir,
      invocationResult.stdout,
      invocationResult.stderr,
      gitOutputs.repoStatus,
      combinedDiff,
      toolCallObservation.eventsText,
    );

    const expectedExitCode = scenario.expectedExitCode ?? 0;
    if (invocationResult.exitCode !== expectedExitCode) {
      const failureClass = classifyInvocationFailure(invocationResult.stderr, invocationResult.timedOut);
      const finishedAt = new Date().toISOString();
      const report: ValidationScenarioReport = {
        scenarioId: scenario.id,
        description: scenario.description,
        targetRepo: scenario.targetRepo,
        surface: scenario.surface,
        taskShape: scenario.taskShape,
        provider: options.provider,
        model: options.model,
        status: "failed",
        failureClass,
        failureMessage: invocationResult.stderr.trim() || invocationResult.stdout.trim() || `Scenario command exited with ${invocationResult.exitCode}, expected ${expectedExitCode}.`,
        startedAt,
        finishedAt,
        durationMs: Date.now() - new Date(startedAt).getTime(),
        sourceRepoPath: sourceRepoRoot,
        isolationPath: workspace.path,
        outputDir,
        command: {
          executable: command.executable,
          args: executedArgs,
          exitCode: invocationResult.exitCode,
        },
        artifactValidation: { passed: false, checks: [] },
        assertionResults: [],
        toolCallAssertionResults: toolCallObservation.assertionResults,
        toolBatchAssertionResults: toolCallObservation.batchAssertionResults,
        observedToolCalls: toolCallObservation.observedToolCalls,
        observedToolBatches: toolCallObservation.observedToolBatches,
        ...(toolCallObservation.eventsSourcePath ? { eventsSourcePath: toolCallObservation.eventsSourcePath } : {}),
        verificationResults: [],
        timing: { durationMs: Date.now() - new Date(startedAt).getTime() },
        cost: {},
        rawOutputs,
      };
      await writeFile(join(outputDir, "report.json"), JSON.stringify(report, null, 2));
      return report;
    }

    const executeValidation = scenario.surface === "execute"
      ? await validateExecuteArtifacts(scenario, artifactDir, invocationResult.stdout)
      : { passed: true, checks: [] as ArtifactValidationCheck[], artifactContents: new Map<string, string>(), cost: {} };
    const assertionEvaluation = evaluateAssertions(renderAssertions(scenario.assertions, variables), {
      stdout: invocationResult.stdout,
      stderr: invocationResult.stderr,
      repoDiff: combinedDiff,
      repoStatus: gitOutputs.repoStatus,
      events: toolCallObservation.eventsText,
      artifacts: executeValidation.artifactContents,
    });
    const verificationResults = await runVerificationCommands(
      scenario,
      variables,
      workspace.path,
      linterPath,
    );
    const verificationPassed = verificationResults.every((entry) => entry.passed);
    const toolCallAssertionsPassed = toolCallObservation.assertionResults.every((entry) => entry.passed);
    const toolBatchAssertionsPassed = toolCallObservation.batchAssertionResults.every((entry) => entry.passed);
    const finishedAt = new Date().toISOString();
    const durationMs = Date.now() - new Date(startedAt).getTime();
    const status = executeValidation.passed
      && assertionEvaluation.passed
      && toolCallAssertionsPassed
      && toolBatchAssertionsPassed
      && verificationPassed
      ? "passed"
      : "failed";
    const failureClass = status === "passed"
      ? undefined
      : !executeValidation.passed || !assertionEvaluation.passed || !toolCallAssertionsPassed || !toolBatchAssertionsPassed
        ? "assertion"
        : "verification";
    const failureMessage = status === "passed"
      ? undefined
      : !executeValidation.passed
        ? "Artifact validation failed."
        : !assertionEvaluation.passed || !toolCallAssertionsPassed || !toolBatchAssertionsPassed
          ? "One or more assertions failed."
          : "Verification command failed.";

    const report: ValidationScenarioReport = {
      scenarioId: scenario.id,
      description: scenario.description,
      targetRepo: scenario.targetRepo,
      surface: scenario.surface,
      taskShape: scenario.taskShape,
      provider: options.provider,
      model: options.model,
      status,
      ...(failureClass ? { failureClass } : {}),
      ...(failureMessage ? { failureMessage } : {}),
      startedAt,
      finishedAt,
      durationMs,
      sourceRepoPath: sourceRepoRoot,
      isolationPath: workspace.path,
      outputDir,
      command: {
        executable: command.executable,
        args: executedArgs,
        exitCode: invocationResult.exitCode,
      },
      artifactValidation: {
        passed: executeValidation.passed,
        checks: executeValidation.checks,
      },
      assertionResults: assertionEvaluation.results,
      toolCallAssertionResults: toolCallObservation.assertionResults,
      toolBatchAssertionResults: toolCallObservation.batchAssertionResults,
      observedToolCalls: toolCallObservation.observedToolCalls,
      observedToolBatches: toolCallObservation.observedToolBatches,
      ...(toolCallObservation.eventsSourcePath ? { eventsSourcePath: toolCallObservation.eventsSourcePath } : {}),
      verificationResults,
      timing: { durationMs },
      cost: executeValidation.cost,
      rawOutputs,
    };
    await writeFile(join(outputDir, "report.json"), JSON.stringify(report, null, 2));
    return report;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return await createFailureReport(
      scenario,
      options,
      sourceRepoRoot,
      outputDir,
      classifySetupFailure(message),
      message,
      workspace?.path ?? "",
    );
  } finally {
    if (workspace && scenario.cleanupPolicy === "destroy") {
      await destroyIsolationWorkspace(workspace);
    }
  }
}
