import { spawn } from "node:child_process";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { validateTaskExecutionRequest } from "@devagent-sdk/validation";
import {
  PROTOCOL_VERSION,
  type ArtifactKind,
  type ArtifactRef,
  type TaskExecutionEvent,
  type TaskExecutionRequest,
  type TaskExecutionResult,
  type WorkflowTaskType,
} from "@devagent-sdk/types";

type ExecuteArgs = {
  requestFile: string;
  artifactDir: string;
};

type WorkflowPhaseResult = {
  summary?: string;
  result?: unknown;
  // Review-specific fields emitted by the review phase prompt
  verdict?: "pass" | "block";
  findings?: Array<{ file?: string; line?: number; severity?: string; message?: string; category?: string }>;
  blockingCount?: number;
};

function extractInstructionBlock(instructions: string[] | undefined, label: string): string | undefined {
  const prefix = `${label}:\n`;
  const entry = instructions?.find((instruction) => instruction.startsWith(prefix));
  if (!entry) {
    return undefined;
  }
  return entry.slice(prefix.length).trim();
}

function parseExecuteArgs(argv: string[]): ExecuteArgs | null {
  const args = argv.slice(2);
  if (args[0] !== "execute") {
    return null;
  }

  let requestFile: string | undefined;
  let artifactDir: string | undefined;
  for (let index = 1; index < args.length; index += 1) {
    const arg = args[index]!;
    if (arg === "--request" && index + 1 < args.length) {
      requestFile = args[++index]!;
    } else if (arg === "--artifact-dir" && index + 1 < args.length) {
      artifactDir = args[++index]!;
    }
  }

  if (!requestFile || !artifactDir) {
    return null;
  }
  return { requestFile, artifactDir };
}

function artifactKindForTask(taskType: WorkflowTaskType): ArtifactKind {
  switch (taskType) {
    case "triage":
      return "triage-report";
    case "plan":
      return "plan";
    case "implement":
      return "implementation-summary";
    case "verify":
      return "verification-report";
    case "review":
      return "review-report";
    case "repair":
      return "final-summary";
  }
}

function artifactFileName(kind: ArtifactKind): string {
  switch (kind) {
    case "triage-report":
      return "triage-report.md";
    case "plan":
      return "plan.md";
    case "implementation-summary":
      return "implementation-summary.md";
    case "verification-report":
      return "verification-report.md";
    case "review-report":
      return "review-report.md";
    case "final-summary":
      return "final-summary.md";
  }
}

function fakeResponseVar(taskType: WorkflowTaskType): string {
  return `DEVAGENT_EXECUTOR_FAKE_RESPONSE_${taskType.toUpperCase()}`;
}

function writeEvent(event: TaskExecutionEvent): void {
  process.stdout.write(`${JSON.stringify(event)}\n`);
}

function createResult(
  request: TaskExecutionRequest,
  startedAt: string,
  status: TaskExecutionResult["status"],
  artifacts: ArtifactRef[],
  error?: TaskExecutionResult["error"],
): TaskExecutionResult {
  const finishedAt = new Date().toISOString();
  return {
    protocolVersion: PROTOCOL_VERSION,
    taskId: request.taskId,
    status,
    artifacts,
    metrics: {
      startedAt,
      finishedAt,
      durationMs: Date.now() - new Date(startedAt).getTime(),
    },
    ...(error ? { error } : {}),
  };
}

async function writeResult(artifactDir: string, result: TaskExecutionResult): Promise<void> {
  await writeFile(join(artifactDir, "result.json"), JSON.stringify(result, null, 2));
}

async function writeArtifact(
  request: TaskExecutionRequest,
  artifactDir: string,
  body: string,
): Promise<ArtifactRef> {
  const kind = request.expectedArtifacts[0] ?? artifactKindForTask(request.taskType);
  const path = join(artifactDir, artifactFileName(kind));
  await writeFile(path, body.endsWith("\n") ? body : `${body}\n`);
  return {
    kind,
    path,
    mimeType: "text/markdown",
    createdAt: new Date().toISOString(),
  };
}

async function runCommand(command: string): Promise<{ exitCode: number; stdout: string; stderr: string }> {
  return await new Promise((resolvePromise) => {
    const child = spawn(process.env["SHELL"] ?? "sh", ["-lc", command], {
      cwd: process.cwd(),
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk: Buffer) => {
      stdout += chunk.toString();
    });
    child.stderr.on("data", (chunk: Buffer) => {
      stderr += chunk.toString();
    });
    child.once("close", (code) => {
      resolvePromise({
        exitCode: code ?? 1,
        stdout,
        stderr,
      });
    });
  });
}

async function runVerifyCommands(commands: string[]): Promise<{ ok: boolean; report: string }> {
  const sections: string[] = [];
  let ok = true;
  for (const command of commands) {
    const result = await runCommand(command);
    const block = [
      `## ${command}`,
      "",
      `Exit code: ${result.exitCode}`,
      "",
      "### Stdout",
      "```text",
      result.stdout.trimEnd() || "(empty)",
      "```",
      "",
      "### Stderr",
      "```text",
      result.stderr.trimEnd() || "(empty)",
      "```",
    ];
    sections.push(block.join("\n"));
    if (result.exitCode !== 0) {
      ok = false;
      break;
    }
  }

  sections.push("", `Overall result: ${ok ? "pass" : "fail"}`);
  return {
    ok,
    report: sections.join("\n"),
  };
}

/**
 * Transform a TaskExecutionRequest into the simplified input format
 * expected by the workflow runner's validateInput / buildPhasePrompt.
 *
 * Each phase has its own required fields:
 *   triage/plan: issueNumber, title
 *   implement:   issueNumber, acceptedPlan
 *   verify:      commands[]
 *   review:      issueNumber
 *   repair:      issueNumber, round
 */
export function buildWorkflowInput(request: TaskExecutionRequest): Record<string, unknown> {
  const base: Record<string, unknown> = {
    issueNumber: request.workItem.externalId,
    title: request.workItem.title ?? "",
    body: request.context.issueBody ?? request.context.summary ?? "",
    author: "",
    labels: [],
  };

  // Forward context fields if available
  if (request.context.comments) {
    base.comments = request.context.comments;
  }
  if (request.context.changedFilesHint) {
    base.changedFilesHint = request.context.changedFilesHint;
  }

  // Phase-specific fields
  switch (request.taskType) {
    case "implement": {
      const acceptedPlan = extractInstructionBlock(request.context.extraInstructions, "Accepted plan");
      if (acceptedPlan) {
        base.acceptedPlan = acceptedPlan;
      } else if (request.context.summary && !/^implement for issue #.+$/i.test(request.context.summary.trim())) {
        base.acceptedPlan = request.context.summary;
      }
      break;
    }
    case "verify":
      base.commands = request.constraints.verifyCommands ?? [];
      break;
    case "review":
      if (request.context.extraInstructions?.length) {
        base.reviewComments = request.context.extraInstructions;
      }
      break;
    case "repair":
      base.round = 1;
      base.findings = request.context.extraInstructions ?? [];
      break;
  }

  return base;
}

/**
 * Format a workflow phase result into a markdown report suitable as an artifact.
 *
 * For review phases, this produces a structured report that includes the hub's
 * expected sentinel ("No defects found.") when the verdict is "pass", so the
 * hub's `reviewRequiresRepair` check works correctly.
 */
export function formatPhaseReport(taskType: WorkflowTaskType, result: WorkflowPhaseResult): string {
  if (taskType === "review") {
    const hasFindings = (result.findings?.length ?? 0) > 0;
    const passesReview = result.verdict !== "block"
      && !hasFindings
      && (result.verdict === "pass" || result.blockingCount === 0);
    const sections: string[] = [];
    sections.push(`# Review Report`);
    sections.push("");
    sections.push(result.summary ?? "No summary provided.");
    sections.push("");
    if (passesReview) {
      sections.push("No defects found.");
    } else {
      sections.push(`**Verdict**: ${result.verdict ?? "block"}`);
      sections.push(`**Blocking findings**: ${result.blockingCount ?? 0}`);
    }
    if (result.findings && result.findings.length > 0) {
      sections.push("");
      sections.push("## Findings");
      for (const f of result.findings) {
        const loc = [f.file, f.line != null ? `:${f.line}` : ""].filter(Boolean).join("");
        sections.push(`- **[${f.severity ?? "info"}]** ${loc ? `\`${loc}\` ` : ""}${f.message ?? ""} *(${f.category ?? "general"})*`);
      }
    }
    return sections.join("\n");
  }
  // Default: use summary text
  return result.summary ?? "";
}

async function runWorkflowFallback(
  request: TaskExecutionRequest,
  _requestFile: string,
  artifactDir: string,
): Promise<WorkflowPhaseResult> {
  const cliPath = fileURLToPath(new URL("./index.js", import.meta.url));
  const outputFile = join(artifactDir, "workflow-output.json");
  const eventsFile = join(artifactDir, "workflow-events.jsonl");

  // Write a transformed input file for the workflow runner
  const workflowInput = buildWorkflowInput(request);
  const workflowInputFile = join(artifactDir, "workflow-input.json");
  await writeFile(workflowInputFile, JSON.stringify(workflowInput, null, 2));

  const args = [
    cliPath,
    "workflow",
    "run",
    "--phase",
    request.taskType,
    "--repo",
    process.cwd(),
    "--input",
    workflowInputFile,
    "--output",
    outputFile,
    "--events",
    eventsFile,
  ];

  if (request.executor.provider) {
    args.push("--provider", request.executor.provider);
  }
  if (request.executor.model) {
    args.push("--model", request.executor.model);
  }
  if (request.constraints.maxIterations) {
    args.push("--max-iterations", String(request.constraints.maxIterations));
  }
  if (request.executor.approvalMode) {
    args.push("--approval", request.executor.approvalMode);
  }

  const { code, stderr } = await new Promise<{ code: number; stderr: string }>((resolvePromise) => {
    const child = spawn(process.execPath, args, {
      cwd: process.cwd(),
      stdio: ["ignore", "ignore", "pipe"],
    });
    let stderr = "";
    child.stderr.on("data", (chunk: Buffer) => {
      stderr += chunk.toString();
    });
    child.once("close", (code) => {
      resolvePromise({ code: code ?? 1, stderr });
    });
  });

  if (code !== 0) {
    throw new Error(stderr.trim() || `Workflow phase ${request.taskType} failed`);
  }
  if (!existsSync(outputFile)) {
    throw new Error(`Workflow phase ${request.taskType} did not produce ${outputFile}`);
  }

  return JSON.parse(await readFile(outputFile, "utf-8")) as WorkflowPhaseResult;
}

export async function handleExecuteCommand(argv: string[]): Promise<void> {
  const executeArgs = parseExecuteArgs(argv);
  if (!executeArgs) {
    process.stderr.write("Usage: devagent execute --request <file> --artifact-dir <dir>\n");
    process.exit(1);
  }

  const request = validateTaskExecutionRequest(JSON.parse(await readFile(executeArgs.requestFile, "utf-8")));
  const startedAt = new Date().toISOString();
  await mkdir(executeArgs.artifactDir, { recursive: true });
  writeEvent({
    protocolVersion: PROTOCOL_VERSION,
    type: "started",
    at: startedAt,
    taskId: request.taskId,
  });

  try {
    if (request.constraints.allowNetwork === false) {
      throw new Error("Network-disabled execution is not supported by the devagent executor.");
    }

    let artifact: ArtifactRef | undefined;

    if (request.taskType === "verify") {
      const commands = request.constraints.verifyCommands ?? [];
      const verification = await runVerifyCommands(commands);
      artifact = await writeArtifact(request, executeArgs.artifactDir, verification.report);
      writeEvent({
        protocolVersion: PROTOCOL_VERSION,
        type: "artifact",
        at: artifact.createdAt,
        taskId: request.taskId,
        artifact,
      });
      if (!verification.ok) {
        const failed = createResult(request, startedAt, "failed", [artifact], {
          code: "EXECUTION_FAILED",
          message: "One or more verification commands failed.",
        });
        await writeResult(executeArgs.artifactDir, failed);
        writeEvent({
          protocolVersion: PROTOCOL_VERSION,
          type: "completed",
          at: failed.metrics.finishedAt,
          taskId: request.taskId,
          status: failed.status,
        });
        process.exit(1);
      }
    } else {
      const fakeResponse = process.env[fakeResponseVar(request.taskType)];
      if (fakeResponse) {
        artifact = await writeArtifact(request, executeArgs.artifactDir, fakeResponse);
      } else {
        const phaseResult = await runWorkflowFallback(request, executeArgs.requestFile, executeArgs.artifactDir);
        const body = formatPhaseReport(request.taskType, phaseResult);
        artifact = await writeArtifact(request, executeArgs.artifactDir, body);
      }
    }

    if (request.taskType !== "verify") {
      writeEvent({
        protocolVersion: PROTOCOL_VERSION,
        type: "artifact",
        at: artifact.createdAt,
        taskId: request.taskId,
        artifact,
      });
    }
    const success = createResult(request, startedAt, "success", [artifact]);
    await writeResult(executeArgs.artifactDir, success);
    writeEvent({
      protocolVersion: PROTOCOL_VERSION,
      type: "completed",
      at: success.metrics.finishedAt,
      taskId: request.taskId,
      status: success.status,
    });
  } catch (error) {
    const failed = createResult(request, startedAt, "failed", [], {
      code: "EXECUTION_FAILED",
      message: error instanceof Error ? error.message : String(error),
    });
    await writeResult(executeArgs.artifactDir, failed);
    writeEvent({
      protocolVersion: PROTOCOL_VERSION,
      type: "completed",
      at: failed.metrics.finishedAt,
      taskId: request.taskId,
      status: failed.status,
    });
    process.exit(1);
  }
}
