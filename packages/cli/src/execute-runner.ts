import { spawn } from "node:child_process";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

type WorkflowTaskType =
  | "triage"
  | "plan"
  | "implement"
  | "verify"
  | "review"
  | "repair";

type ArtifactKind =
  | "triage-report"
  | "plan"
  | "implementation-summary"
  | "verification-report"
  | "review-report"
  | "final-summary";

type ExecutorSpec = {
  executorId: "devagent" | "codex" | "claude" | "opencode";
  profileName?: string;
  provider?: string;
  model?: string;
  reasoning?: "low" | "medium" | "high";
  approvalMode?: "suggest" | "auto-edit" | "full-auto";
};

type TaskExecutionRequest = {
  protocolVersion: string;
  taskId: string;
  taskType: WorkflowTaskType;
  project: {
    id: string;
    name: string;
    repoRoot?: string;
    repoFullName?: string;
  };
  workspace: {
    sourceRepoPath: string;
    baseRef?: string;
    workBranch: string;
    isolation: "git-worktree" | "temp-copy";
    readOnly?: boolean;
  };
  executor: ExecutorSpec;
  constraints: {
    maxIterations?: number;
    timeoutSec?: number;
    allowNetwork?: boolean;
    verifyCommands?: string[];
  };
  context: {
    summary?: string;
    issueBody?: string;
    comments?: Array<{ author?: string; body: string }>;
    changedFilesHint?: string[];
    skills?: string[];
    extraInstructions?: string[];
  };
  expectedArtifacts: ArtifactKind[];
};

type ArtifactRef = {
  kind: ArtifactKind;
  path: string;
  mimeType?: string;
  createdAt: string;
};

type TaskExecutionResult = {
  protocolVersion: string;
  taskId: string;
  status: "success" | "failed" | "cancelled";
  artifacts: ArtifactRef[];
  metrics: {
    startedAt: string;
    finishedAt: string;
    durationMs: number;
  };
  error?: {
    code: string;
    message: string;
  };
};

type TaskExecutionEvent =
  | {
      protocolVersion: string;
      type: "started";
      at: string;
      taskId: string;
    }
  | {
      protocolVersion: string;
      type: "artifact";
      at: string;
      taskId: string;
      artifact: ArtifactRef;
    }
  | {
      protocolVersion: string;
      type: "completed";
      at: string;
      taskId: string;
      status: TaskExecutionResult["status"];
    };

type ExecuteArgs = {
  requestFile: string;
  artifactDir: string;
};

type WorkflowPhaseResult = {
  summary?: string;
  result?: unknown;
};

const PROTOCOL_VERSION = "0.1";

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

async function runWorkflowFallback(
  request: TaskExecutionRequest,
  requestFile: string,
  artifactDir: string,
): Promise<WorkflowPhaseResult> {
  const cliPath = fileURLToPath(new URL("./index.js", import.meta.url));
  const outputFile = join(artifactDir, "workflow-output.json");
  const args = [
    cliPath,
    "workflow",
    "run",
    "--phase",
    request.taskType,
    "--repo",
    process.cwd(),
    "--input",
    requestFile,
    "--output",
    outputFile,
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

  const request = JSON.parse(await readFile(executeArgs.requestFile, "utf-8")) as TaskExecutionRequest;
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
      if (!verification.ok) {
        const failed = createResult(request, startedAt, "failed", [], {
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
      artifact = await writeArtifact(request, executeArgs.artifactDir, verification.report);
    } else {
      const fakeResponse = process.env[fakeResponseVar(request.taskType)];
      const body = fakeResponse
        ?? (await runWorkflowFallback(request, executeArgs.requestFile, executeArgs.artifactDir)).summary
        ?? "";
      artifact = await writeArtifact(request, executeArgs.artifactDir, body);
    }

    writeEvent({
      protocolVersion: PROTOCOL_VERSION,
      type: "artifact",
      at: artifact.createdAt,
      taskId: request.taskId,
      artifact,
    });
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
