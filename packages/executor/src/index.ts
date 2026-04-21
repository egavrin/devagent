import {
  SkillLoader,
  SkillRegistry,
  SkillResolver,
  extractErrorMessage,
} from "@devagent/runtime";
import {
  PROTOCOL_VERSION,
  type ArtifactKind,
  type ArtifactRef,
  type BreakdownDoc,
  type ContinuationSession,
  type IssueSpecDoc,
  type TaskExecutionEvent,
  type TaskExecutionRequest,
  type TaskExecutionResult,
} from "@devagent-sdk/types";
import {
  validateBreakdownDoc,
  validateIssueSpecDoc,
} from "@devagent-sdk/validation";
import { exec, type ExecException } from "node:child_process";
import { mkdir, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";

import { buildTaskQuery } from "./query.js";
import type { ResolvedRequestedSkill } from "./query.js";
import {
  classifyFailedWorkflowResult,
  classifySuccessArtifactOutcome,
  readFakeTaskResponse,
  resolveVerifyNodeBinary,
  rewriteVerifyCommand,
} from "./verify-helpers.js";
import type {
  PendingArtifact,
  StructuredArtifactEnvelope,
  TaskLoopEnvelope,
  VerifyCommandRun,
} from "./verify-helpers.js";

export { buildTaskQuery };
export {
  loadTaskExecutionRequest,
  parseExecuteArgs,
  readFakeTaskResponse,
  resolveVerifyNodeBinary,
  rewriteVerifyCommand,
} from "./verify-helpers.js";

export type { ResolvedRequestedSkill };

export interface WorkflowQueryResult {
  success: boolean;
  responseText: string;
  iterations: number;
  session?: ContinuationSession;
  outcome?: TaskExecutionResult["outcome"];
  outcomeReason?: TaskExecutionResult["outcomeReason"];
}

export interface ExecuteTaskOptions {
  request: TaskExecutionRequest;
  artifactDir: string;
  repoRoot: string;
  runQuery: (options: {
    query: string;
    taskType: TaskExecutionRequest["taskType"];
    repoPath: string;
    provider?: string;
    model?: string;
    maxIterations?: number;
    approvalMode: string;
    reasoning?: string;
    eventsPath: string;
    requestedSkills?: string[];
    continuation?: TaskExecutionRequest["continuation"];
  }) => Promise<WorkflowQueryResult>;
  emit: (event: TaskExecutionEvent) => void;
}

interface TaskRunState {
  artifactResponseText: string;
  outcomeContent: string;
  status: TaskExecutionResult["status"];
  error?: TaskExecutionResult["error"];
  session?: ContinuationSession;
  outcome?: TaskExecutionResult["outcome"];
  outcomeReason?: TaskExecutionResult["outcomeReason"];
}

interface FinalResultOptions {
  readonly request: TaskExecutionRequest;
  readonly artifactDir: string;
  readonly startedAt: string;
  readonly state: TaskRunState;
  readonly artifacts: ArtifactRef[];
  readonly emit: ExecuteTaskOptions["emit"];
}

export function validateExecutionCapabilities(request: TaskExecutionRequest): void {
  if (request.executor.executorId !== "devagent") {
    throw new Error(`Unsupported executor for devagent execute: ${request.executor.executorId}`);
  }
  if (
    request.constraints.maxIterations !== undefined &&
    (!Number.isInteger(request.constraints.maxIterations) || request.constraints.maxIterations < 1)
  ) {
    throw new Error("Unsupported maxIterations: expected integer >= 1");
  }
  if (request.constraints.allowNetwork === false) {
    throw new Error("Unsupported constraint: allowNetwork=false");
  }
}

export async function resolveRequestedSkills(
  repoRoot: string,
  requestedSkills: string[] | undefined,
  sessionId: string,
  onWarning?: (message: string) => void,
): Promise<ResolvedRequestedSkill[]> {
  if (!requestedSkills?.length) {
    return [];
  }

  const skillLoader = new SkillLoader();
  const registry = new SkillRegistry();
  registry.register(skillLoader.discover({ repoRoot }));
  const resolver = new SkillResolver();

  const resolved: ResolvedRequestedSkill[] = [];
  for (const skillName of requestedSkills) {
    try {
      const skill = await registry.load(skillName);
      const loaded = await resolver.resolve(skill, "", {
        sessionId,
        allowShellPreprocess: skill.source === "project",
      });
      resolved.push({
        name: loaded.name,
        description: loaded.description,
        source: loaded.source,
        instructions: loaded.resolvedInstructions,
      });
    } catch (error) {
      const message = `Requested skill "${skillName}" could not be loaded and will be skipped: ${extractErrorMessage(error)}`;
      onWarning?.(message);
    }
  }

  return resolved;
}

type ArtifactInfo = {
  kind: ArtifactKind;
  fileName: string;
};

const ARTIFACT_INFO_BY_TASK: Record<TaskExecutionRequest["taskType"], ArtifactInfo> = {
  "task-intake": { kind: "task-spec", fileName: "task-spec.md" },
  design: { kind: "design-doc", fileName: "design-doc.md" },
  breakdown: { kind: "breakdown-doc", fileName: "breakdown-doc.md" },
  "issue-generation": { kind: "issue-spec", fileName: "issue-spec.md" },
  triage: { kind: "triage-report", fileName: "triage-report.md" },
  plan: { kind: "plan", fileName: "plan.md" },
  "test-plan": { kind: "test-plan", fileName: "test-plan.md" },
  implement: { kind: "implementation-summary", fileName: "implementation-summary.md" },
  verify: { kind: "verification-report", fileName: "verification-report.md" },
  review: { kind: "review-report", fileName: "review-report.md" },
  repair: { kind: "final-summary", fileName: "final-summary.md" },
  completion: { kind: "workflow-summary", fileName: "workflow-summary.md" },
};

export function artifactInfoForTask(taskType: TaskExecutionRequest["taskType"]): ArtifactInfo {
  return ARTIFACT_INFO_BY_TASK[taskType];
}

function isStrictStructuredArtifactTask(taskType: TaskExecutionRequest["taskType"]): boolean {
  return taskType === "breakdown" || taskType === "issue-generation";
}

function extractJsonCandidate(text: string): string {
  const trimmed = text.trim();
  const fencedMatch = trimmed.match(/^```(?:json)?\s*([\s\S]*?)\s*```$/i);
  if (fencedMatch?.[1]) {
    return fencedMatch[1].trim();
  }
  const wrappedJsonMatch = trimmed.match(/^# [^\n]+\n\n(\{[\s\S]*\})$/);
  if (wrappedJsonMatch?.[1]) {
    return wrappedJsonMatch[1].trim();
  }
  return trimmed;
}

function parseStructuredArtifactEnvelope(taskType: TaskExecutionRequest["taskType"], responseText: string): {
  structured: BreakdownDoc | IssueSpecDoc;
  rendered: string;
} {
  const candidate = extractJsonCandidate(extractArtifactBody(responseText));
  let parsed: StructuredArtifactEnvelope;
  try {
    parsed = JSON.parse(candidate) as StructuredArtifactEnvelope;
  } catch (error) {
    throw new Error(`Expected ${taskType} to return strict JSON with structured and rendered fields: ${extractErrorMessage(error)}`);
  }

  if (!parsed || typeof parsed !== "object" || typeof parsed.rendered !== "string" || !("structured" in parsed)) {
    throw new Error(`Expected ${taskType} to return a JSON object with 'structured' and 'rendered' fields.`);
  }

  if (taskType === "breakdown") {
    return {
      structured: validateBreakdownDoc(parsed.structured),
      rendered: parsed.rendered.trim(),
    };
  }

  return {
    structured: validateIssueSpecDoc(parsed.structured),
    rendered: parsed.rendered.trim(),
  };
}

export function extractArtifactBody(responseText: string): string {
  const trimmed = responseText.trim();
  const candidates = [trimmed];
  const wrappedJsonMatch = trimmed.match(/^# [^\n]+\n\n(\{[\s\S]*\})$/);
  if (wrappedJsonMatch?.[1]) {
    candidates.push(wrappedJsonMatch[1]);
  }

  for (const candidate of candidates) {
    if (!candidate.startsWith("{")) {
      continue;
    }
    try {
      const parsed = JSON.parse(candidate) as TaskLoopEnvelope;
      if (typeof parsed.result === "string" && parsed.result.trim().length > 0) {
        return parsed.result;
      }
      if (typeof parsed.responseText === "string" && parsed.responseText.trim().length > 0) {
        return parsed.responseText;
      }
    } catch {
      // Keep the original response when it is plain text or non-envelope JSON.
    }
  }

  return responseText;
}

function artifactsForResponse(
  request: TaskExecutionRequest,
  responseText: string,
): { artifacts: PendingArtifact[]; outcomeContent: string } {
  const info = artifactInfoForTask(request.taskType);
  if (!isStrictStructuredArtifactTask(request.taskType)) {
    const content = extractArtifactBody(responseText).trim();
    return {
      artifacts: [{
        kind: info.kind,
        fileName: info.fileName,
        content,
        mimeType: "text/markdown",
      }],
      outcomeContent: content,
    };
  }

  const { structured, rendered } = parseStructuredArtifactEnvelope(request.taskType, responseText);
  const baseName = info.fileName.replace(/\.md$/i, "");
  return {
    artifacts: [
      {
        kind: info.kind,
        fileName: `${baseName}.json`,
        content: JSON.stringify(structured, null, 2),
        mimeType: "application/json",
        variant: "structured",
      },
      {
        kind: info.kind,
        fileName: info.fileName,
        content: rendered,
        mimeType: "text/markdown",
        variant: "rendered",
      },
    ],
    outcomeContent: rendered,
  };
}

export async function writeTaskArtifacts(
  artifactDir: string,
  artifacts: PendingArtifact[],
): Promise<ArtifactRef[]> {
  await mkdir(artifactDir, { recursive: true });
  const createdAt = new Date().toISOString();
  return await Promise.all(artifacts.map(async (artifact) => {
    const path = join(artifactDir, artifact.fileName);
    await writeFile(path, artifact.content.trim() + "\n");
    return {
      kind: artifact.kind,
      path,
      variant: artifact.variant,
      mimeType: artifact.mimeType,
      createdAt,
    };
  }));
}

interface WriteTaskResultOptions {
  readonly request: TaskExecutionRequest;
  readonly artifactDir: string;
  readonly status: TaskExecutionResult["status"];
  readonly startedAt: string;
  readonly artifacts: ArtifactRef[];
  readonly error?: TaskExecutionResult["error"];
  readonly metadata?: Pick<TaskExecutionResult, "session" | "outcome" | "outcomeReason">;
}

export async function writeTaskResult(options: WriteTaskResultOptions): Promise<TaskExecutionResult> {
  const { request, artifactDir, status, startedAt, artifacts, error, metadata = {} } = options;
  const result: TaskExecutionResult = {
    protocolVersion: PROTOCOL_VERSION,
    taskId: request.taskId,
    status,
    artifacts,
    session: metadata.session,
    outcome: metadata.outcome,
    outcomeReason: metadata.outcomeReason,
    metrics: {
      startedAt,
      finishedAt: new Date().toISOString(),
      durationMs: Date.now() - new Date(startedAt).getTime(),
    },
    error,
  };
  await mkdir(artifactDir, { recursive: true });
  await writeFile(join(artifactDir, "result.json"), JSON.stringify(result, null, 2));
  return result;
}

async function runShellCommand(command: string, cwd: string): Promise<VerifyCommandRun> {
  let resolvedCommand = command;
  const nodeBinary = await resolveVerifyNodeBinary();
  if (nodeBinary) {
    resolvedCommand = rewriteVerifyCommand(command, nodeBinary);
  }

  return new Promise((resolveCommand) => {
    exec(
      resolvedCommand,
      {
        cwd,
        encoding: "utf-8",
        shell: process.env["SHELL"] ?? "/bin/sh",
        env: process.env,
      },
      (error: ExecException | null, stdout: string, stderr: string) => {
      const exitCode =
        typeof (error as { code?: unknown } | null)?.code === "number"
          ? ((error as { code: number }).code)
          : error
            ? 1
            : 0;
      resolveCommand({
        command,
        status: error ? "failed" : "passed",
        exitCode,
        stdout: stdout.trim(),
        stderr: stderr.trim(),
      });
      },
    );
  });
}

export async function executeVerifyCommands(
  commands: string[] | undefined,
  cwd: string,
  emit?: (line: { stream: "stdout" | "stderr"; message: string }) => void,
): Promise<{ success: boolean; runs: VerifyCommandRun[]; report: string }> {
  const runs: VerifyCommandRun[] = [];
  for (const command of commands ?? []) {
    const result = await runShellCommand(command, cwd);
    runs.push(result);
    if (result.stdout) {
      emit?.({ stream: "stdout", message: `$ ${command}\n${result.stdout}` });
    }
    if (result.stderr) {
      emit?.({ stream: "stderr", message: `$ ${command}\n${result.stderr}` });
    }
  }

  const success = runs.every((run) => run.status === "passed");
  const sections = [
    "# Verification Report",
    "",
    runs.length === 0 ? "No verification commands were provided." : "",
    ...runs.flatMap((run) => [
      `## ${run.command}`,
      `- Status: ${run.status}`,
      `- Exit code: ${run.exitCode}`,
      run.stdout ? `- Stdout:\n\n\`\`\`\n${run.stdout}\n\`\`\`` : "- Stdout: (empty)",
      run.stderr ? `- Stderr:\n\n\`\`\`\n${run.stderr}\n\`\`\`` : "- Stderr: (empty)",
      "",
    ]),
    `Overall result: ${success ? "pass" : "fail"}`,
  ];

  return {
    success,
    runs,
    report: sections.filter(Boolean).join("\n"),
  };
}
export async function executeTask(options: ExecuteTaskOptions): Promise<TaskExecutionResult> {
  const { request, artifactDir, repoRoot, runQuery, emit } = options;
  validateExecutionCapabilities(request);

  const startedAt = new Date().toISOString();
  emitStartedEvents(request, emit, startedAt);

  try {
    const resolvedSkills = await resolveAndLogRequestedSkills(request, repoRoot, emit);
    const state = await runTaskExecution({
      request,
      artifactDir,
      repoRoot,
      runQuery,
      emit,
      resolvedSkills,
    });
    return await writeExecutionResult(request, artifactDir, startedAt, state, emit);
  } catch (error) {
    return writeCaughtExecutionFailure(request, artifactDir, startedAt, error, emit);
  }
}

function emitStartedEvents(
  request: TaskExecutionRequest,
  emit: ExecuteTaskOptions["emit"],
  startedAt: string,
): void {
  emit({ protocolVersion: PROTOCOL_VERSION, type: "started", at: startedAt, taskId: request.taskId });
  emit({
    protocolVersion: PROTOCOL_VERSION,
    type: "progress",
    at: new Date().toISOString(),
    taskId: request.taskId,
    message: `Executing ${request.taskType}`,
  });
}

async function resolveAndLogRequestedSkills(
  request: TaskExecutionRequest,
  repoRoot: string,
  emit: ExecuteTaskOptions["emit"],
): Promise<ResolvedRequestedSkill[]> {
  const resolvedSkills = await resolveRequestedSkills(
    repoRoot,
    request.context.skills,
    request.taskId,
    (message) => emitLog(request, emit, "stderr", message),
  );
  if (resolvedSkills.length) {
    emitLog(
      request,
      emit,
      "stdout",
      `Resolved skills: ${resolvedSkills.map((skill) => skill.name).join(", ")}`,
    );
  }
  return resolvedSkills;
}

function emitLog(
  request: TaskExecutionRequest,
  emit: ExecuteTaskOptions["emit"],
  stream: "stdout" | "stderr",
  message: string,
): void {
  emit({ protocolVersion: PROTOCOL_VERSION, type: "log", at: new Date().toISOString(), taskId: request.taskId, stream, message });
}

async function runTaskExecution(
  options: ExecuteTaskOptions & { resolvedSkills: ResolvedRequestedSkill[] },
): Promise<TaskRunState> {
  if (options.request.taskType === "verify") {
    return runVerifyTask(options.request, options.repoRoot, options.emit);
  }
  const fakeResponse = readFakeTaskResponse(options.request.taskType);
  if (fakeResponse !== undefined) {
    return { artifactResponseText: fakeResponse, outcomeContent: "", status: "success" };
  }
  return runWorkflowTask(options);
}

async function runVerifyTask(
  request: TaskExecutionRequest,
  repoRoot: string,
  emit: ExecuteTaskOptions["emit"],
): Promise<TaskRunState> {
  const verifyResult = await executeVerifyCommands(request.constraints.verifyCommands, repoRoot, (line) => {
    emitLog(request, emit, line.stream, line.message);
  });
  return {
    artifactResponseText: verifyResult.report,
    outcomeContent: verifyResult.report,
    status: verifyResult.success ? "success" : "failed",
    error: verifyResult.success
      ? undefined
      : { code: "EXECUTION_FAILED", message: "One or more verification commands failed" },
    outcome: verifyResult.success ? "completed" : undefined,
  };
}

async function runWorkflowTask(
  options: ExecuteTaskOptions & { resolvedSkills: ResolvedRequestedSkill[] },
): Promise<TaskRunState> {
  const queryResult = await options.runQuery(buildRunQueryOptions(options));
  const failed = !queryResult.success;
  return {
    artifactResponseText: queryResult.responseText,
    outcomeContent: "",
    status: queryResult.success ? "success" : "failed",
    error: failed ? failedWorkflowError(queryResult) : undefined,
    session: queryResult.session,
    outcome: failed ? "no_progress" : undefined,
    outcomeReason: failed ? classifyFailedWorkflowResult(queryResult) : undefined,
  };
}

function buildRunQueryOptions(
  options: ExecuteTaskOptions & { resolvedSkills: ResolvedRequestedSkill[] },
): Parameters<ExecuteTaskOptions["runQuery"]>[0] {
  const { request, artifactDir, repoRoot, resolvedSkills } = options;
  return {
    query: buildTaskQuery(request, resolvedSkills),
    taskType: request.taskType,
    repoPath: repoRoot,
    provider: request.executor.provider,
    model: request.executor.model,
    maxIterations: request.constraints.maxIterations,
    approvalMode: request.executor.approvalMode ?? "full-auto",
    reasoning: request.executor.reasoning,
    eventsPath: resolve(artifactDir, "engine-events.jsonl"),
    requestedSkills: request.context.skills,
    continuation: request.continuation,
  };
}

function failedWorkflowError(queryResult: WorkflowQueryResult): TaskExecutionResult["error"] {
  const message = queryResult.outcomeReason === "iteration_limit"
    ? "Task loop exhausted the iteration limit"
    : queryResult.outcomeReason === "no_code"
    ? "Task loop produced no final answer"
    : "Task loop failed";
  return { code: "EXECUTION_FAILED", message };
}

async function writeExecutionResult(
  request: TaskExecutionRequest,
  artifactDir: string,
  startedAt: string,
  state: TaskRunState,
  emit: ExecuteTaskOptions["emit"],
): Promise<TaskExecutionResult> {
  if (request.taskType !== "verify" && state.status !== "success") {
    return writeFinalResult({ request, artifactDir, startedAt, state, artifacts: [], emit });
  }
  const artifactOutputs = artifactsForResponse(request, state.artifactResponseText);
  const outcomeContent = state.outcomeContent || artifactOutputs.outcomeContent;
  const artifacts = await writeAndEmitArtifacts(request, artifactDir, artifactOutputs.artifacts, emit);
  const successOutcome = state.status === "success" ? classifySuccessArtifactOutcome(outcomeContent) : {};
  return writeFinalResult({
    request,
    artifactDir,
    startedAt,
    state: { ...state, ...successOutcome },
    artifacts,
    emit,
  });
}

async function writeAndEmitArtifacts(
  request: TaskExecutionRequest,
  artifactDir: string,
  pendingArtifacts: PendingArtifact[],
  emit: ExecuteTaskOptions["emit"],
): Promise<ArtifactRef[]> {
  const artifacts = await writeTaskArtifacts(artifactDir, pendingArtifacts);
  for (const artifact of artifacts) {
    emit({ protocolVersion: PROTOCOL_VERSION, type: "artifact", at: new Date().toISOString(), taskId: request.taskId, artifact });
  }
  return artifacts;
}

async function writeFinalResult(options: FinalResultOptions): Promise<TaskExecutionResult> {
  const { request, artifactDir, startedAt, state, artifacts, emit } = options;
  const result = await writeTaskResult({
    request,
    artifactDir,
    status: state.status,
    startedAt,
    artifacts,
    error: state.error,
    metadata: {
      session: state.session,
      outcome: state.outcome,
      outcomeReason: state.outcomeReason,
    },
  });
  emitCompleted(request, emit, result.status);
  return result;
}

async function writeCaughtExecutionFailure(
  request: TaskExecutionRequest,
  artifactDir: string,
  startedAt: string,
  error: unknown,
  emit: ExecuteTaskOptions["emit"],
): Promise<TaskExecutionResult> {
  const state: TaskRunState = {
    artifactResponseText: "",
    outcomeContent: "",
    status: "failed",
    error: { code: "EXECUTION_FAILED", message: extractErrorMessage(error) },
    outcome: "no_progress",
  };
  return writeFinalResult({ request, artifactDir, startedAt, state, artifacts: [], emit });
}

function emitCompleted(
  request: TaskExecutionRequest,
  emit: ExecuteTaskOptions["emit"],
  status: TaskExecutionResult["status"],
): void {
  emit({ protocolVersion: PROTOCOL_VERSION, type: "completed", at: new Date().toISOString(), taskId: request.taskId, status });
}
