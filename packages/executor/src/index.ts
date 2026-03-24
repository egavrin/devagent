import { exec, execFile, type ExecException, type ExecFileException } from "node:child_process";
import { constants } from "node:fs";
import { access, mkdir, readFile, writeFile } from "node:fs/promises";
import { delimiter, join, resolve } from "node:path";
import {
  SkillLoader,
  SkillRegistry,
  SkillResolver,
  extractErrorMessage,
} from "@devagent/runtime";
import {
  validateBreakdownDoc,
  validateIssueSpecDoc,
  validateTaskExecutionRequest,
} from "@devagent-sdk/validation";
import {
  PROTOCOL_VERSION,
  type ArtifactKind,
  type ArtifactRef,
  type ArtifactVariant,
  type BreakdownDoc,
  type ContinuationSession,
  type IssueSpecDoc,
  type RepositoryRef,
  type ReviewableRef,
  type TaskExecutionEvent,
  type TaskExecutionRequest,
  type TaskExecutionResult,
} from "@devagent-sdk/types";

export interface ExecuteArgs {
  requestPath: string;
  artifactDir: string;
}

export interface ResolvedRequestedSkill {
  name: string;
  description: string;
  source: string;
  instructions: string;
}

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

function primaryRepositoryForRequest(request: TaskExecutionRequest): RepositoryRef | undefined {
  return request.repositories.find((repository) => repository.id === request.workspaceRef.primaryRepositoryId);
}

function buildRepositoryContext(request: TaskExecutionRequest): string[] {
  const repositoryNames = request.repositories.map((repository) => {
    const role = repository.id === request.workspaceRef.primaryRepositoryId ? "primary" : "secondary";
    return `- ${repository.alias} (${role}): ${repository.repoRoot}`;
  });

  const targetRepositories = request.targetRepositoryIds
    .map((targetId) => request.repositories.find((repository) => repository.id === targetId))
    .filter((repository): repository is RepositoryRef => Boolean(repository))
    .map((repository) => repository.alias);

  const lines = [
    `Workspace: ${request.workspaceRef.name}`,
    `Workspace provider: ${request.workspaceRef.provider}`,
  ];

  if (repositoryNames.length) {
    lines.push(`Repositories:\n${repositoryNames.join("\n")}`);
  }

  if (targetRepositories.length) {
    lines.push(`Target repositories: ${targetRepositories.join(", ")}`);
  }

  return lines;
}

function buildReviewableContext(reviewable: ReviewableRef | undefined): string[] {
  if (!reviewable) {
    return [];
  }

  const lines = [
    `Review target: ${reviewable.type} ${reviewable.externalId}`,
  ];
  if (reviewable.title) {
    lines.push(`Review title: ${reviewable.title}`);
  }
  if (reviewable.url) {
    lines.push(`Review URL: ${reviewable.url}`);
  }
  return lines;
}

export interface VerifyCommandRun {
  command: string;
  status: "passed" | "failed";
  exitCode: number;
  stdout: string;
  stderr: string;
}

type TaskLoopEnvelope = {
  result?: string;
  responseText?: string;
};

type StructuredArtifactEnvelope = {
  structured: unknown;
  rendered: string;
};

type PendingArtifact = {
  kind: ArtifactKind;
  fileName: string;
  content: string;
  mimeType: string;
  variant?: ArtifactVariant;
};

function classifyFailedWorkflowResult(result: WorkflowQueryResult): TaskExecutionResult["outcomeReason"] | undefined {
  if (result.outcomeReason) {
    return result.outcomeReason;
  }
  return undefined;
}

function classifySuccessArtifactOutcome(content: string): Pick<TaskExecutionResult, "outcome" | "outcomeReason"> {
  if (content.trim().length === 0) {
    return {
      outcome: "no_progress",
      outcomeReason: "empty_artifact",
    };
  }
  return {
    outcome: "completed",
  };
}

const PREFERRED_VERIFY_PATHS = [
  "/opt/homebrew/bin",
  "/opt/homebrew/sbin",
  "/usr/local/bin",
  "/usr/local/sbin",
  "/usr/bin",
  "/usr/sbin",
  "/bin",
  "/sbin",
];

const LEADING_NODE_COMMAND = /^(\s*(?:[A-Za-z_][A-Za-z0-9_]*=(?:"[^"]*"|'[^']*'|[^\s]+)\s+)*)((?:\/usr\/bin\/env|env)\s+)?node(?=\s|$)/;

function buildVerifyNodeSearchPath(currentPath = process.env["PATH"] ?? ""): string[] {
  return Array.from(new Set([
    ...currentPath.split(delimiter).filter(Boolean),
    ...PREFERRED_VERIFY_PATHS,
  ]));
}

function quoteShellArgument(value: string): string {
  return `'${value.replaceAll("'", `'\"'\"'`)}'`;
}

async function isExecutable(path: string): Promise<boolean> {
  try {
    await access(path, constants.X_OK);
    return true;
  } catch {
    return false;
  }
}

async function isRealNodeBinary(path: string): Promise<boolean> {
  return await new Promise((resolveCommand) => {
    execFile(
      path,
      ["-p", "process.versions && process.versions.bun ? 'bun' : ((process.release && process.release.name) || '')"],
      {
        encoding: "utf8",
        timeout: 5_000,
      },
      (error: ExecFileException | null, stdout: string) => {
        resolveCommand(!error && stdout.trim() === "node");
      },
    );
  });
}

export async function resolveVerifyNodeBinary(currentPath = process.env["PATH"] ?? ""): Promise<string | null> {
  for (const entry of buildVerifyNodeSearchPath(currentPath)) {
    const candidate = join(entry, "node");
    if (!await isExecutable(candidate)) {
      continue;
    }
    if (await isRealNodeBinary(candidate)) {
      return candidate;
    }
  }

  return null;
}

export function rewriteVerifyCommand(command: string, nodeBinary: string): string {
  if (!LEADING_NODE_COMMAND.test(command)) {
    return command;
  }

  return command.replace(LEADING_NODE_COMMAND, (_, prefix: string) => `${prefix}${quoteShellArgument(nodeBinary)}`);
}

function fakeResponseEnvKey(taskType: TaskExecutionRequest["taskType"]): string {
  return `DEVAGENT_EXECUTOR_FAKE_RESPONSE_${taskType.replace(/[^A-Za-z0-9]+/g, "_").toUpperCase()}`;
}

export function readFakeTaskResponse(taskType: TaskExecutionRequest["taskType"]): string | undefined {
  return process.env[fakeResponseEnvKey(taskType)] ?? process.env["DEVAGENT_EXECUTOR_FAKE_RESPONSE"];
}

export function parseExecuteArgs(argv: string[]): ExecuteArgs | null {
  const args = argv.slice(2);
  if (args[0] !== "execute") return null;

  let requestPath: string | null = null;
  let artifactDir: string | null = null;

  for (let i = 1; i < args.length; i++) {
    const arg = args[i]!;
    if (arg === "--request" && args[i + 1]) {
      requestPath = args[++i]!;
      continue;
    }
    if (arg === "--artifact-dir" && args[i + 1]) {
      artifactDir = args[++i]!;
      continue;
    }
  }

  if (!requestPath || !artifactDir) {
    throw new Error("Usage: devagent execute --request <file> --artifact-dir <dir>");
  }

  return {
    requestPath: resolve(requestPath),
    artifactDir: resolve(artifactDir),
  };
}

export async function loadTaskExecutionRequest(requestPath: string): Promise<TaskExecutionRequest> {
  const parsed = JSON.parse(await readFile(requestPath, "utf-8")) as unknown;
  return validateTaskExecutionRequest(parsed);
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

export function buildTaskQuery(
  request: TaskExecutionRequest,
  resolvedSkills: ResolvedRequestedSkill[] = [],
): string {
  const primaryRepository = primaryRepositoryForRequest(request);
  const workItemLabel = request.workItem.kind === "local-task" ? "Task" : "Issue";
  const lines = [
    `Task type: ${request.taskType}`,
    `${workItemLabel}: ${request.workItem.title ?? request.workItem.externalId}`,
  ];

  if (request.workItem.kind === "local-task") {
    lines.push("Task source: local/manual");
  } else {
    lines.push(`Task source: ${request.workItem.kind}`);
  }

  if (primaryRepository) {
    lines.push(`Primary repository: ${primaryRepository.alias} (${primaryRepository.repoRoot})`);
  }

  lines.push(...buildRepositoryContext(request));
  lines.push(...buildReviewableContext(request.reviewable));

  if (request.context.summary) lines.push(`Summary: ${request.context.summary}`);
  if (request.context.issueBody) lines.push(`Issue body:\n${request.context.issueBody}`);
  if (request.context.comments?.length) {
    lines.push(
      `Comments:\n${request.context.comments
        .map((comment) => `- ${comment.author ?? "unknown"}: ${comment.body}`)
        .join("\n")}`,
    );
  }
  if (request.context.changedFilesHint?.length) {
    lines.push(`Changed file hints:\n${request.context.changedFilesHint.join("\n")}`);
  }
  if (request.issueUnit) {
    lines.push(`Issue unit: [${request.issueUnit.sequence}] ${request.issueUnit.title}`);
    if (request.issueUnit.acceptanceCriteria.length > 0) {
      lines.push(`Issue acceptance criteria:\n${request.issueUnit.acceptanceCriteria.map((criterion) => `- ${criterion}`).join("\n")}`);
    }
  }
  if (request.contextBundle) {
    lines.push(
      `Context bundle: ${request.contextBundle.id}\nSummary: ${request.contextBundle.summary}\nArtifact version ids: ${
        request.contextBundle.artifactVersionIds.join(", ") || "(none)"
      }`,
    );
  }
  if (resolvedSkills.length) {
    lines.push(
      `Requested skills:\n${resolvedSkills
        .map((skill) => `## ${skill.name}\nSource: ${skill.source}\n${skill.instructions}`)
        .join("\n\n")}`,
    );
  }
  if (request.context.extraInstructions?.length) {
    lines.push(`Extra instructions:\n${request.context.extraInstructions.join("\n")}`);
  }

  switch (request.taskType) {
    case "task-intake":
      lines.push("Workspace is analysis-only for task intake. No file changes are allowed.");
      lines.push("Do not run project verification commands unless the request explicitly requires them.");
      lines.push("Do not use update_plan for this stage. Inspect the repo as needed, then return the final artifact directly.");
      lines.push("Produce a structured task specification with goals, constraints, assumptions, and acceptance criteria.");
      break;
    case "design":
      lines.push("Workspace is design-only for this stage. No file changes are allowed.");
      lines.push("Do not run project verification commands unless the request explicitly requires them.");
      lines.push("Do not use update_plan for this stage. Inspect the repo as needed, then return the final artifact directly.");
      lines.push("Produce a structured design document with architecture outline, interfaces, risks, tradeoffs, and validation strategy.");
      break;
    case "breakdown":
      lines.push("Workspace is breakdown-only for this stage. No file changes are allowed.");
      lines.push("Do not run project verification commands unless the request explicitly requires them.");
      lines.push("Do not use update_plan for this stage. Inspect the repo as needed, then return the final artifact directly.");
      lines.push("Produce an implementation breakdown as an ordered checklist of small executable tasks, not a design narrative.");
      lines.push("Ground every task in the approved design, current repository state, and concrete repo paths or symbols you inspected.");
      lines.push("Every task must be independently executable, reviewable, and scoped to fewer than 500 changed lines.");
      lines.push("Include explicit dependencies, acceptance criteria, expected changes, validation commands, and risk notes for each task.");
      lines.push("Do not emit section headings as tasks and do not emit prose-only summaries in place of task records.");
      lines.push("Return strict JSON with this exact top-level shape: {\"structured\": <BreakdownDoc>, \"rendered\": <Markdown checklist>}.");
      lines.push(`Use exactly this BreakdownDoc schema:
{
  "summary": "short summary",
  "executionOrder": ["B1", "B2"],
  "tasks": [
    {
      "id": "B1",
      "title": "short title",
      "checklistLabel": "B1. concrete checklist item",
      "objective": "why this task exists",
      "rationale": "why this slice belongs here",
      "grounding": {
        "designRefs": ["DesignDoc#Section"],
        "repoPaths": ["src/file.ts"],
        "codeSymbols": ["functionName"]
      },
      "dependencies": [],
      "acceptanceCriteria": ["observable outcome"],
      "expectedChanges": ["planned edit"],
      "validation": ["command or check"],
      "riskNotes": ["risk"],
      "sizeBudget": {
        "maxEstimatedChangedLines": 120,
        "estimateReason": "why this stays under 500 lines"
      }
    }
  ]
}`);
      lines.push("Use exactly those property names. Do not rename keys, omit required fields, or add extra fields.");
      lines.push("The rendered markdown must use one checklist item per task in execution order, for example '- [ ] B1. Add input normalization in src/foo.ts'.");
      break;
    case "issue-generation":
      lines.push("Workspace is issue-generation only. No file changes are allowed.");
      lines.push("Do not run project verification commands unless the request explicitly requires them.");
      lines.push("Do not use update_plan for this stage. Inspect the repo as needed, then return the final artifact directly.");
      lines.push("Generate executable issue specs directly from the approved breakdown tasks. Do not infer issues from document headings.");
      lines.push("Every issue must link to one or more approved breakdown task ids, preserve the breakdown execution order, and reference concrete repo paths or symbols.");
      lines.push("Do not invent standalone issues outside the approved breakdown or drop any approved breakdown tasks.");
      lines.push("Return strict JSON with this exact top-level shape: {\"structured\": <IssueSpecDoc>, \"rendered\": <Markdown summary>}.");
      lines.push(`Use exactly this IssueSpecDoc schema:
{
  "summary": "short summary",
  "issues": [
    {
      "id": "I1",
      "title": "short title",
      "problemStatement": "problem to solve",
      "rationale": "why this issue exists",
      "scope": ["in-scope item"],
      "acceptanceCriteria": ["observable outcome"],
      "dependencies": [],
      "linkedDesignSections": ["DesignDoc#Section"],
      "linkedBreakdownTaskIds": ["B1"],
      "grounding": {
        "repoPaths": ["src/file.ts"],
        "codeSymbols": ["functionName"]
      },
      "requiredTests": ["test obligation"],
      "outOfScope": ["not included"],
      "implementationNotes": ["implementation note"]
    }
  ]
}`);
      lines.push("Use exactly those property names. Do not rename keys, omit required fields, or add extra fields.");
      break;
    case "triage":
      lines.push("Workspace is analysis-only for triage. No file changes are allowed.");
      lines.push("Do not run project verification commands unless the request explicitly requires them.");
      lines.push("Do not use update_plan for this stage. Inspect the repo as needed, then return the final artifact directly.");
      lines.push("Produce a concise triage report covering issue understanding, impact area, risks, unknowns, and next step.");
      break;
    case "plan":
      lines.push("Workspace is planning-only for plan. No file changes are allowed.");
      lines.push("Do not run project verification commands unless the request explicitly requires them.");
      lines.push("Do not use update_plan for this stage. Inspect the repo as needed, then return the final artifact directly.");
      lines.push("Produce a concise implementation plan covering steps, affected files/components, test strategy, and rollback/risk notes.");
      break;
    case "test-plan":
      lines.push("Workspace is planning-only for test-plan. No file changes are allowed.");
      lines.push("Do not run project verification commands unless the request explicitly requires them.");
      lines.push("Do not use update_plan for this stage. Inspect the repo as needed, then return the final artifact directly.");
      lines.push("Produce a test plan with scenarios, edge cases, regression risks, required tests, and expected outcomes.");
      break;
    case "implement":
      lines.push("Implement the requested change in the current workspace, then summarize the changed files, edits, and blockers.");
      break;
    case "verify":
      lines.push(
        `Verification commands will run outside the model. Summarize the verification outcome and any follow-up actions based on these commands:\n${request.constraints.verifyCommands?.join("\n") ?? "No commands provided."}`,
      );
      break;
    case "review":
      lines.push("Review the current workspace changes and produce a report with either `No defects found.` or one section per defect using the format `Severity: <low|medium|high|critical>` plus a concrete fix recommendation.");
      break;
    case "repair":
      lines.push("Apply repairs for the current issue, address the review findings, and summarize fixes applied plus remaining concerns.");
      break;
    case "completion":
      lines.push("Workspace is completion-only for this stage. No file changes are allowed.");
      lines.push("Do not run project verification commands unless the request explicitly requires them.");
      lines.push("Do not use update_plan for this stage. Inspect the repo as needed, then return the final artifact directly.");
      lines.push("Produce a workflow summary covering completed issues, remaining risks, key decisions, and artifact chain highlights.");
      break;
  }

  if (isStrictStructuredArtifactTask(request.taskType)) {
    lines.push("Return only the JSON object without code fences or surrounding commentary.");
  } else {
    lines.push("Return plain Markdown without code fences around the entire response.");
  }
  return lines.join("\n\n");
}

export function artifactInfoForTask(taskType: TaskExecutionRequest["taskType"]): {
  kind: ArtifactKind;
  fileName: string;
} {
  switch (taskType) {
    case "task-intake":
      return { kind: "task-spec", fileName: "task-spec.md" };
    case "design":
      return { kind: "design-doc", fileName: "design-doc.md" };
    case "breakdown":
      return { kind: "breakdown-doc", fileName: "breakdown-doc.md" };
    case "issue-generation":
      return { kind: "issue-spec", fileName: "issue-spec.md" };
    case "triage":
      return { kind: "triage-report", fileName: "triage-report.md" };
    case "plan":
      return { kind: "plan", fileName: "plan.md" };
    case "test-plan":
      return { kind: "test-plan", fileName: "test-plan.md" };
    case "implement":
      return { kind: "implementation-summary", fileName: "implementation-summary.md" };
    case "verify":
      return { kind: "verification-report", fileName: "verification-report.md" };
    case "review":
      return { kind: "review-report", fileName: "review-report.md" };
    case "repair":
      return { kind: "final-summary", fileName: "final-summary.md" };
    case "completion":
      return { kind: "workflow-summary", fileName: "workflow-summary.md" };
  }
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

export async function writeTaskResult(
  request: TaskExecutionRequest,
  artifactDir: string,
  status: TaskExecutionResult["status"],
  startedAt: string,
  artifacts: ArtifactRef[],
  error?: TaskExecutionResult["error"],
  metadata: Pick<TaskExecutionResult, "session" | "outcome" | "outcomeReason"> = {},
): Promise<TaskExecutionResult> {
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
  if (LEADING_NODE_COMMAND.test(command)) {
    const nodeBinary = await resolveVerifyNodeBinary();
    if (!nodeBinary) {
      return {
        command,
        status: "failed",
        exitCode: 1,
        stdout: "",
        stderr: "Unable to locate a real Node.js binary for verification. Install Node.js or add it to PATH.",
      };
    }
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
  emit({
    protocolVersion: PROTOCOL_VERSION,
    type: "started",
    at: startedAt,
    taskId: request.taskId,
  });
  emit({
    protocolVersion: PROTOCOL_VERSION,
    type: "progress",
    at: new Date().toISOString(),
    taskId: request.taskId,
    message: `Executing ${request.taskType}`,
  });

  try {
    const resolvedSkills = await resolveRequestedSkills(
      repoRoot,
      request.context.skills,
      request.taskId,
      (message) => {
        emit({
          protocolVersion: PROTOCOL_VERSION,
          type: "log",
          at: new Date().toISOString(),
          taskId: request.taskId,
          stream: "stderr",
          message,
        });
      },
    );
    if (resolvedSkills.length) {
      emit({
        protocolVersion: PROTOCOL_VERSION,
        type: "log",
        at: new Date().toISOString(),
        taskId: request.taskId,
        stream: "stdout",
        message: `Resolved skills: ${resolvedSkills.map((skill) => skill.name).join(", ")}`,
      });
    }

    let artifactResponseText: string;
    let outcomeContent = "";
    let status: TaskExecutionResult["status"];
    let error: TaskExecutionResult["error"] | undefined;
    let session: ContinuationSession | undefined;
    let outcome: TaskExecutionResult["outcome"] | undefined;
    let outcomeReason: TaskExecutionResult["outcomeReason"] | undefined;

    if (request.taskType === "verify") {
      const verifyResult = await executeVerifyCommands(request.constraints.verifyCommands, repoRoot, (line) => {
        emit({
          protocolVersion: PROTOCOL_VERSION,
          type: "log",
          at: new Date().toISOString(),
          taskId: request.taskId,
          stream: line.stream,
          message: line.message,
        });
      });
      artifactResponseText = verifyResult.report;
      outcomeContent = verifyResult.report;
      status = verifyResult.success ? "success" : "failed";
      error = verifyResult.success
        ? undefined
        : { code: "EXECUTION_FAILED", message: "One or more verification commands failed" };
      outcome = verifyResult.success ? "completed" : undefined;
    } else {
      const fakeResponse = readFakeTaskResponse(request.taskType);
      if (fakeResponse !== undefined) {
        artifactResponseText = fakeResponse;
        status = "success";
        error = undefined;
      } else {
        const queryResult = await runQuery({
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
        });
        artifactResponseText = queryResult.responseText;
        session = queryResult.session;
        status = queryResult.success ? "success" : "failed";
        error = queryResult.success
          ? undefined
          : {
            code: "EXECUTION_FAILED",
            message: queryResult.outcomeReason === "iteration_limit"
              ? "Task loop exhausted the iteration limit"
              : queryResult.outcomeReason === "no_code"
              ? "Task loop produced no final answer"
              : "Task loop failed",
          };
        if (!queryResult.success) {
          outcome = "no_progress";
          outcomeReason = classifyFailedWorkflowResult(queryResult);
        }
      }
    }

    if (request.taskType !== "verify" && status !== "success") {
      const result = await writeTaskResult(
        request,
        artifactDir,
        status,
        startedAt,
        [],
        error,
        {
          session,
          outcome,
          outcomeReason,
        },
      );
      emit({
        protocolVersion: PROTOCOL_VERSION,
        type: "completed",
        at: new Date().toISOString(),
        taskId: request.taskId,
        status: result.status,
      });
      return result;
    }

    const artifactOutputs = artifactsForResponse(request, artifactResponseText);
    outcomeContent = outcomeContent || artifactOutputs.outcomeContent;
    if (status === "success") {
      ({ outcome, outcomeReason } = classifySuccessArtifactOutcome(outcomeContent));
    }

    const artifacts = await writeTaskArtifacts(artifactDir, artifactOutputs.artifacts);
    for (const artifact of artifacts) {
      emit({
        protocolVersion: PROTOCOL_VERSION,
        type: "artifact",
        at: new Date().toISOString(),
        taskId: request.taskId,
        artifact,
      });
    }

    const result = await writeTaskResult(
      request,
      artifactDir,
      status,
      startedAt,
      artifacts,
      error,
      {
        session,
        outcome,
        outcomeReason,
      },
    );
    emit({
      protocolVersion: PROTOCOL_VERSION,
      type: "completed",
      at: new Date().toISOString(),
      taskId: request.taskId,
      status: result.status,
    });
    return result;
  } catch (error) {
    const result = await writeTaskResult(
      request,
      artifactDir,
      "failed",
      startedAt,
      [],
      { code: "EXECUTION_FAILED", message: extractErrorMessage(error) },
      {
        outcome: "no_progress",
      },
    );
    emit({
      protocolVersion: PROTOCOL_VERSION,
      type: "completed",
      at: new Date().toISOString(),
      taskId: request.taskId,
      status: result.status,
    });
    return result;
  }
}
