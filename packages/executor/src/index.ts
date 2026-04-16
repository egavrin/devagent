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

type RequestComment = NonNullable<TaskExecutionRequest["context"]["comments"]>[number];

type WorkflowCommentAuthor =
  | "design-artifact"
  | "breakdown-artifact"
  | "issue-spec-artifact"
  | "implementation-summary"
  | "review-report";

type ContextSectionId =
  | "summary"
  | "issueBody"
  | "designArtifact"
  | "breakdownArtifact"
  | "issueSpecArtifact"
  | "implementationSummary"
  | "reviewReport"
  | "issueUnit"
  | "contextBundle"
  | "focusFiles"
  | "comments"
  | "skills"
  | "extraInstructions";

const WORKFLOW_CONTEXT_PREVIEW_CHARS = 4_000;

const WORKFLOW_COMMENT_SECTION_LABELS: Record<WorkflowCommentAuthor, string> = {
  "design-artifact": "Approved design artifact",
  "breakdown-artifact": "Approved breakdown artifact",
  "issue-spec-artifact": "Approved issue spec artifact",
  "implementation-summary": "Implementation summary artifact",
  "review-report": "Review report artifact",
};

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

function formatSection(title: string, body: string | ReadonlyArray<string> | undefined): string | undefined {
  if (body === undefined) {
    return undefined;
  }

  const normalized = typeof body === "string" ? body : body.join("\n");
  const trimmed = normalized.trim();
  if (trimmed.length === 0) {
    return undefined;
  }

  return `${title}:\n${trimmed}`;
}

function truncateWorkflowContextPreview(content: string): string {
  const trimmed = content.trim();
  if (trimmed.length <= WORKFLOW_CONTEXT_PREVIEW_CHARS) {
    return trimmed;
  }

  return `${trimmed.slice(0, WORKFLOW_CONTEXT_PREVIEW_CHARS).trimEnd()}\n[workflow context truncated at ${WORKFLOW_CONTEXT_PREVIEW_CHARS} chars]`;
}

function normalizeWorkflowCommentBody(author: WorkflowCommentAuthor, body: string): string {
  const label = WORKFLOW_COMMENT_SECTION_LABELS[author];
  const normalized = body.replace(/\r\n/g, "\n").trim();
  const prefix = `${label}:`;
  if (!normalized.startsWith(prefix)) {
    return normalized;
  }

  return normalized.slice(prefix.length).trimStart();
}

function isWorkflowCommentAuthor(author: string | undefined): author is WorkflowCommentAuthor {
  if (!author) {
    return false;
  }

  return author in WORKFLOW_COMMENT_SECTION_LABELS;
}

function classifyContextComments(
  comments: ReadonlyArray<RequestComment> | undefined,
): {
  readonly workflow: Partial<Record<WorkflowCommentAuthor, string[]>>;
  readonly generic: RequestComment[];
} {
  const workflow: Partial<Record<WorkflowCommentAuthor, string[]>> = {};
  const generic: RequestComment[] = [];

  for (const comment of comments ?? []) {
    if (isWorkflowCommentAuthor(comment.author)) {
      const author = comment.author;
      const normalizedBody = normalizeWorkflowCommentBody(author, comment.body);
      workflow[author] = [...(workflow[author] ?? []), normalizedBody];
      continue;
    }
    generic.push(comment);
  }

  return { workflow, generic };
}

function renderWorkflowCommentSection(
  author: WorkflowCommentAuthor,
  bodies: ReadonlyArray<string> | undefined,
): string | undefined {
  if (!bodies || bodies.length === 0) {
    return undefined;
  }

  const rendered = bodies.map((body, index) => {
    const preview = truncateWorkflowContextPreview(body);
    if (bodies.length === 1) {
      return preview;
    }
    return `Excerpt ${index + 1}:\n${preview}`;
  }).join("\n\n");

  return formatSection(WORKFLOW_COMMENT_SECTION_LABELS[author], rendered);
}

function formatGenericComment(comment: RequestComment): string {
  const author = comment.author ?? "unknown";
  if (!comment.body.includes("\n")) {
    return `- ${author}: ${comment.body}`;
  }

  const indentedBody = comment.body
    .split("\n")
    .map((line) => `  ${line}`)
    .join("\n");
  return `- ${author}:\n${indentedBody}`;
}

function renderIssueUnitSection(issueUnit: TaskExecutionRequest["issueUnit"]): string | undefined {
  if (!issueUnit) {
    return undefined;
  }

  const lines = [
    `Issue unit: [${issueUnit.sequence}] ${issueUnit.title}`,
  ];
  if (issueUnit.dependencyIds.length > 0) {
    lines.push(`Dependencies: ${issueUnit.dependencyIds.join(", ")}`);
  }
  if (issueUnit.linkedArtifactVersionIds.length > 0) {
    lines.push(`Linked artifact version ids: ${issueUnit.linkedArtifactVersionIds.join(", ")}`);
  }
  if (issueUnit.acceptanceCriteria.length > 0) {
    lines.push(`Issue acceptance criteria:\n${issueUnit.acceptanceCriteria.map((criterion) => `- ${criterion}`).join("\n")}`);
  }

  return formatSection("Issue unit details", lines);
}

function renderContextBundleSection(contextBundle: TaskExecutionRequest["contextBundle"]): string | undefined {
  if (!contextBundle) {
    return undefined;
  }

  return formatSection("Context bundle details", [
    `Context bundle: ${contextBundle.id}`,
    `Summary: ${contextBundle.summary}`,
    `Artifact version ids: ${contextBundle.artifactVersionIds.join(", ") || "(none)"}`,
  ]);
}

function dedupePreservingOrder(values: ReadonlyArray<string> | undefined): string[] {
  const seen = new Set<string>();
  const deduped: string[] = [];

  for (const value of values ?? []) {
    if (seen.has(value)) {
      continue;
    }
    seen.add(value);
    deduped.push(value);
  }

  return deduped;
}

function buildContextSectionOrder(taskType: TaskExecutionRequest["taskType"]): readonly ContextSectionId[] {
  switch (taskType) {
    case "breakdown":
      return [
        "designArtifact",
        "summary",
        "issueBody",
        "contextBundle",
        "breakdownArtifact",
        "issueSpecArtifact",
        "implementationSummary",
        "reviewReport",
        "issueUnit",
        "focusFiles",
        "comments",
        "skills",
        "extraInstructions",
      ];
    case "issue-generation":
      return [
        "designArtifact",
        "breakdownArtifact",
        "summary",
        "issueBody",
        "contextBundle",
        "issueSpecArtifact",
        "implementationSummary",
        "reviewReport",
        "issueUnit",
        "focusFiles",
        "comments",
        "skills",
        "extraInstructions",
      ];
    case "implement":
      return [
        "issueSpecArtifact",
        "issueUnit",
        "focusFiles",
        "summary",
        "issueBody",
        "contextBundle",
        "breakdownArtifact",
        "designArtifact",
        "implementationSummary",
        "reviewReport",
        "comments",
        "skills",
        "extraInstructions",
      ];
    case "review":
      return [
        "issueSpecArtifact",
        "implementationSummary",
        "issueUnit",
        "focusFiles",
        "summary",
        "issueBody",
        "contextBundle",
        "breakdownArtifact",
        "designArtifact",
        "reviewReport",
        "comments",
        "skills",
        "extraInstructions",
      ];
    case "repair":
      return [
        "reviewReport",
        "implementationSummary",
        "issueSpecArtifact",
        "issueUnit",
        "focusFiles",
        "summary",
        "issueBody",
        "contextBundle",
        "breakdownArtifact",
        "designArtifact",
        "comments",
        "skills",
        "extraInstructions",
      ];
    default:
      return [
        "summary",
        "issueBody",
        "contextBundle",
        "designArtifact",
        "breakdownArtifact",
        "issueSpecArtifact",
        "implementationSummary",
        "reviewReport",
        "issueUnit",
        "focusFiles",
        "comments",
        "skills",
        "extraInstructions",
      ];
  }
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
  const sections = [
    `Task type: ${request.taskType}`,
    `${workItemLabel}: ${request.workItem.title ?? request.workItem.externalId}`,
  ];

  if (request.workItem.kind === "local-task") {
    sections.push("Task source: local/manual");
  } else {
    sections.push(`Task source: ${request.workItem.kind}`);
  }

  if (primaryRepository) {
    sections.push(`Primary repository: ${primaryRepository.alias} (${primaryRepository.repoRoot})`);
  }

  const repositorySection = formatSection("Repository context", buildRepositoryContext(request));
  if (repositorySection) {
    sections.push(repositorySection);
  }

  const reviewableSection = formatSection("Reviewable context", buildReviewableContext(request.reviewable));
  if (reviewableSection) {
    sections.push(reviewableSection);
  }

  const classifiedComments = classifyContextComments(request.context.comments);
  const contextSections: Partial<Record<ContextSectionId, string>> = {
    summary: formatSection("Summary", request.context.summary),
    issueBody: formatSection("Issue body", request.context.issueBody),
    designArtifact: renderWorkflowCommentSection("design-artifact", classifiedComments.workflow["design-artifact"]),
    breakdownArtifact: renderWorkflowCommentSection("breakdown-artifact", classifiedComments.workflow["breakdown-artifact"]),
    issueSpecArtifact: renderWorkflowCommentSection("issue-spec-artifact", classifiedComments.workflow["issue-spec-artifact"]),
    implementationSummary: renderWorkflowCommentSection("implementation-summary", classifiedComments.workflow["implementation-summary"]),
    reviewReport: renderWorkflowCommentSection("review-report", classifiedComments.workflow["review-report"]),
    issueUnit: renderIssueUnitSection(request.issueUnit),
    contextBundle: renderContextBundleSection(request.contextBundle),
    focusFiles: formatSection(
      "Focus files",
      dedupePreservingOrder(request.context.changedFilesHint).map((filePath) => `- ${filePath}`),
    ),
    comments: formatSection("Comments", classifiedComments.generic.map(formatGenericComment)),
    skills: resolvedSkills.length > 0
      ? formatSection(
          "Requested skills",
          resolvedSkills.map((skill) => `## ${skill.name}\nSource: ${skill.source}\n${skill.instructions}`).join("\n\n"),
        )
      : undefined,
    extraInstructions: request.context.extraInstructions?.length
      ? formatSection("Extra instructions", request.context.extraInstructions)
      : undefined,
  };

  for (const sectionId of buildContextSectionOrder(request.taskType)) {
    const section = contextSections[sectionId];
    if (section) {
      sections.push(section);
    }
  }

  switch (request.taskType) {
    case "task-intake":
      sections.push("Workspace is analysis-only for task intake. No file changes are allowed.");
      sections.push("Do not run project verification commands unless the request explicitly requires them.");
      sections.push("Do not use update_plan for this stage. Inspect the repo as needed, then return the final artifact directly.");
      sections.push("Produce a structured task specification with goals, constraints, assumptions, and acceptance criteria.");
      break;
    case "design":
      sections.push("Workspace is design-only for this stage. No file changes are allowed.");
      sections.push("Do not run project verification commands unless the request explicitly requires them.");
      sections.push("Do not use update_plan for this stage. Inspect the repo as needed, then return the final artifact directly.");
      sections.push("Produce a structured design document with architecture outline, interfaces, risks, tradeoffs, and validation strategy.");
      break;
    case "breakdown":
      sections.push("Workspace is breakdown-only for this stage. No file changes are allowed.");
      sections.push("Do not run project verification commands unless the request explicitly requires them.");
      sections.push("Do not use update_plan for this stage. Inspect the repo as needed, then return the final artifact directly.");
      sections.push("Produce an implementation breakdown as an ordered checklist of small executable tasks, not a design narrative.");
      sections.push("Ground every task in the approved design, current repository state, and concrete repo paths or symbols you inspected.");
      sections.push("Every task must be independently executable, reviewable, and scoped to fewer than 500 changed lines.");
      sections.push("Include explicit dependencies, acceptance criteria, expected changes, validation commands, and risk notes for each task.");
      sections.push("Do not emit section headings as tasks and do not emit prose-only summaries in place of task records.");
      sections.push("Return strict JSON with this exact top-level shape: {\"structured\": <BreakdownDoc>, \"rendered\": <Markdown checklist>}.");
      sections.push(`Use exactly this BreakdownDoc schema:
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
      sections.push("Use exactly those property names. Do not rename keys, omit required fields, or add extra fields.");
      sections.push("The rendered markdown must use one checklist item per task in execution order, for example '- [ ] B1. Add input normalization in src/foo.ts'.");
      break;
    case "issue-generation":
      sections.push("Workspace is issue-generation only. No file changes are allowed.");
      sections.push("Do not run project verification commands unless the request explicitly requires them.");
      sections.push("Do not use update_plan for this stage. Inspect the repo as needed, then return the final artifact directly.");
      sections.push("Generate executable issue specs directly from the approved breakdown tasks. Do not infer issues from document headings.");
      sections.push("Every issue must link to one or more approved breakdown task ids, preserve the breakdown execution order, and reference concrete repo paths or symbols.");
      sections.push("Do not invent standalone issues outside the approved breakdown or drop any approved breakdown tasks.");
      sections.push("Return strict JSON with this exact top-level shape: {\"structured\": <IssueSpecDoc>, \"rendered\": <Markdown summary>}.");
      sections.push(`Use exactly this IssueSpecDoc schema:
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
      sections.push("Use exactly those property names. Do not rename keys, omit required fields, or add extra fields.");
      break;
    case "triage":
      sections.push("Workspace is analysis-only for triage. No file changes are allowed.");
      sections.push("Do not run project verification commands unless the request explicitly requires them.");
      sections.push("Do not use update_plan for this stage. Inspect the repo as needed, then return the final artifact directly.");
      sections.push("Produce a concise triage report covering issue understanding, impact area, risks, unknowns, and next step.");
      break;
    case "plan":
      sections.push("Workspace is planning-only for plan. No file changes are allowed.");
      sections.push("Do not run project verification commands unless the request explicitly requires them.");
      sections.push("Do not use update_plan for this stage. Inspect the repo as needed, then return the final artifact directly.");
      sections.push("Produce a concise implementation plan covering steps, affected files/components, test strategy, and rollback/risk notes.");
      break;
    case "test-plan":
      sections.push("Workspace is planning-only for test-plan. No file changes are allowed.");
      sections.push("Do not run project verification commands unless the request explicitly requires them.");
      sections.push("Do not use update_plan for this stage. Inspect the repo as needed, then return the final artifact directly.");
      sections.push("Produce a test plan with scenarios, edge cases, regression risks, required tests, and expected outcomes.");
      break;
    case "implement":
      sections.push("Implement the requested change in the current workspace, then summarize the changed files, edits, and blockers.");
      break;
    case "verify":
      sections.push(
        `Verification commands will run outside the model. Summarize the verification outcome and any follow-up actions based on these commands:\n${request.constraints.verifyCommands?.join("\n") ?? "No commands provided."}`,
      );
      break;
    case "review":
      sections.push("Review the current workspace changes and produce a report with either `No defects found.` or one section per defect using the format `Severity: <low|medium|high|critical>` plus a concrete fix recommendation.");
      break;
    case "repair":
      sections.push("Apply repairs for the current issue, address the review findings, and summarize fixes applied plus remaining concerns.");
      break;
    case "completion":
      sections.push("Workspace is completion-only for this stage. No file changes are allowed.");
      sections.push("Do not run project verification commands unless the request explicitly requires them.");
      sections.push("Do not use update_plan for this stage. Inspect the repo as needed, then return the final artifact directly.");
      sections.push("Produce a workflow summary covering completed issues, remaining risks, key decisions, and artifact chain highlights.");
      break;
  }

  if (isStrictStructuredArtifactTask(request.taskType)) {
    sections.push("Return only the JSON object without code fences or surrounding commentary.");
  } else {
    sections.push("Return plain Markdown without code fences around the entire response.");
  }
  return sections.join("\n\n");
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
