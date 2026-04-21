import type { RepositoryRef, ReviewableRef, TaskExecutionRequest } from "@devagent-sdk/types";

export interface ResolvedRequestedSkill {
  name: string;
  description: string;
  source: string;
  instructions: string;
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

const BREAKDOWN_SCHEMA_INSTRUCTION = `Use exactly this BreakdownDoc schema:
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
}`;

const ISSUE_SPEC_SCHEMA_INSTRUCTION = `Use exactly this IssueSpecDoc schema:
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
}`;

const TASK_TYPE_INSTRUCTIONS: Partial<Record<TaskExecutionRequest["taskType"], readonly string[]>> = {
  "task-intake": [
    "Workspace is analysis-only for task intake. No file changes are allowed.",
    "Do not run project verification commands unless the request explicitly requires them.",
    "Do not use update_plan for this stage. Inspect the repo as needed, then return the final artifact directly.",
    "Produce a structured task specification with goals, constraints, assumptions, and acceptance criteria.",
  ],
  design: [
    "Workspace is design-only for this stage. No file changes are allowed.",
    "Do not run project verification commands unless the request explicitly requires them.",
    "Do not use update_plan for this stage. Inspect the repo as needed, then return the final artifact directly.",
    "Produce a structured design document with architecture outline, interfaces, risks, tradeoffs, and validation strategy.",
  ],
  breakdown: [
    "Workspace is breakdown-only for this stage. No file changes are allowed.",
    "Do not run project verification commands unless the request explicitly requires them.",
    "Do not use update_plan for this stage. Inspect the repo as needed, then return the final artifact directly.",
    "Produce an implementation breakdown as an ordered checklist of small executable tasks, not a design narrative.",
    "Ground every task in the approved design, current repository state, and concrete repo paths or symbols you inspected.",
    "Every task must be independently executable, reviewable, and scoped to fewer than 500 changed lines.",
    "Include explicit dependencies, acceptance criteria, expected changes, validation commands, and risk notes for each task.",
    "Do not emit section headings as tasks and do not emit prose-only summaries in place of task records.",
    "Return strict JSON with this exact top-level shape: {\"structured\": <BreakdownDoc>, \"rendered\": <Markdown checklist>}.",
    BREAKDOWN_SCHEMA_INSTRUCTION,
    "Use exactly those property names. Do not rename keys, omit required fields, or add extra fields.",
    "The rendered markdown must use one checklist item per task in execution order, for example '- [ ] B1. Add input normalization in src/foo.ts'.",
  ],
  "issue-generation": [
    "Workspace is issue-generation only. No file changes are allowed.",
    "Do not run project verification commands unless the request explicitly requires them.",
    "Do not use update_plan for this stage. Inspect the repo as needed, then return the final artifact directly.",
    "Generate executable issue specs directly from the approved breakdown tasks. Do not infer issues from document headings.",
    "Every issue must link to one or more approved breakdown task ids, preserve the breakdown execution order, and reference concrete repo paths or symbols.",
    "Do not invent standalone issues outside the approved breakdown or drop any approved breakdown tasks.",
    "Return strict JSON with this exact top-level shape: {\"structured\": <IssueSpecDoc>, \"rendered\": <Markdown summary>}.",
    ISSUE_SPEC_SCHEMA_INSTRUCTION,
    "Use exactly those property names. Do not rename keys, omit required fields, or add extra fields.",
  ],
  triage: [
    "Workspace is analysis-only for triage. No file changes are allowed.",
    "Do not run project verification commands unless the request explicitly requires them.",
    "Do not use update_plan for this stage. Inspect the repo as needed, then return the final artifact directly.",
    "Produce a concise triage report covering issue understanding, impact area, risks, unknowns, and next step.",
  ],
  plan: [
    "Workspace is planning-only for plan. No file changes are allowed.",
    "Do not run project verification commands unless the request explicitly requires them.",
    "Do not use update_plan for this stage. Inspect the repo as needed, then return the final artifact directly.",
    "Produce a concise implementation plan covering steps, affected files/components, test strategy, and rollback/risk notes.",
  ],
  "test-plan": [
    "Workspace is planning-only for test-plan. No file changes are allowed.",
    "Do not run project verification commands unless the request explicitly requires them.",
    "Do not use update_plan for this stage. Inspect the repo as needed, then return the final artifact directly.",
    "Produce a test plan with scenarios, edge cases, regression risks, required tests, and expected outcomes.",
  ],
  implement: [
    "Implement the requested change in the current workspace, then summarize the changed files, edits, and blockers.",
  ],
  review: [
    "Workspace is review-only for this stage. No file changes are allowed.",
    "Do not use update_plan for this stage. Inspect the current workspace changes as needed, then return the final review artifact directly.",
    "Produce a direct review report with either exactly `No defects found.` or one section per defect using the format `Severity: <low|medium|high|critical>` plus a concrete fix recommendation.",
  ],
  repair: [
    "Apply repairs for the current issue, address the review findings, and summarize fixes applied plus remaining concerns.",
  ],
  completion: [
    "Workspace is completion-only for this stage. No file changes are allowed.",
    "Do not run project verification commands unless the request explicitly requires them.",
    "Do not use update_plan for this stage. Inspect the repo as needed, then return the final artifact directly.",
    "Produce a workflow summary covering completed issues, remaining risks, key decisions, and artifact chain highlights.",
  ],
};

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

const DEFAULT_CONTEXT_SECTION_ORDER: readonly ContextSectionId[] = [
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

const CONTEXT_SECTION_ORDERS: Partial<Record<TaskExecutionRequest["taskType"], readonly ContextSectionId[]>> = {
  breakdown: [
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
  ],
  "issue-generation": [
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
  ],
  implement: [
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
  ],
  review: [
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
  ],
  repair: [
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
  ],
};

function buildContextSectionOrder(taskType: TaskExecutionRequest["taskType"]): readonly ContextSectionId[] {
  return CONTEXT_SECTION_ORDERS[taskType] ?? DEFAULT_CONTEXT_SECTION_ORDER;
}

export function buildTaskQuery(
  request: TaskExecutionRequest,
  resolvedSkills: ResolvedRequestedSkill[] = [],
): string {
  const sections = [
    ...buildTaskHeaderSections(request),
    ...buildOrderedContextSections(request, resolvedSkills),
  ];
  sections.push(...buildTaskTypeInstructions(request));
  sections.push(buildResponseFormatInstruction(request.taskType));
  return sections.join("\n\n");
}

function buildTaskHeaderSections(request: TaskExecutionRequest): string[] {
  const primaryRepository = primaryRepositoryForRequest(request);
  return [
    `Task type: ${request.taskType}`,
    `${workItemLabel(request)}: ${request.workItem.title ?? request.workItem.externalId}`,
    request.workItem.kind === "local-task" ? "Task source: local/manual" : `Task source: ${request.workItem.kind}`,
    primaryRepository ? `Primary repository: ${primaryRepository.alias} (${primaryRepository.repoRoot})` : undefined,
    formatSection("Repository context", buildRepositoryContext(request)),
    formatSection("Reviewable context", buildReviewableContext(request.reviewable)),
  ].filter((section): section is string => Boolean(section));
}

function workItemLabel(request: TaskExecutionRequest): string {
  return request.workItem.kind === "local-task" ? "Task" : "Issue";
}

function buildOrderedContextSections(
  request: TaskExecutionRequest,
  resolvedSkills: ResolvedRequestedSkill[],
): string[] {
  const contextSections = buildContextSections(request, resolvedSkills);
  return buildContextSectionOrder(request.taskType)
    .map((sectionId) => contextSections[sectionId])
    .filter((section): section is string => Boolean(section));
}

function buildContextSections(
  request: TaskExecutionRequest,
  resolvedSkills: ResolvedRequestedSkill[],
): Partial<Record<ContextSectionId, string>> {
  const classifiedComments = classifyContextComments(request.context.comments);
  return {
    summary: formatSection("Summary", request.context.summary),
    issueBody: formatSection("Issue body", request.context.issueBody),
    designArtifact: renderWorkflowCommentSection("design-artifact", classifiedComments.workflow["design-artifact"]),
    breakdownArtifact: renderWorkflowCommentSection("breakdown-artifact", classifiedComments.workflow["breakdown-artifact"]),
    issueSpecArtifact: renderWorkflowCommentSection("issue-spec-artifact", classifiedComments.workflow["issue-spec-artifact"]),
    implementationSummary: renderWorkflowCommentSection("implementation-summary", classifiedComments.workflow["implementation-summary"]),
    reviewReport: renderWorkflowCommentSection("review-report", classifiedComments.workflow["review-report"]),
    issueUnit: renderIssueUnitSection(request.issueUnit),
    contextBundle: renderContextBundleSection(request.contextBundle),
    focusFiles: renderFocusFilesSection(request),
    comments: formatSection("Comments", classifiedComments.generic.map(formatGenericComment)),
    skills: renderResolvedSkillsSection(resolvedSkills),
    extraInstructions: renderExtraInstructionsSection(request),
  };
}

function renderFocusFilesSection(request: TaskExecutionRequest): string | undefined {
  const files = dedupePreservingOrder(request.context.changedFilesHint).map((filePath) => `- ${filePath}`);
  return formatSection("Focus files", files);
}

function renderResolvedSkillsSection(resolvedSkills: ResolvedRequestedSkill[]): string | undefined {
  if (resolvedSkills.length === 0) return undefined;
  const body = resolvedSkills
    .map((skill) => `## ${skill.name}\nSource: ${skill.source}\n${skill.instructions}`)
    .join("\n\n");
  return formatSection("Requested skills", body);
}

function renderExtraInstructionsSection(request: TaskExecutionRequest): string | undefined {
  return request.context.extraInstructions?.length
    ? formatSection("Extra instructions", request.context.extraInstructions)
    : undefined;
}

function buildResponseFormatInstruction(taskType: TaskExecutionRequest["taskType"]): string {
  return isStrictStructuredArtifactTask(taskType)
    ? "Return only the JSON object without code fences or surrounding commentary."
    : "Return plain Markdown without code fences around the entire response.";
}

function isStrictStructuredArtifactTask(taskType: TaskExecutionRequest["taskType"]): boolean {
  return taskType === "breakdown" || taskType === "issue-generation";
}

function buildTaskTypeInstructions(request: TaskExecutionRequest): string[] {
  if (request.taskType === "verify") {
    return [
      `Verification commands will run outside the model. Summarize the verification outcome and any follow-up actions based on these commands:\n${request.constraints.verifyCommands?.join("\n") ?? "No commands provided."}`,
    ];
  }
  return [...(TASK_TYPE_INSTRUCTIONS[request.taskType] ?? [])];
}
