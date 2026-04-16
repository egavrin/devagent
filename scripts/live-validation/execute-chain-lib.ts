import { PROTOCOL_VERSION, type BreakdownDoc, type IssueSpecDoc, type IssueUnitRef, type TaskExecutionRequest } from "@devagent-sdk/types";

export const EXECUTE_CHAIN_STAGES = [
  "design",
  "breakdown",
  "issue-generation",
  "implement",
  "review",
  "repair",
] as const;

export type ExecuteChainStage = typeof EXECUTE_CHAIN_STAGES[number];

export interface ExecuteChainArtifactContext {
  readonly designDoc?: string;
  readonly breakdownDoc?: string;
  readonly breakdownStructured?: BreakdownDoc;
  readonly issueSpec?: string;
  readonly issueStructured?: IssueSpecDoc;
  readonly implementationSummary?: string;
  readonly reviewReport?: string;
}

interface ExecuteChainRequestOptions {
  readonly stage: ExecuteChainStage;
  readonly workspaceRoot: string;
  readonly sourceRepoRoot: string;
  readonly provider: string;
  readonly model: string;
  readonly taskIdPrefix: string;
  readonly changedFilesHint?: ReadonlyArray<string>;
  readonly priorArtifacts: ExecuteChainArtifactContext;
}

const EXECUTE_CHAIN_REPO_ID = "repo-1";
const EXECUTE_CHAIN_WORKSPACE_ID = "workspace-1";
const EXECUTE_CHAIN_TITLE = "Document the bytecode validation workflow in README.md";

function buildStageSummary(stage: ExecuteChainStage): string {
  switch (stage) {
    case "design":
      return "Produce a design doc for a minimal docs-only README.md change that explains the bytecode validation workflow: assembler input, abc output, verifier responsibilities, and where docs/bc_verification fits.";
    case "breakdown":
      return "Turn the approved design into the smallest executable docs-only breakdown, preferring a single README.md task unless repo inspection proves a second file is required.";
    case "issue-generation":
      return "Generate executable issue specs directly from the approved breakdown, preserving task order and keeping the implementation docs-only when the breakdown allows it.";
    case "implement":
      return "Implement the approved docs-only issue in the current workspace. Make a small, human-readable README.md update that explains the bytecode validation workflow.";
    case "review":
      return "Review the current workspace changes against the approved issue spec. Focus on correctness, clarity, and whether README.md accurately describes the assembler and verifier workflow.";
    case "repair":
      return "Address concrete review findings if any were reported. If review reported no defects, leave the workspace unchanged and produce a final summary.";
  }
}

function buildStageIssueBody(stage: ExecuteChainStage): string {
  switch (stage) {
    case "design":
      return "Ground the design in README.md, docs/bc_verification, assembler-related repo areas, and verifier-related repo areas. The final implementation should remain docs-only and should prefer editing README.md only.";
    case "breakdown":
      return "Use the approved design artifact as the source of truth. Keep the breakdown small enough to complete in one implementation pass and prefer README.md-only work if repo inspection supports it.";
    case "issue-generation":
      return "Use the approved breakdown as the source of truth. Prefer a single issue if the breakdown stays within one README.md slice.";
    case "implement":
      return "Implement only the approved docs issue. Keep the edit minimal, preserve existing README structure, and explain the assembler -> verifier -> docs/bc_verification flow without inventing unsupported behavior.";
    case "review":
      return "Review the current workspace diff only. Report concrete defects only, or `No defects found.` if the README change is accurate and coherent.";
    case "repair":
      return "Use the review artifact as the source of truth for repairs. If it says `No defects found.`, do not make additional edits and just summarize the state.";
  }
}

function buildStageExtraInstructions(stage: ExecuteChainStage): string[] {
  switch (stage) {
    case "design":
      return [
        "Aim for a single small implementation slice that can be completed in one issue.",
        "Call out assembler responsibilities and verifier responsibilities separately.",
        "Make README.md the preferred implementation target unless inspection proves a second docs file is necessary.",
      ];
    case "breakdown":
      return [
        "Prefer exactly one executable breakdown task if README.md-only is sufficient.",
        "Keep the task under 150 changed lines and docs-only.",
      ];
    case "issue-generation":
      return [
        "Prefer exactly one issue if the approved breakdown contains one task.",
        "Carry over README.md grounding paths and implementation notes directly from the breakdown.",
      ];
    case "implement":
      return [
        "Prefer editing README.md only.",
        "Add a short subsection or paragraph that explains the bytecode validation workflow in plain language.",
        "Mention docs/bc_verification explicitly if the repo inspection supports that reference.",
      ];
    case "review":
      return [
        "Focus on factual drift, missing caveats, and broken README structure.",
        "Do not propose speculative improvements unrelated to the approved issue.",
      ];
    case "repair":
      return [
        "Only fix defects explicitly called out by the review report.",
        "If no defects were reported, keep the workspace unchanged.",
      ];
  }
}

function buildStageComments(
  stage: ExecuteChainStage,
  priorArtifacts: ExecuteChainArtifactContext,
): TaskExecutionRequest["context"]["comments"] {
  const comments: Array<{ author?: string; body: string }> = [];
  if (stage !== "design" && priorArtifacts.designDoc) {
    comments.push({
      author: "design-artifact",
      body: `Approved design artifact:\n\n${priorArtifacts.designDoc}`,
    });
  }
  if ((stage === "issue-generation" || stage === "implement") && priorArtifacts.breakdownDoc) {
    comments.push({
      author: "breakdown-artifact",
      body: `Approved breakdown artifact:\n\n${priorArtifacts.breakdownDoc}`,
    });
  }
  if ((stage === "implement" || stage === "review" || stage === "repair") && priorArtifacts.issueSpec) {
    comments.push({
      author: "issue-spec-artifact",
      body: `Approved issue spec artifact:\n\n${priorArtifacts.issueSpec}`,
    });
  }
  if ((stage === "review" || stage === "repair") && priorArtifacts.implementationSummary) {
    comments.push({
      author: "implementation-summary",
      body: `Implementation summary artifact:\n\n${priorArtifacts.implementationSummary}`,
    });
  }
  if (stage === "repair" && priorArtifacts.reviewReport) {
    comments.push({
      author: "review-report",
      body: `Review report artifact:\n\n${priorArtifacts.reviewReport}`,
    });
  }
  return comments.length > 0 ? comments : undefined;
}

export function extractIssueUnitFromIssueSpec(issueSpec: IssueSpecDoc): IssueUnitRef {
  const firstIssue = issueSpec.issues[0];
  if (!firstIssue) {
    throw new Error("Issue spec did not contain any issues.");
  }
  return {
    id: firstIssue.id,
    title: firstIssue.title,
    sequence: 1,
    dependencyIds: [...firstIssue.dependencies],
    acceptanceCriteria: [...firstIssue.acceptanceCriteria],
    linkedArtifactVersionIds: [...firstIssue.linkedBreakdownTaskIds],
  };
}

export function buildExecuteChainRequest(options: ExecuteChainRequestOptions): TaskExecutionRequest {
  const comments = buildStageComments(options.stage, options.priorArtifacts);
  const issueUnit = options.stage === "implement" && options.priorArtifacts.issueStructured
    ? extractIssueUnitFromIssueSpec(options.priorArtifacts.issueStructured)
    : undefined;

  const contextBundle = options.stage === "implement" || options.stage === "review" || options.stage === "repair"
    ? {
        id: `bundle-${options.stage}`,
        artifactVersionIds: [
          ...(options.priorArtifacts.breakdownStructured ? ["breakdown-structured"] : []),
          ...(options.priorArtifacts.issueStructured ? ["issue-structured"] : []),
        ],
        summary: "Chained execute flow using approved design, breakdown, and issue-spec artifacts from earlier stages.",
      }
    : undefined;

  const changedFilesHint = options.stage === "implement"
    ? options.priorArtifacts.issueStructured?.issues[0]?.grounding.repoPaths ?? options.changedFilesHint
    : options.changedFilesHint;

  return {
    protocolVersion: PROTOCOL_VERSION,
    taskId: `${options.taskIdPrefix}-${options.stage}`,
    taskType: options.stage,
    workspaceRef: {
      id: EXECUTE_CHAIN_WORKSPACE_ID,
      name: "execute-chain-workspace",
      provider: "local",
      primaryRepositoryId: EXECUTE_CHAIN_REPO_ID,
    },
    repositories: [{
      id: EXECUTE_CHAIN_REPO_ID,
      workspaceId: EXECUTE_CHAIN_WORKSPACE_ID,
      alias: "primary",
      name: "workspace",
      repoRoot: options.workspaceRoot,
      repoFullName: "arkcompiler_runtime_core_docs",
      defaultBranch: "main",
      provider: "local",
    }],
    workItem: {
      id: `item-${options.stage}`,
      kind: "local-task",
      externalId: `execute-chain-${options.stage}`,
      title: EXECUTE_CHAIN_TITLE,
      repositoryId: EXECUTE_CHAIN_REPO_ID,
    },
    execution: {
      primaryRepositoryId: EXECUTE_CHAIN_REPO_ID,
      repositories: [{
        repositoryId: EXECUTE_CHAIN_REPO_ID,
        alias: "primary",
        sourceRepoPath: options.sourceRepoRoot,
        workBranch: "devagent/live/execute-chain",
        isolation: "git-worktree",
      }],
    },
    targetRepositoryIds: [EXECUTE_CHAIN_REPO_ID],
    executor: {
      executorId: "devagent",
      provider: options.provider,
      model: options.model,
      approvalMode: "full-auto",
    },
    constraints: {
      allowNetwork: true,
      maxIterations: options.stage === "implement" || options.stage === "repair" ? 20 : 16,
    },
    ...(issueUnit ? { issueUnit } : {}),
    ...(contextBundle ? { contextBundle } : {}),
    capabilities: {
      canSyncTasks: true,
      canCreateTask: true,
      canComment: true,
      canReview: true,
      canMerge: true,
      canOpenReviewable: true,
    },
    context: {
      summary: buildStageSummary(options.stage),
      issueBody: buildStageIssueBody(options.stage),
      ...(comments ? { comments } : {}),
      ...(changedFilesHint && changedFilesHint.length > 0 ? { changedFilesHint: [...changedFilesHint] } : {}),
      extraInstructions: buildStageExtraInstructions(options.stage),
    },
    expectedArtifacts: [artifactKindForChainStage(options.stage)],
  };
}

function artifactKindForChainStage(stage: ExecuteChainStage): TaskExecutionRequest["expectedArtifacts"][number] {
  switch (stage) {
    case "design":
      return "design-doc";
    case "breakdown":
      return "breakdown-doc";
    case "issue-generation":
      return "issue-spec";
    case "implement":
      return "implementation-summary";
    case "review":
      return "review-report";
    case "repair":
      return "final-summary";
  }
}

export function artifactFileNamesForChainStage(stage: ExecuteChainStage): string[] {
  switch (stage) {
    case "design":
      return ["design-doc.md"];
    case "breakdown":
      return ["breakdown-doc.md", "breakdown-doc.json"];
    case "issue-generation":
      return ["issue-spec.md", "issue-spec.json"];
    case "implement":
      return ["implementation-summary.md"];
    case "review":
      return ["review-report.md"];
    case "repair":
      return ["final-summary.md"];
  }
}
