/**
 * Workflow contract types — shared interface between DevAgent (stage runtime)
 * and DevAgent-Hub (workflow control plane).
 *
 * These types define the typed input/output for each workflow phase,
 * valid phase names, approval modes, reasoning levels, and exit codes.
 */

// ─── Phase Names ─────────────────────────────────────────────

export const WORKFLOW_PHASES = [
  "triage",
  "plan",
  "implement",
  "verify",
  "review",
  "repair",
  "gate",
] as const;

export type WorkflowPhase = (typeof WORKFLOW_PHASES)[number];

export function isValidPhase(phase: string): phase is WorkflowPhase {
  return (WORKFLOW_PHASES as readonly string[]).includes(phase);
}

// ─── Approval Modes (canonical) ──────────────────────────────

export const WORKFLOW_APPROVAL_MODES = [
  "suggest",
  "auto-edit",
  "full-auto",
] as const;

export type WorkflowApprovalMode = (typeof WORKFLOW_APPROVAL_MODES)[number];

export function isValidApprovalMode(mode: string): mode is WorkflowApprovalMode {
  return (WORKFLOW_APPROVAL_MODES as readonly string[]).includes(mode);
}

// ─── Reasoning Levels ────────────────────────────────────────

export const REASONING_LEVELS = ["low", "medium", "high", "xhigh"] as const;

export type ReasoningLevel = (typeof REASONING_LEVELS)[number];

export function isValidReasoningLevel(level: string): level is ReasoningLevel {
  return (REASONING_LEVELS as readonly string[]).includes(level);
}

// ─── Exit Codes ──────────────────────────────────────────────

export const EXIT_CODE = {
  SUCCESS: 0,
  PHASE_FAILED: 1,
  INVALID_ARGS: 2,
} as const;

// ─── Phase Input Types ───────────────────────────────────────

export interface TriageInput {
  issueNumber: number;
  title: string;
  body: string;
  labels: string[];
  author: string;
}

export interface PlanInput {
  issueNumber: number;
  title: string;
  body: string;
  labels: string[];
  author: string;
  triageReport?: TriageOutput;
}

export interface ImplementInput {
  issueNumber: number;
  title: string;
  body: string;
  acceptedPlan: PlanOutput;
}

export interface VerifyInput {
  commands: string[];
  changedFiles?: string[];
}

export interface ReviewInput {
  issueNumber: number;
  prNumber?: number | null;
  branch?: string | null;
  diff?: string;
  ciChecks?: CICheck[];
  reviewComments?: ReviewComment[];
}

export interface RepairInput {
  round: number;
  issueNumber: number;
  prNumber?: number | null;
  findings: ReviewFinding[];
  ciFailures?: string[];
}

export interface GateInput {
  sourcePhase: "triage" | "plan" | "implement";
  issueNumber: number;
  stageOutput: Record<string, unknown>;
}

export type PhaseInput =
  | TriageInput
  | PlanInput
  | ImplementInput
  | VerifyInput
  | ReviewInput
  | RepairInput
  | GateInput;

// ─── Phase Output Types ──────────────────────────────────────

export interface TriageOutput {
  summary: string;
  complexity: "trivial" | "small" | "medium" | "large" | "epic";
  suggestedLabels: string[];
  suggestedAssignee?: string;
  blockers?: string[];
  relatedFiles?: string[];
}

export interface PlanOutput {
  summary: string;
  steps: PlanStep[];
  filesToCreate: string[];
  filesToModify: string[];
  testStrategy: string;
  risks: string[];
}

export interface ImplementOutput {
  summary: string;
  changedFiles: string[];
  suggestedCommitMessage: string;
  diffSummary: string;
}

export interface VerifyOutput {
  summary: string;
  passed: boolean;
  results: VerifyCommandResult[];
}

export interface ReviewOutput {
  summary: string;
  verdict: "pass" | "block";
  findings: ReviewFinding[];
  blockingCount: number;
}

export interface RepairOutput {
  summary: string;
  fixedFindings: string[];
  remainingFindings: number;
  verificationPassed: boolean;
  changedFiles: string[];
}

export interface GateOutput {
  summary: string;
  verdict: "pass" | "block";
  findings: ReviewFinding[];
  blockingCount: number;
  confidence: number;
}

export type PhaseOutput =
  | TriageOutput
  | PlanOutput
  | ImplementOutput
  | VerifyOutput
  | ReviewOutput
  | RepairOutput
  | GateOutput;

// ─── Supporting Types ────────────────────────────────────────

export interface PlanStep {
  description: string;
  file?: string;
  type: "create" | "modify" | "delete" | "test" | "config";
}

export interface VerifyCommandResult {
  command: string;
  exitCode: number;
  stdout: string;
  stderr: string;
  passed: boolean;
}

export interface ReviewFinding {
  file: string;
  line?: number;
  severity: "critical" | "major" | "minor" | "suggestion";
  message: string;
  category: string;
}

export interface ReviewComment {
  author: string;
  body: string;
  file?: string;
  line?: number;
}

export interface CICheck {
  name: string;
  status: "success" | "failure" | "pending";
  url?: string;
}

// ─── Workflow Run Args ───────────────────────────────────────

export interface WorkflowRunArgs {
  phase: WorkflowPhase;
  inputPath: string;
  outputPath: string;
  eventsPath: string;
  repoPath: string;
  provider?: string;
  model?: string;
  maxIterations?: number;
  approvalMode?: WorkflowApprovalMode;
  reasoning?: ReasoningLevel;
}

// ─── Runner Description ──────────────────────────────────────

export interface RunnerDescription {
  version: string;
  supportedPhases: WorkflowPhase[];
  availableProviders: string[];
  supportedApprovalModes: WorkflowApprovalMode[];
  supportedReasoningLevels: ReasoningLevel[];
}
