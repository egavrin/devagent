/**
 * Versioned phase result schemas for headless workflow execution.
 */

export const WORKFLOW_SCHEMA_VERSION = 1;

export type WorkflowPhase =
  | "triage"
  | "plan"
  | "implement"
  | "verify"
  | "review"
  | "repair";

export interface PhaseResult<T> {
  schemaVersion: number;
  phase: WorkflowPhase;
  timestamp: string;
  durationMs: number;
  result: T;
  summary: string; // Markdown summary
}

export interface TriageReport {
  issueId: string;
  title: string;
  acceptanceCriteria: string[];
  risks: string[];
  missingContext: string[];
  suggestedLabels: string[];
  complexity: "trivial" | "small" | "medium" | "large" | "epic";
  duplicateSignals: string[];
}

export interface PlanDraft {
  issueId: string;
  steps: PlanStep[];
  affectedFiles: string[];
  testStrategy: string;
  rollbackRisks: string[];
  estimatedPhases: number;
}

export interface PlanStep {
  order: number;
  description: string;
  files: string[];
  tests: string[];
  dependencies: number[];
}

export interface ExecutionReport {
  issueId: string;
  planStepsCompleted: number;
  planStepsTotal: number;
  filesModified: string[];
  filesCreated: string[];
  testsAdded: string[];
  iterations: number;
  cost: { inputTokens: number; outputTokens: number; totalUsd: number };
}

export interface VerificationReport {
  commands: VerificationCommand[];
  allPassed: boolean;
  failingSummary: string | null;
}

export interface VerificationCommand {
  command: string;
  exitCode: number;
  passed: boolean;
  stdout: string;
  stderr: string;
  durationMs: number;
}

export interface ReviewReport {
  findings: ReviewFinding[];
  blockingCount: number;
  warningCount: number;
  infoCount: number;
  verdict: "pass" | "warn" | "block";
}

export interface ReviewFinding {
  severity: "blocking" | "warning" | "info";
  file: string;
  line?: number;
  message: string;
  suggestion?: string;
}

export interface RepairReport {
  round: number;
  inputFindings: number;
  fixedFindings: number;
  remainingFindings: number;
  filesModified: string[];
  verificationPassed: boolean;
}
