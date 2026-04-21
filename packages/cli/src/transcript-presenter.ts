import { formatFileEditSummary } from "./file-edit-presentation.js";
import type { PlanStep } from "./tui/PlanView.js";
import type {
  ToolAfterEvent,
  ToolBeforeEvent,
  ToolMessageEvent,
  ContextCompactingEvent,
  ContextCompactedEvent,
  ApprovalRequestEvent,
  ApprovalResponseEvent,

  ToolFileChangePreview,
  ToolCommandResultMetadata,
  ToolValidationResultMetadata} from "@devagent/runtime";

export interface PresentedToolEvent {
  readonly id: string;
  readonly name: string;
  readonly summary: string;
  readonly iteration: number;
  readonly maxIterations: number;
  readonly status: "running" | "success" | "error";
  readonly durationMs?: number;
  readonly error?: string;
  readonly preview?: string;
}

export interface PresentedToolGroup {
  readonly name: string;
  readonly count: number;
  readonly summaries: ReadonlyArray<string>;
  readonly iteration: number;
  readonly maxIterations: number;
  readonly status: "running" | "success" | "error";
  readonly totalDurationMs?: number;
}

export interface PresentedFileEdit {
  readonly toolId: string;
  readonly fileEdit: ToolFileChangePreview;
  readonly summary: string;
}

export interface PresentedStatus {
  readonly title: string;
  readonly lines: ReadonlyArray<string>;
  readonly tone?: "info" | "success" | "warning" | "error";
}

export interface PresentedProgress {
  readonly title: string;
  readonly detail?: string;
}

export interface PresentedApproval {
  readonly id: string;
  readonly action: string;
  readonly toolName: string;
  readonly details: string;
  readonly status: "pending";
}

export interface PresentedReasoning {
  readonly text: string;
}

export interface PresentedError {
  readonly message: string;
  readonly code: string;
}

export interface PresentedFinalOutput {
  readonly text: string;
}

export interface PresentedTurnSummary {
  readonly iterations: number;
  readonly toolCalls: number;
  readonly cost: number;
  readonly elapsedMs: number;
}

export interface PresentedUser {
  readonly text: string;
}

export interface PresentedInfo {
  readonly title: string;
  readonly lines: ReadonlyArray<string>;
}

export interface PresentedFileEditOverflow {
  readonly hiddenCount: number;
}

export interface PresentedCommandResult {
  readonly toolId: string;
  readonly command: string;
  readonly cwd: string;
  readonly status: "success" | "warning" | "error" | "timeout";
  readonly statusLine: string;
  readonly stdoutPreview: string;
  readonly stderrPreview: string;
  readonly stdoutTruncated: boolean;
  readonly stderrTruncated: boolean;
}

export interface PresentedValidationResult {
  readonly toolId: string;
  readonly passed: boolean;
  readonly summary: string;
  readonly testSummaryLine?: string;
  readonly testOutputPreview?: string;
  readonly diagnosticCount: number;
}

export interface PresentedDiagnosticList {
  readonly toolId: string;
  readonly title: string;
  readonly diagnostics: ReadonlyArray<string>;
  readonly hiddenCount: number;
}

export type TranscriptPart =
  | { readonly kind: "tool"; readonly event: PresentedToolEvent }
  | { readonly kind: "tool-group"; readonly event: PresentedToolGroup }
  | { readonly kind: "file-edit"; readonly data: PresentedFileEdit }
  | { readonly kind: "file-edit-overflow"; readonly data: PresentedFileEditOverflow }
  | { readonly kind: "command-result"; readonly data: PresentedCommandResult }
  | { readonly kind: "validation-result"; readonly data: PresentedValidationResult }
  | { readonly kind: "diagnostic-list"; readonly data: PresentedDiagnosticList }
  | { readonly kind: "status"; readonly data: PresentedStatus }
  | { readonly kind: "progress"; readonly data: PresentedProgress }
  | { readonly kind: "approval"; readonly data: PresentedApproval }
  | { readonly kind: "reasoning"; readonly data: PresentedReasoning }
  | { readonly kind: "plan"; readonly data: ReadonlyArray<PlanStep> }
  | { readonly kind: "error"; readonly data: PresentedError }
  | { readonly kind: "final-output"; readonly data: PresentedFinalOutput }
  | { readonly kind: "turn-summary"; readonly data: PresentedTurnSummary }
  | { readonly kind: "user"; readonly data: PresentedUser }
  | { readonly kind: "info"; readonly data: PresentedInfo };

const MAX_PRESENTED_DIAGNOSTICS = 5;

export function summarizeToolParamsForTranscript(name: string, params: Record<string, unknown>): string {
  const path = params["path"] as string | undefined;
  if (path) return path;
  const command = params["command"] as string | undefined;
  if (command) return command.slice(0, 80);
  const pattern = params["pattern"] as string | undefined;
  if (pattern) return `"${pattern}"`;
  return "";
}

function extractToolPreviewForTranscript(toolName: string, output: string): string | undefined {
  if (!output || output.length < 10) return undefined;
  if (toolName === "search_files") {
    const match = output.match(/^(\d+) match/);
    if (match) return output.split("\n")[0]!.slice(0, 80);
  }
  if (toolName === "run_command") {
    const lines = output.split("\n").filter((line) => line.trim() && !line.startsWith("Exit code:"));
    if (lines.length > 0) return lines[0]!.trim().slice(0, 80);
  }
  if (toolName === "find_files") {
    const lines = output.split("\n").filter((line) => line.trim());
    if (lines.length > 0) return `${lines.length} file(s) found`;
  }
  return undefined;
}

export function presentToolBeforeEvent(
  event: ToolBeforeEvent,
  iteration: number,
  maxIterations: number,
): TranscriptPart {
  return {
    kind: "tool",
    event: {
      id: event.callId,
      name: event.name,
      summary: summarizeToolParamsForTranscript(event.name, event.params),
      iteration,
      maxIterations,
      status: "running",
    },
  };
}
export function presentToolAfterEvent(
  event: ToolAfterEvent,
  iteration: number,
  maxIterations: number,
): ReadonlyArray<TranscriptPart> {
  const commandResult = extractCommandResultMetadata(event.result.metadata);
  const validationResult = extractValidationResultMetadata(event.result.metadata);
  const parts: TranscriptPart[] = [
    buildToolAfterPart(event, iteration, maxIterations, commandResult, validationResult),
  ];

  if (event.result.success) {
    appendFileEditParts(parts, event);
  }

  if (commandResult) {
    parts.push({ kind: "command-result", data: presentCommandResult(event.callId, commandResult) });
  }

  if (validationResult) {
    appendValidationParts(parts, event.callId, validationResult);
  }

  return parts;
}

function buildToolAfterPart(
  event: ToolAfterEvent,
  iteration: number,
  maxIterations: number,
  commandResult: ToolCommandResultMetadata | undefined,
  validationResult: ToolValidationResultMetadata | undefined,
): TranscriptPart {
  return {
    kind: "tool",
    event: {
      id: event.callId,
      name: event.name,
      summary: "",
      iteration,
      maxIterations,
      status: event.result.success ? "success" : "error",
      durationMs: event.durationMs,
      error: event.result.error ?? undefined,
      preview: event.result.success && !commandResult && !validationResult
        ? extractToolPreviewForTranscript(event.name, event.result.output)
        : undefined,
    },
  };
}

function appendFileEditParts(parts: TranscriptPart[], event: ToolAfterEvent): void {
  for (const fileEdit of event.fileEdits ?? []) {
    parts.push({
      kind: "file-edit",
      data: {
        toolId: event.callId,
        fileEdit,
        summary: formatFileEditSummary(fileEdit),
      },
    });
  }
  if ((event.fileEditHiddenCount ?? 0) > 0) {
    parts.push({
      kind: "file-edit-overflow",
      data: { hiddenCount: event.fileEditHiddenCount ?? 0 },
    });
  }
}

function appendValidationParts(
  parts: TranscriptPart[],
  toolId: string,
  validationResult: ToolValidationResultMetadata,
): void {
  parts.push({ kind: "validation-result", data: presentValidationResult(toolId, validationResult) });

  if (validationResult.diagnosticErrors.length > 0) {
    parts.push({
      kind: "diagnostic-list",
      data: {
        toolId,
        title: `diagnostics (${validationResult.diagnosticErrors.length})`,
        diagnostics: validationResult.diagnosticErrors.slice(0, MAX_PRESENTED_DIAGNOSTICS),
        hiddenCount: Math.max(0, validationResult.diagnosticErrors.length - MAX_PRESENTED_DIAGNOSTICS),
      },
    });
  }
}

export function presentToolGroupEvent(event: PresentedToolGroup): TranscriptPart {
  return { kind: "tool-group", event };
}

export function presentContextCompactingEvent(event: ContextCompactingEvent): TranscriptPart {
  return {
    kind: "progress",
    data: {
      title: "Compacting context",
      detail: `~${Math.round(event.estimatedTokens / 1000)}k / ${Math.round(event.maxTokens / 1000)}k tokens`,
    },
  };
}

export function presentContextCompactedEvent(event: ContextCompactedEvent): TranscriptPart {
  return {
    kind: "status",
    data: {
      title: "context",
      lines: [
        `Compacted ${Math.round(event.tokensBefore / 1000)}k -> ${Math.round(event.estimatedTokens / 1000)}k tokens`,
      ],
      tone: "info",
    },
  };
}

export function presentApprovalRequestEvent(event: ApprovalRequestEvent): TranscriptPart {
  return {
    kind: "approval",
    data: {
      id: event.id,
      action: event.action,
      toolName: event.toolName,
      details: event.details,
      status: "pending",
    },
  };
}

export function presentApprovalResponseEvent(event: ApprovalResponseEvent): TranscriptPart {
  return {
    kind: "status",
    data: {
      title: "approval",
      lines: [event.approved ? "Approved" : "Denied"],
      tone: event.approved ? "success" : "warning",
    },
  };
}

export function presentSummaryToolMessage(event: ToolMessageEvent): TranscriptPart {
  let content = event.content;
  content = content.replace(/Subagent \S+ /, "");
  return {
    kind: "status",
    data: {
      title: "delegate",
      lines: [content],
      tone: "info",
    },
  };
}

export function makeStatusPart(data: PresentedStatus): TranscriptPart {
  return { kind: "status", data };
}

export function makePlanPart(steps: ReadonlyArray<PlanStep>): TranscriptPart {
  return { kind: "plan", data: steps };
}

export function makeErrorPart(data: PresentedError): TranscriptPart {
  return { kind: "error", data };
}

export function makeFinalOutputPart(text: string): TranscriptPart {
  return { kind: "final-output", data: { text } };
}

export function makeTurnSummaryPart(
  data: PresentedTurnSummary,
): Extract<TranscriptPart, { readonly kind: "turn-summary" }> {
  return { kind: "turn-summary", data };
}

export function makeInfoPart(title: string, lines: ReadonlyArray<string>): TranscriptPart {
  return { kind: "info", data: { title, lines } };
}
function extractCommandResultMetadata(
  metadata: Record<string, unknown> | undefined,
): ToolCommandResultMetadata | undefined {
  const value = metadata?.["commandResult"];
  if (!value || typeof value !== "object") return undefined;
  const record = value as Record<string, unknown>;
  const exitCode = record["exitCode"];
  if (!isCommandResultRecord(record) || !(typeof exitCode === "number" || exitCode === null)) {
    return undefined;
  }

  return {
    command: record["command"],
    cwd: record["cwd"],
    exitCode,
    timedOut: record["timedOut"],
    warningOnly: record["warningOnly"],
    stdoutPreview: record["stdoutPreview"],
    stderrPreview: record["stderrPreview"],
    stdoutTruncated: record["stdoutTruncated"],
    stderrTruncated: record["stderrTruncated"],
  };
}

function isCommandResultRecord(record: Record<string, unknown>): record is Record<string, unknown> & {
  command: string;
  cwd: string;
  stdoutPreview: string;
  stderrPreview: string;
  stdoutTruncated: boolean;
  stderrTruncated: boolean;
  timedOut: boolean;
  warningOnly: boolean;
} {
  return [
    typeof record["command"] === "string",
    typeof record["cwd"] === "string",
    typeof record["stdoutPreview"] === "string",
    typeof record["stderrPreview"] === "string",
    typeof record["stdoutTruncated"] === "boolean",
    typeof record["stderrTruncated"] === "boolean",
    typeof record["timedOut"] === "boolean",
    typeof record["warningOnly"] === "boolean",
  ].every(Boolean);
}

function extractValidationResultMetadata(
  metadata: Record<string, unknown> | undefined,
): ToolValidationResultMetadata | undefined {
  const value = metadata?.["validationResult"];
  if (!value || typeof value !== "object") return undefined;
  const record = value as Record<string, unknown>;
  if (!isValidationResultRecord(record)) {
    return undefined;
  }

  return {
    passed: record["passed"],
    diagnosticErrors: record["diagnosticErrors"],
    testPassed: record["testPassed"],
    ...buildOptionalValidationMetadata(record),
  };
}

function isValidationResultRecord(record: Record<string, unknown>): record is Record<string, unknown> & {
  passed: boolean;
  diagnosticErrors: ReadonlyArray<string>;
  testPassed: boolean | null;
} {
  return typeof record["passed"] === "boolean" &&
    Array.isArray(record["diagnosticErrors"]) &&
    record["diagnosticErrors"].every((item) => typeof item === "string") &&
    (typeof record["testPassed"] === "boolean" || record["testPassed"] === null);
}

function buildOptionalValidationMetadata(
  record: Record<string, unknown>,
): Partial<ToolValidationResultMetadata> {
  const testSummary = extractTestSummary(record["testSummary"]);
  return {
    ...(testSummary ? { testSummary } : {}),
    ...(typeof record["testOutputPreview"] === "string" || record["testOutputPreview"] === null
      ? { testOutputPreview: record["testOutputPreview"] }
      : {}),
    ...(typeof record["baselineFiltered"] === "number"
      ? { baselineFiltered: record["baselineFiltered"] }
      : {}),
  };
}

function extractTestSummary(value: unknown): ToolValidationResultMetadata["testSummary"] | undefined {
  if (!value || typeof value !== "object") return undefined;
  const record = value as Record<string, unknown>;
  if (!isTestSummaryRecord(record)) return undefined;
  return {
    framework: record["framework"],
    passed: record["passed"],
    failed: record["failed"],
    failureMessages: record["failureMessages"],
  };
}

function isTestSummaryRecord(record: Record<string, unknown>): record is {
  framework: string;
  passed: number;
  failed: number;
  failureMessages: ReadonlyArray<string>;
} {
  return typeof record["framework"] === "string" &&
    typeof record["passed"] === "number" &&
    typeof record["failed"] === "number" &&
    Array.isArray(record["failureMessages"]) &&
    record["failureMessages"].every((item) => typeof item === "string");
}

function presentCommandResult(toolId: string, metadata: ToolCommandResultMetadata): PresentedCommandResult {
  const status = metadata.timedOut
    ? "timeout"
    : metadata.warningOnly
      ? "warning"
      : metadata.exitCode === null || metadata.exitCode === 0
        ? "success"
        : "error";
  const statusLine = metadata.timedOut
    ? "Timed out"
    : metadata.warningOnly
      ? `Completed with warnings (exit ${metadata.exitCode ?? "?"})`
      : metadata.exitCode === null || metadata.exitCode === 0
        ? "Exited successfully"
        : `Exited with code ${metadata.exitCode}`;

  return {
    toolId,
    command: metadata.command,
    cwd: metadata.cwd,
    status,
    statusLine,
    stdoutPreview: metadata.stdoutPreview,
    stderrPreview: metadata.stderrPreview,
    stdoutTruncated: metadata.stdoutTruncated,
    stderrTruncated: metadata.stderrTruncated,
  };
}

function presentValidationResult(
  toolId: string,
  metadata: ToolValidationResultMetadata,
): PresentedValidationResult {
  const diagnosticCount = metadata.diagnosticErrors.length;
  return {
    toolId,
    passed: metadata.passed,
    summary: metadata.passed
      ? "Validation passed"
      : `Validation failed${diagnosticCount > 0 ? ` · ${diagnosticCount} diagnostic${diagnosticCount === 1 ? "" : "s"}` : ""}`,
    ...(metadata.testSummary
      ? {
        testSummaryLine: `${metadata.testSummary.framework}: ${metadata.testSummary.passed} passed, ${metadata.testSummary.failed} failed`,
      }
      : metadata.testPassed === false
        ? { testSummaryLine: "Tests failed" }
        : metadata.testPassed === true
          ? { testSummaryLine: "Tests passed" }
          : {}),
    ...(typeof metadata.testOutputPreview === "string" && metadata.testOutputPreview.length > 0
      ? { testOutputPreview: metadata.testOutputPreview }
      : {}),
    diagnosticCount,
  };
}
