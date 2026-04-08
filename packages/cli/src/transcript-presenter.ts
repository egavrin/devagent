import type {
  ToolAfterEvent,
  ToolBeforeEvent,
  ToolMessageEvent,
  ContextCompactingEvent,
  ContextCompactedEvent,
  ApprovalRequestEvent,
  ApprovalResponseEvent,
} from "@devagent/runtime";
import type {
  ToolFileChangePreview,
  ToolCommandResultMetadata,
  ToolValidationResultMetadata,
} from "@devagent/runtime";
import type { PlanStep } from "./tui/PlanView.js";
import { formatFileEditSummary } from "./file-edit-presentation.js";

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

export type TranscriptPartKind = TranscriptPart["kind"];
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

export function extractToolPreviewForTranscript(toolName: string, output: string): string | undefined {
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
  const parts: TranscriptPart[] = [{
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
  }];

  if (event.result.success) {
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

  if (commandResult) {
    parts.push({
      kind: "command-result",
      data: presentCommandResult(event.callId, commandResult),
    });
  }

  if (validationResult) {
    parts.push({
      kind: "validation-result",
      data: presentValidationResult(event.callId, validationResult),
    });

    if (validationResult.diagnosticErrors.length > 0) {
      parts.push({
        kind: "diagnostic-list",
        data: {
          toolId: event.callId,
          title: `diagnostics (${validationResult.diagnosticErrors.length})`,
          diagnostics: validationResult.diagnosticErrors.slice(0, MAX_PRESENTED_DIAGNOSTICS),
          hiddenCount: Math.max(0, validationResult.diagnosticErrors.length - MAX_PRESENTED_DIAGNOSTICS),
        },
      });
    }
  }

  return parts;
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

export function makeProgressPart(data: PresentedProgress): TranscriptPart {
  return { kind: "progress", data };
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

export function makeUserPart(text: string): TranscriptPart {
  return { kind: "user", data: { text } };
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
  if (
    typeof record["command"] !== "string" ||
    typeof record["cwd"] !== "string" ||
    typeof record["stdoutPreview"] !== "string" ||
    typeof record["stderrPreview"] !== "string" ||
    typeof record["stdoutTruncated"] !== "boolean" ||
    typeof record["stderrTruncated"] !== "boolean" ||
    typeof record["timedOut"] !== "boolean" ||
    typeof record["warningOnly"] !== "boolean"
  ) {
    return undefined;
  }

  const exitCode = record["exitCode"];
  if (!(typeof exitCode === "number" || exitCode === null)) {
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

function extractValidationResultMetadata(
  metadata: Record<string, unknown> | undefined,
): ToolValidationResultMetadata | undefined {
  const value = metadata?.["validationResult"];
  if (!value || typeof value !== "object") return undefined;
  const record = value as Record<string, unknown>;
  if (
    typeof record["passed"] !== "boolean" ||
    !Array.isArray(record["diagnosticErrors"]) ||
    !record["diagnosticErrors"].every((item) => typeof item === "string") ||
    !(typeof record["testPassed"] === "boolean" || record["testPassed"] === null)
  ) {
    return undefined;
  }

  const testSummaryValue = record["testSummary"];
  const testSummary = testSummaryValue && typeof testSummaryValue === "object"
    && typeof (testSummaryValue as Record<string, unknown>)["framework"] === "string"
    && typeof (testSummaryValue as Record<string, unknown>)["passed"] === "number"
    && typeof (testSummaryValue as Record<string, unknown>)["failed"] === "number"
    && Array.isArray((testSummaryValue as Record<string, unknown>)["failureMessages"])
    && ((testSummaryValue as Record<string, unknown>)["failureMessages"] as ReadonlyArray<unknown>)
      .every((item) => typeof item === "string")
    ? {
      framework: (testSummaryValue as Record<string, unknown>)["framework"] as string,
      passed: (testSummaryValue as Record<string, unknown>)["passed"] as number,
      failed: (testSummaryValue as Record<string, unknown>)["failed"] as number,
      failureMessages: (testSummaryValue as Record<string, unknown>)["failureMessages"] as ReadonlyArray<string>,
    }
    : undefined;

  return {
    passed: record["passed"],
    diagnosticErrors: record["diagnosticErrors"] as ReadonlyArray<string>,
    testPassed: record["testPassed"] as boolean | null,
    ...(testSummary ? { testSummary } : {}),
    ...(typeof record["testOutputPreview"] === "string" || record["testOutputPreview"] === null
      ? { testOutputPreview: record["testOutputPreview"] as string | null }
      : {}),
    ...(typeof record["baselineFiltered"] === "number"
      ? { baselineFiltered: record["baselineFiltered"] as number }
      : {}),
  };
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
