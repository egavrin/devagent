import { formatDuration } from "@devagent/runtime";

import { bold, dim, red, truncate } from "./format-colors.js";
import type {
  DelegatedWorkSummary,
  SubagentErrorEvent,
  SubagentStartEvent,
  SubagentUpdateEvent,
} from "@devagent/runtime";

interface SessionSummaryData {
  readonly sessionId: string;
  readonly totalIterations: number;
  readonly totalToolCalls: number;
  readonly toolUsage: ReadonlyMap<string, number>;
  readonly filesChanged: ReadonlyArray<string>;
  readonly planSteps?: ReadonlyArray<{ description: string; status: string }>;
  readonly totalCost: number;
  readonly totalInputTokens: number;
  readonly totalOutputTokens: number;
  readonly elapsedMs: number;
  readonly completionReason: string;
  readonly delegatedWork?: DelegatedWorkSummary;
}

export function formatSessionSummary(data: SessionSummaryData): string {
  const lines: string[] = [];
  lines.push("");
  lines.push(bold("Session Summary"));
  lines.push(dim("─".repeat(50)));
  appendSessionCore(lines, data);
  appendSessionToolUsage(lines, data.toolUsage);
  appendSessionFilesChanged(lines, data.filesChanged);
  appendSessionPlan(lines, data.planSteps);
  appendSessionCost(lines, data);
  appendSessionDelegatedWork(lines, data.delegatedWork);
  lines.push(dim("─".repeat(50)));
  return lines.join("\n");
}

function appendSessionCore(lines: string[], data: SessionSummaryData): void {
  lines.push(`  Session:      ${dim(data.sessionId)}`);
  lines.push(`  Duration:     ${formatDuration(data.elapsedMs)}`);
  lines.push(`  Iterations:   ${data.totalIterations}`);
  lines.push(`  Tool calls:   ${data.totalToolCalls}`);
}

function appendSessionToolUsage(lines: string[], toolUsage: ReadonlyMap<string, number>): void {
  if (toolUsage.size === 0) {
    return;
  }
  lines.push("  Tool usage:");
  for (const [name, count] of [...toolUsage.entries()].sort((a, b) => b[1] - a[1])) {
    lines.push(`    ${name}: ${count}`);
  }
}

function appendSessionFilesChanged(lines: string[], filesChanged: ReadonlyArray<string>): void {
  if (filesChanged.length === 0) {
    return;
  }
  lines.push(`  Files changed (${filesChanged.length}):`);
  lines.push(...filesChanged.slice(0, 15).map((file) => `    ${file}`));
  if (filesChanged.length > 15) {
    lines.push(dim(`    ... (+${filesChanged.length - 15} more)`));
  }
}

function appendSessionPlan(
  lines: string[],
  planSteps: SessionSummaryData["planSteps"],
): void {
  if (!planSteps || planSteps.length === 0) {
    return;
  }
  const completed = planSteps.filter((step) => step.status === "completed").length;
  lines.push(`  Plan:         ${completed}/${planSteps.length} completed`);
}

function appendSessionCost(lines: string[], data: SessionSummaryData): void {
  if (data.totalCost > 0) {
    lines.push(`  Cost:         $${data.totalCost.toFixed(4)}`);
  }
  if (data.totalInputTokens > 0 || data.totalOutputTokens > 0) {
    const kIn = Math.round(data.totalInputTokens / 1000);
    const kOut = Math.round(data.totalOutputTokens / 1000);
    lines.push(`  Tokens:       ${kIn}k in / ${kOut}k out`);
  }
}

function appendSessionDelegatedWork(
  lines: string[],
  delegatedWork: DelegatedWorkSummary | undefined,
): void {
  if (!delegatedWork || delegatedWork.childCount === 0) {
    return;
  }
  lines.push(`  Subagents:    ${delegatedWork.childCount}`);
  const byTypeEntries = Object.entries(delegatedWork.byType);
  if (byTypeEntries.length > 0) {
    lines.push(`  By type:      ${byTypeEntries.map(([type, count]) => `${type}=${count}`).join(", ")}`);
  }
  if (delegatedWork.lanes.length > 0) {
    lines.push(`  Lanes:        ${delegatedWork.lanes.join(", ")}`);
  }
  lines.push(`  Delegated:    ${formatDuration(delegatedWork.totalDelegatedDurationMs)} total`);
  lines.push(`  Parallel:     ${delegatedWork.parallelBatchCount} batch(es), max ${delegatedWork.maxParallelChildren} child(ren)`);
}

export function formatSubagentBatchLaunch(
  agentType: string,
  batchSize: number,
): string {
  return `${dim("  ↳")} ${bold(`Launching ${batchSize} ${agentType} subagents in parallel`)}`;
}

export function formatSubagentStart(event: SubagentStartEvent): string {
  const lane = event.laneLabel ? ` ${dim(event.laneLabel)}` : "";
  const modelBits = [event.model, event.reasoningEffort].filter(Boolean).join(", ");
  const model = modelBits ? ` ${dim(`(${modelBits})`)}` : "";
  return `${dim("  ↳")} ${bold(`Subagent ${event.agentId}`)} ${dim(event.agentType)}${lane}${model}`;
}

export function formatSubagentError(event: SubagentErrorEvent): string {
  return `  ${red("✗")} ${dim(`Subagent ${event.agentId} failed`)} ${dim(`(${formatDuration(event.durationMs)})`)}${red(`: ${truncate(event.error, 80)}`)}`;
}

export function summarizeSubagentUpdate(event: SubagentUpdateEvent): string {
  if (event.summary && event.summary.trim().length > 0) {
    return event.summary;
  }
  if (event.milestone === "iteration:start") {
    return event.iteration ? `Starting iteration ${event.iteration}` : "Starting iteration";
  }
  if (event.milestone === "tool:before") {
    return event.toolName ? `Running ${event.toolName}` : "Running tool";
  }
  if (event.toolName) {
    return event.toolSuccess === false ? `Failed ${event.toolName}` : `Completed ${event.toolName}`;
  }
  return "Updated progress";
}
