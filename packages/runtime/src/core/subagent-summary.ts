import type {
  SubagentStartEvent,
  SubagentEndEvent,
  SubagentErrorEvent,
} from "./events.js";
import type { AgentType, ReasoningEffort } from "./types.js";

export interface LoggedSubagentRun {
  readonly agentId: string;
  readonly parentAgentId: string | null;
  readonly depth: number;
  readonly agentType: AgentType;
  readonly laneLabel?: string | null;
  readonly objective: string;
  readonly model: string;
  readonly reasoningEffort?: ReasoningEffort;
  readonly status: "running" | "completed" | "error";
  readonly durationMs?: number;
  readonly iterations?: number;
  readonly batchId?: string;
  readonly batchSize?: number;
  readonly toolCalls: number;
  readonly quality?: {
    readonly score: number;
    readonly completeness: string;
    readonly note?: string;
  };
}

export interface DelegatedWorkSummary {
  readonly childCount: number;
  readonly children: ReadonlyArray<LoggedSubagentRun>;
  readonly byType: Readonly<Record<string, number>>;
  readonly lanes: ReadonlyArray<string>;
  readonly totalDelegatedDurationMs: number;
  readonly parallelBatchCount: number;
  readonly maxParallelChildren: number;
}

export function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  const mins = Math.floor(ms / 60000);
  const secs = Math.round((ms % 60000) / 1000);
  return `${mins}m ${secs}s`;
}

export function loggedSubagentRunFromEvent(
  event: SubagentStartEvent | SubagentEndEvent | SubagentErrorEvent,
  existing?: LoggedSubagentRun,
): LoggedSubagentRun {
  const base: LoggedSubagentRun = {
    agentId: event.agentId,
    parentAgentId: event.parentAgentId,
    depth: event.depth,
    agentType: event.agentType,
    laneLabel: event.laneLabel,
    objective: event.objective,
    model: event.model,
    reasoningEffort: event.reasoningEffort,
    status: event.status,
    batchId: event.batchId,
    batchSize: event.batchSize,
    toolCalls: existing?.toolCalls ?? 0,
  };

  if (event.status === "running") {
    return base;
  }

  if (event.status === "completed") {
    return {
      ...base,
      durationMs: event.durationMs,
      iterations: event.iterations,
      quality: event.quality,
    };
  }

  return {
    ...base,
    durationMs: event.durationMs,
  };
}

export function aggregateDelegatedWork(
  runs: ReadonlyArray<LoggedSubagentRun>,
): DelegatedWorkSummary {
  const byType = new Map<string, number>();
  const lanes = new Set<string>();
  const batches = new Map<string, number>();
  let totalDelegatedDurationMs = 0;

  for (const run of runs) {
    byType.set(run.agentType, (byType.get(run.agentType) ?? 0) + 1);
    if (run.laneLabel) lanes.add(run.laneLabel);
    if (run.durationMs) totalDelegatedDurationMs += run.durationMs;
    if (run.batchId && (run.batchSize ?? 0) > 1) {
      batches.set(run.batchId, run.batchSize ?? 0);
    }
  }

  return {
    childCount: runs.length,
    children: [...runs],
    byType: Object.fromEntries(byType.entries()),
    lanes: [...lanes],
    totalDelegatedDurationMs,
    parallelBatchCount: batches.size,
    maxParallelChildren: batches.size > 0 ? Math.max(...batches.values()) : 0,
  };
}
