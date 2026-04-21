/**
 * useAgentLog — shared hook for agent event processing.
 *
 * Extracts all common state, refs, callbacks, and bus event wiring
 * used by both App.tsx (interactive) and SingleShotApp.tsx (single-shot).
 */

import { useState, useCallback, useEffect, useRef } from "react";

import {
  type LogEntry,
  type TranscriptNode,
} from "./shared.js";
import type { StatusBarProps } from "./StatusBar.js";
import type { SubagentState } from "./SubagentPanel.js";
import { TranscriptComposer } from "../transcript-composer.js";
import {
  type PresentedToolGroup,
  makeErrorPart,
  makePlanPart,
  makeStatusPart,
  presentApprovalRequestEvent,
  presentApprovalResponseEvent,
  presentContextCompactedEvent,
  presentContextCompactingEvent,
  presentSummaryToolMessage,
  presentToolAfterEvent,
  presentToolBeforeEvent,
  presentToolGroupEvent,
  summarizeToolParamsForTranscript,
} from "../transcript-presenter.js";
import type { EventBus, PlanUpdatedEvent, SubagentEndEvent, ToolAfterEvent } from "@devagent/runtime";

// ─── Hook Options ──────────────────────────────────────────

interface UseAgentLogOptions {
  readonly bus: EventBus;
  readonly model: string;
  /** Called on each streaming text chunk (App uses this for live preview). */
  readonly onStreamingText?: (content: string) => void;
  /** Called when a tool starts (App uses this to clear streaming text). */
  readonly onToolStart?: () => void;
  /** Called on approval:request events. */
  readonly onApproval?: (event: { id: string; toolName: string; details: string }) => void;
  /** Whether to buffer consecutive failures into a single line. Default: false. */
  readonly collapseFailures?: boolean;
  /** Whether to show compact plan progress bars. Default: false. */
  readonly compactPlanProgress?: boolean;
}

// ─── Hook Result ───────────────────────────────────────────

interface UseAgentLogResult {
  readonly transcriptNodes: ReadonlyArray<TranscriptNode>;
  readonly status: StatusBarProps;
  readonly subagents: Map<string, SubagentState>;
  readonly spinnerMessage: string | undefined;
  readonly appendStandalonePart: (id: string, part: LogEntry["part"]) => void;
  readonly startTurn: (id: string, userText: string, startedAt: number) => void;
  readonly appendTurnPart: (id: string, part: LogEntry["part"]) => void;
  readonly completeTurn: (
    id: string,
    summaryPart: Extract<LogEntry["part"], { readonly kind: "turn-summary" }>,
    options?: { readonly status?: "completed" | "error" | "budget_exceeded"; readonly finishedAt?: number },
  ) => void;
  readonly flushThinking: () => void;
  readonly flushGroup: () => void;
  readonly nextId: (prefix: string) => string;
  readonly setStatus: React.Dispatch<React.SetStateAction<StatusBarProps>>;
  readonly setSubagents: React.Dispatch<React.SetStateAction<Map<string, SubagentState>>>;
  readonly setSpinnerMessage: React.Dispatch<React.SetStateAction<string | undefined>>;
  /** Refs exposed for turn lifecycle management. */
  readonly refs: {
    readonly textBuffer: React.MutableRefObject<string>;
    readonly thinkingStart: React.MutableRefObject<number | null>;
    readonly turnStart: React.MutableRefObject<number>;
    readonly turnToolCount: React.MutableRefObject<number>;
    readonly costAccum: React.MutableRefObject<number>;
  };
}

interface PendingToolGroup {
  name: string;
  count: number;
  summaries: string[];
  totalMs: number;
  lastSuccess: boolean;
}

interface AgentLogEventRuntime {
  readonly bus: EventBus;
  readonly appendTurnPart: (id: string, part: LogEntry["part"]) => void;
  readonly addParts: (prefix: string, parts: ReadonlyArray<LogEntry["part"]>) => void;
  readonly flushThinking: () => void;
  readonly flushGroup: () => void;
  readonly nextId: (prefix: string) => string;
  readonly shouldGroupTool: (name: string) => boolean;
  readonly collapseFailures: boolean;
  readonly compactPlanProgress: boolean;
  readonly onStreamingText: ((content: string) => void) | undefined;
  readonly onToolStart: (() => void) | undefined;
  readonly onApproval: ((event: { id: string; toolName: string; details: string }) => void) | undefined;
  readonly setStatus: React.Dispatch<React.SetStateAction<StatusBarProps>>;
  readonly setSubagents: React.Dispatch<React.SetStateAction<Map<string, SubagentState>>>;
  readonly setSpinnerMessage: React.Dispatch<React.SetStateAction<string | undefined>>;
  readonly statusRef: React.MutableRefObject<StatusBarProps>;
  readonly thinkingStartRef: React.MutableRefObject<number | null>;
  readonly turnToolCountRef: React.MutableRefObject<number>;
  readonly costAccumRef: React.MutableRefObject<number>;
  readonly pendingGroupRef: React.MutableRefObject<PendingToolGroup | null>;
}

interface AgentLogSubscriptionOptions {
  readonly bus: EventBus;
  readonly appendTurnPart: (id: string, part: LogEntry["part"]) => void;
  readonly addParts: (prefix: string, parts: ReadonlyArray<LogEntry["part"]>) => void;
  readonly flushThinking: () => void;
  readonly flushGroup: () => void;
  readonly nextId: (prefix: string) => string;
  readonly shouldGroupTool: (name: string) => boolean;
  readonly collapseFailures: boolean;
  readonly compactPlanProgress: boolean;
  readonly onStreamingText: ((content: string) => void) | undefined;
  readonly onToolStart: (() => void) | undefined;
  readonly onApproval: ((event: { id: string; toolName: string; details: string }) => void) | undefined;
  readonly setStatus: React.Dispatch<React.SetStateAction<StatusBarProps>>;
  readonly setSubagents: React.Dispatch<React.SetStateAction<Map<string, SubagentState>>>;
  readonly setSpinnerMessage: React.Dispatch<React.SetStateAction<string | undefined>>;
  readonly statusRef: React.MutableRefObject<StatusBarProps>;
  readonly thinkingStartRef: React.MutableRefObject<number | null>;
  readonly turnToolCountRef: React.MutableRefObject<number>;
  readonly costAccumRef: React.MutableRefObject<number>;
  readonly pendingGroupRef: React.MutableRefObject<PendingToolGroup | null>;
}

interface TranscriptActions {
  readonly appendStandalonePart: (id: string, part: LogEntry["part"]) => void;
  readonly startTurn: (id: string, userText: string, startedAt: number) => void;
  readonly appendTurnPart: (id: string, part: LogEntry["part"]) => void;
  readonly completeTurn: (
    id: string,
    summaryPart: Extract<LogEntry["part"], { readonly kind: "turn-summary" }>,
    options?: { readonly status?: "completed" | "error" | "budget_exceeded"; readonly finishedAt?: number },
  ) => void;
  readonly addParts: (prefix: string, parts: ReadonlyArray<LogEntry["part"]>) => void;
}

interface ToolGroupControls {
  readonly flushGroup: () => void;
  readonly flushThinking: () => void;
  readonly shouldGroupTool: (name: string) => boolean;
}

function flushFailureBuffer(
  runtime: AgentLogEventRuntime,
  failureBuffer: { count: number },
): void {
  if (!runtime.collapseFailures || failureBuffer.count === 0) {
    return;
  }
  runtime.appendTurnPart(
    runtime.nextId("fail-batch"),
    makeStatusPart({ title: "tools", lines: [`${failureBuffer.count} calls failed`], tone: "warning" }),
  );
  failureBuffer.count = 0;
}

function registerStatusEvents(unsubs: Array<() => void>, runtime: AgentLogEventRuntime): void {
  unsubs.push(runtime.bus.on("iteration:start", (event) => {
    if (event.agentId) return;
    runtime.setStatus((status) => ({
      ...status,
      iteration: event.iteration,
      maxIterations: event.maxIterations,
      inputTokens: event.estimatedTokens,
      maxContextTokens: event.maxContextTokens,
    }));
  }));

  unsubs.push(runtime.bus.on("cost:update", (event) => {
    runtime.costAccumRef.current += event.totalCost;
    runtime.setStatus((status) => ({ ...status, cost: status.cost + event.totalCost }));
  }));
}

function registerAssistantEvents(unsubs: Array<() => void>, runtime: AgentLogEventRuntime): void {
  unsubs.push(runtime.bus.on("message:assistant", (event) => {
    if (event.agentId || !event.partial) return;
    if (event.chunk?.type === "text") runtime.onStreamingText?.(event.content);
    if (event.chunk?.type === "thinking" && runtime.thinkingStartRef.current === null) {
      runtime.thinkingStartRef.current = Date.now();
    }
  }));
}

function registerToolBeforeEvent(
  unsubs: Array<() => void>,
  runtime: AgentLogEventRuntime,
  failureBuffer: { count: number },
): void {
  unsubs.push(runtime.bus.on("tool:before", (event) => {
    if (event.agentId || event.name === "delegate" || event.name === "update_plan" || event.name === "execute_tool_script") return;
    runtime.flushThinking();
    runtime.onToolStart?.();
    runtime.turnToolCountRef.current++;
    flushFailureBuffer(runtime, failureBuffer);

    const summary = summarizeToolParamsForTranscript(event.name, event.params);
    runtime.setSpinnerMessage(`${event.name} ${summary}`.trim());
    if (appendToPendingToolGroup(event.name, summary, runtime)) return;

    runtime.flushGroup();
    if (runtime.shouldGroupTool(event.name)) {
      runtime.pendingGroupRef.current = {
        name: event.name,
        count: 1,
        summaries: summary ? [summary] : [],
        totalMs: 0,
        lastSuccess: true,
      };
    }
    runtime.appendTurnPart(
      event.callId,
      presentToolBeforeEvent(event, runtime.statusRef.current.iteration, runtime.statusRef.current.maxIterations),
    );
  }));
}

function appendToPendingToolGroup(
  toolName: string,
  summary: string,
  runtime: AgentLogEventRuntime,
): boolean {
  const pendingGroup = runtime.pendingGroupRef.current;
  if (!runtime.shouldGroupTool(toolName) || pendingGroup?.name !== toolName) {
    return false;
  }
  pendingGroup.count++;
  if (summary) pendingGroup.summaries.push(summary);
  return true;
}

function shouldSkipToolAfter(event: ToolAfterEvent): boolean {
  return Boolean(
    event.agentId
    || event.name === "update_plan"
    || event.name === "execute_tool_script"
    || (event.name === "delegate" && event.result.metadata?.["agentMeta"])
  );
}

function addToolAfterParts(
  event: ToolAfterEvent,
  runtime: AgentLogEventRuntime,
): void {
  runtime.addParts(
    `${event.callId}-done`,
    presentToolAfterEvent(event, runtime.statusRef.current.iteration, runtime.statusRef.current.maxIterations),
  );
}

function registerToolAfterEvent(
  unsubs: Array<() => void>,
  runtime: AgentLogEventRuntime,
  failureBuffer: { count: number },
): void {
  unsubs.push(runtime.bus.on("tool:after", (event) => {
    if (shouldSkipToolAfter(event)) return;
    if (runtime.collapseFailures && !event.result.success) {
      failureBuffer.count++;
      return;
    }
    flushFailureBuffer(runtime, failureBuffer);
    if (applyPendingToolResult(event, runtime)) return;
    addToolAfterParts(event, runtime);
  }));
}

function registerToolMessageEvents(unsubs: Array<() => void>, runtime: AgentLogEventRuntime): void {
  unsubs.push(runtime.bus.on("message:tool", (event) => {
    if (event.agentId || !event.summaryOnly) return;
    runtime.appendTurnPart(runtime.nextId("tool-msg"), presentSummaryToolMessage(event));
  }));
}

function applyPendingToolResult(
  event: ToolAfterEvent,
  runtime: AgentLogEventRuntime,
): boolean {
  const pendingGroup = runtime.pendingGroupRef.current;
  if (!runtime.shouldGroupTool(event.name) || pendingGroup?.name !== event.name) {
    return false;
  }
  pendingGroup.totalMs += event.durationMs;
  if (!event.result.success) pendingGroup.lastSuccess = false;
  if (pendingGroup.count === 1) addToolAfterParts(event, runtime);
  return true;
}

function registerPlanAndContextEvents(unsubs: Array<() => void>, runtime: AgentLogEventRuntime): void {
  let lastPlanStepCount = 0;
  unsubs.push(runtime.bus.on("plan:updated", (event) => {
    lastPlanStepCount = handlePlanUpdate(event, lastPlanStepCount, runtime);
  }));
  unsubs.push(runtime.bus.on("context:compacting", (event) => {
    runtime.setSpinnerMessage("Compacting context…");
    runtime.appendTurnPart(runtime.nextId("progress"), presentContextCompactingEvent(event));
  }));
  unsubs.push(runtime.bus.on("context:compacted", (event) => {
    runtime.setSpinnerMessage(undefined);
    runtime.appendTurnPart(runtime.nextId("compact"), presentContextCompactedEvent(event));
  }));
}

function handlePlanUpdate(
  event: PlanUpdatedEvent,
  lastPlanStepCount: number,
  runtime: AgentLogEventRuntime,
): number {
  if (event.steps.length !== lastPlanStepCount) {
    runtime.appendTurnPart(runtime.nextId("plan"), makePlanPart(event.steps));
    return event.steps.length;
  }
  appendPlanProgress(event.steps, runtime);
  return lastPlanStepCount;
}

function appendPlanProgress(
  steps: ReadonlyArray<{ readonly description: string; readonly status: string }>,
  runtime: AgentLogEventRuntime,
): void {
  const active = steps.find((step) => step.status === "in_progress");
  const completed = steps.filter((step) => step.status === "completed").length;
  const total = steps.length;
  const lines = runtime.compactPlanProgress
    ? [formatCompactPlanProgress(active?.description, completed, total, runtime.costAccumRef.current)]
    : [`${completed}/${total} ${active?.description ?? "All completed"}`];
  runtime.appendTurnPart(runtime.nextId("plan-status"), makeStatusPart({ title: "plan", lines, tone: "info" }));
}

function formatCompactPlanProgress(
  activeDescription: string | undefined,
  completed: number,
  total: number,
  cost: number,
): string {
  const bar = "█".repeat(completed) + "░".repeat(total - completed);
  const costStr = cost > 0 ? ` $${cost.toFixed(4)}` : "";
  const label = activeDescription ?? `All completed${costStr}`;
  return `[${bar}] ${completed}/${total} ${label}`;
}

function registerApprovalAndSessionEvents(unsubs: Array<() => void>, runtime: AgentLogEventRuntime): void {
  unsubs.push(runtime.bus.on("error", (event) => {
    runtime.appendTurnPart(runtime.nextId("error"), makeErrorPart(event));
  }));
  unsubs.push(runtime.bus.on("approval:request", (event) => {
    runtime.appendTurnPart(runtime.nextId("approval"), presentApprovalRequestEvent(event));
    runtime.onApproval?.({ id: event.id, toolName: event.toolName, details: event.details });
  }));
  unsubs.push(runtime.bus.on("approval:response", (event) => {
    runtime.appendTurnPart(runtime.nextId("approval-response"), presentApprovalResponseEvent(event));
  }));
  unsubs.push(runtime.bus.on("session:end", () => {
    runtime.setSubagents(new Map());
  }));
}

function registerSubagentEvents(unsubs: Array<() => void>, runtime: AgentLogEventRuntime): void {
  unsubs.push(runtime.bus.on("subagent:start", (event) => {
    runtime.setSubagents((prev) => {
      const next = new Map(prev);
      next.set(event.agentId, {
        agentId: event.agentId,
        agentType: String(event.agentType),
        laneLabel: event.laneLabel,
        status: "running",
        iteration: 0,
        startedAt: Date.now(),
        activity: "Starting…",
      });
      return next;
    });
  }));
  unsubs.push(runtime.bus.on("subagent:update", (event) => {
    runtime.setSubagents((prev) => updateSubagentActivity(prev, event.agentId, event.iteration, event.summary ?? event.toolName));
  }));
  unsubs.push(runtime.bus.on("subagent:end", (event) => {
    runtime.setSubagents((prev) => completeSubagent(prev, event));
  }));
  unsubs.push(runtime.bus.on("subagent:error", (event) => {
    runtime.setSubagents((prev) => failSubagent(prev, event.agentId, event.error));
    runtime.appendTurnPart(
      runtime.nextId("sa-err"),
      makeErrorPart({ message: `Subagent ${event.agentType} failed: ${event.error}`, code: "SUBAGENT_ERROR" }),
    );
  }));
}

function updateSubagentActivity(
  prev: Map<string, SubagentState>,
  agentId: string,
  iteration: number | undefined,
  activity: string | undefined,
): Map<string, SubagentState> {
  const existing = prev.get(agentId);
  if (!existing) return prev;
  const newIter = iteration ?? existing.iteration;
  const newActivity = activity ?? existing.activity;
  if (newIter === existing.iteration && newActivity === existing.activity) return prev;
  const next = new Map(prev);
  next.set(agentId, { ...existing, iteration: newIter, activity: newActivity });
  return next;
}

function completeSubagent(prev: Map<string, SubagentState>, event: SubagentEndEvent): Map<string, SubagentState> {
  const existing = prev.get(event.agentId);
  if (!existing) return prev;
  const next = new Map(prev);
  next.set(event.agentId, {
    ...existing,
    status: "completed",
    iteration: event.iterations,
    quality: event.quality ? { score: event.quality.score, completeness: event.quality.completeness } : undefined,
  });
  return next;
}

function failSubagent(prev: Map<string, SubagentState>, agentId: string, error: string): Map<string, SubagentState> {
  const existing = prev.get(agentId);
  if (!existing) return prev;
  const next = new Map(prev);
  next.set(agentId, { ...existing, status: "error", error });
  return next;
}

function useTranscriptActions(
  composerRef: React.MutableRefObject<TranscriptComposer>,
  refreshTranscript: () => void,
): TranscriptActions {
  const appendStandalonePart = useCallback((id: string, part: LogEntry["part"]) => {
    composerRef.current.appendStandalone(id, part);
    refreshTranscript();
  }, [composerRef, refreshTranscript]);

  const startTurn = useCallback((id: string, userText: string, startedAt: number) => {
    composerRef.current.startTurn(id, userText, startedAt);
    refreshTranscript();
  }, [composerRef, refreshTranscript]);

  const appendTurnPart = useCallback((id: string, part: LogEntry["part"]) => {
    composerRef.current.appendPart(id, part);
    refreshTranscript();
  }, [composerRef, refreshTranscript]);

  const completeTurn = useCallback<TranscriptActions["completeTurn"]>((id, summaryPart, options) => {
    composerRef.current.completeTurn(id, summaryPart, options);
    refreshTranscript();
  }, [composerRef, refreshTranscript]);

  const addParts = useCallback((prefix: string, parts: ReadonlyArray<LogEntry["part"]>) => {
    for (const [index, part] of parts.entries()) {
      composerRef.current.appendPart(`${prefix}-${index + 1}`, part);
    }
    refreshTranscript();
  }, [composerRef, refreshTranscript]);

  return { appendStandalonePart, startTurn, appendTurnPart, completeTurn, addParts };
}

function useToolGroupControls(
  pendingGroupRef: React.MutableRefObject<PendingToolGroup | null>,
  statusRef: React.MutableRefObject<StatusBarProps>,
  thinkingStartRef: React.MutableRefObject<number | null>,
  appendTurnPart: (id: string, part: LogEntry["part"]) => void,
  nextId: (prefix: string) => string,
): ToolGroupControls {
  const ungroupedToolNamesRef = useRef(new Set(["write_file", "replace_in_file"]));
  const flushGroup = useCallback(() => {
    const group = pendingGroupRef.current;
    if (!group || group.count <= 1) { pendingGroupRef.current = null; return; }
    appendTurnPart(nextId("group"), presentToolGroupEvent({
      name: group.name,
      count: group.count,
      summaries: group.summaries,
      iteration: statusRef.current.iteration,
      maxIterations: statusRef.current.maxIterations,
      status: group.lastSuccess ? "success" : "error",
      totalDurationMs: group.totalMs,
    } satisfies PresentedToolGroup));
    pendingGroupRef.current = null;
  }, [appendTurnPart, nextId, pendingGroupRef, statusRef]);

  const flushThinking = useCallback(() => {
    if (thinkingStartRef.current === null) return;
    const duration = Date.now() - thinkingStartRef.current;
    if (duration > 500) {
      appendTurnPart(nextId("think-dur"), {
        kind: "info",
        data: { title: "thinking", lines: [`thought for ${(duration / 1000).toFixed(1)}s`] },
      });
    }
    thinkingStartRef.current = null;
  }, [appendTurnPart, nextId, thinkingStartRef]);

  const shouldGroupTool = useCallback((name: string): boolean => {
    return !ungroupedToolNamesRef.current.has(name);
  }, []);

  return { flushGroup, flushThinking, shouldGroupTool };
}

function useAgentLogEventSubscriptions(options: AgentLogSubscriptionOptions): void {
  useEffect(() => {
    const unsubs: Array<() => void> = [];
    const failureBuffer = { count: 0 };
    const runtime: AgentLogEventRuntime = { ...options };

    registerStatusEvents(unsubs, runtime);
    registerAssistantEvents(unsubs, runtime);
    registerToolBeforeEvent(unsubs, runtime, failureBuffer);
    registerToolAfterEvent(unsubs, runtime, failureBuffer);
    registerToolMessageEvents(unsubs, runtime);
    registerPlanAndContextEvents(unsubs, runtime);
    registerApprovalAndSessionEvents(unsubs, runtime);
    registerSubagentEvents(unsubs, runtime);

    return () => {
      for (const unsub of unsubs) unsub();
    };
  }, [
    options,
  ]);
}
export function useAgentLog(options: UseAgentLogOptions): UseAgentLogResult {
  const {
    bus, model,
    onStreamingText, onToolStart, onApproval,
    collapseFailures = false, compactPlanProgress = false,
  } = options;

  const composerRef = useRef(new TranscriptComposer());
  const [transcriptNodes, setTranscriptNodes] = useState<ReadonlyArray<TranscriptNode>>([]);
  const [status, setStatus] = useState<StatusBarProps>({
    model, cost: 0, inputTokens: 0, maxContextTokens: 0,
    iteration: 0, maxIterations: 0,
  });
  const [spinnerMessage, setSpinnerMessage] = useState<string | undefined>();
  const [subagents, setSubagents] = useState<Map<string, SubagentState>>(new Map());

  const statusRef = useRef(status);
  statusRef.current = status;

  const idCounterRef = useRef(0);
  const nextId = useCallback((prefix: string) => `${prefix}-${++idCounterRef.current}`, []);

  const textBufferRef = useRef("");
  const thinkingStartRef = useRef<number | null>(null);
  const turnStartRef = useRef(Date.now());
  const turnToolCountRef = useRef(0);
  const costAccumRef = useRef(0);
  const pendingGroupRef = useRef<PendingToolGroup | null>(null);

  const refreshTranscript = useCallback(() => {
    setTranscriptNodes(composerRef.current.getNodes());
  }, []);

  const {
    appendStandalonePart,
    startTurn,
    appendTurnPart,
    completeTurn,
    addParts,
  } = useTranscriptActions(composerRef, refreshTranscript);
  const { flushGroup, flushThinking, shouldGroupTool } = useToolGroupControls(
    pendingGroupRef,
    statusRef,
    thinkingStartRef,
    appendTurnPart,
    nextId,
  );

  useAgentLogEventSubscriptions({
    bus, appendTurnPart, addParts, flushThinking, flushGroup, nextId, shouldGroupTool,
    collapseFailures, compactPlanProgress, onStreamingText, onToolStart, onApproval,
    setStatus, setSubagents, setSpinnerMessage, statusRef, thinkingStartRef,
    turnToolCountRef, costAccumRef, pendingGroupRef,
  });

  return {
    transcriptNodes, status, subagents, spinnerMessage,
    appendStandalonePart, startTurn, appendTurnPart, completeTurn, flushThinking, flushGroup, nextId,
    setStatus, setSubagents, setSpinnerMessage,
    refs: {
      textBuffer: textBufferRef,
      thinkingStart: thinkingStartRef,
      turnStart: turnStartRef,
      turnToolCount: turnToolCountRef,
      costAccum: costAccumRef,
    },
  };
}
