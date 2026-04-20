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
import type { EventBus } from "@devagent/runtime";

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
  const pendingGroupRef = useRef<{ name: string; count: number; summaries: string[]; totalMs: number; lastSuccess: boolean } | null>(null);
  const ungroupedToolNamesRef = useRef(new Set(["write_file", "replace_in_file"]));

  const refreshTranscript = useCallback(() => {
    setTranscriptNodes(composerRef.current.getNodes());
  }, []);

  const appendStandalonePart = useCallback((id: string, part: LogEntry["part"]) => {
    composerRef.current.appendStandalone(id, part);
    refreshTranscript();
  }, [refreshTranscript]);

  const startTurn = useCallback((id: string, userText: string, startedAt: number) => {
    composerRef.current.startTurn(id, userText, startedAt);
    refreshTranscript();
  }, [refreshTranscript]);

  const appendTurnPart = useCallback((id: string, part: LogEntry["part"]) => {
    composerRef.current.appendPart(id, part);
    refreshTranscript();
  }, [refreshTranscript]);

  const completeTurn = useCallback((
    id: string,
    summaryPart: Extract<LogEntry["part"], { readonly kind: "turn-summary" }>,
    options?: { readonly status?: "completed" | "error" | "budget_exceeded"; readonly finishedAt?: number },
  ) => {
    composerRef.current.completeTurn(id, summaryPart, options);
    refreshTranscript();
  }, [refreshTranscript]);

  const addParts = useCallback((prefix: string, parts: ReadonlyArray<LogEntry["part"]>) => {
    for (const [index, part] of parts.entries()) {
      composerRef.current.appendPart(`${prefix}-${index + 1}`, part);
    }
    refreshTranscript();
  }, [refreshTranscript]);

  const flushGroup = useCallback(() => {
    const group = pendingGroupRef.current;
    if (!group || group.count <= 1) { pendingGroupRef.current = null; return; }
    appendTurnPart(nextId("group"), presentToolGroupEvent({
        name: group.name, count: group.count, summaries: group.summaries,
        iteration: statusRef.current.iteration, maxIterations: statusRef.current.maxIterations,
        status: group.lastSuccess ? "success" : "error", totalDurationMs: group.totalMs,
      } satisfies PresentedToolGroup));
    pendingGroupRef.current = null;
  }, [appendTurnPart, nextId]);

  const flushThinking = useCallback(() => {
    if (thinkingStartRef.current !== null) {
      const duration = Date.now() - thinkingStartRef.current;
      if (duration > 500) appendTurnPart(nextId("think-dur"), { kind: "info", data: { title: "thinking", lines: [`thought for ${(duration / 1000).toFixed(1)}s`] } });
      thinkingStartRef.current = null;
    }
  }, [appendTurnPart, nextId]);

  const shouldGroupTool = useCallback((name: string): boolean => {
    return !ungroupedToolNamesRef.current.has(name);
  }, []);

  // Wire bus events
  useEffect(() => {
    const unsubs: Array<() => void> = [];
    const failureBuffer = { count: 0 };

    unsubs.push(bus.on("iteration:start", (e) => {
      if (e.agentId) return;
      setStatus((s) => ({
        ...s, iteration: e.iteration, maxIterations: e.maxIterations,
        inputTokens: e.estimatedTokens, maxContextTokens: e.maxContextTokens,
      }));
    }));

    unsubs.push(bus.on("cost:update", (e) => {
      costAccumRef.current += e.totalCost;
      setStatus((s) => ({ ...s, cost: s.cost + e.totalCost }));
    }));

    unsubs.push(bus.on("message:assistant", (e) => {
      if (e.agentId) return;
      if (e.partial) {
        if (e.chunk?.type === "text") onStreamingText?.(e.content);
        if (e.chunk?.type === "thinking" && thinkingStartRef.current === null) thinkingStartRef.current = Date.now();
      }
    }));

    unsubs.push(bus.on("tool:before", (e) => {
      if (e.agentId || e.name === "delegate" || e.name === "update_plan" || e.name === "execute_tool_script") return;
      flushThinking();
      onToolStart?.();
      turnToolCountRef.current++;

      // Flush buffered failures
      if (collapseFailures && failureBuffer.count > 0) {
        appendTurnPart(nextId("fail-batch"), makeStatusPart({ title: "tools", lines: [`${failureBuffer.count} calls failed`], tone: "warning" }));
        failureBuffer.count = 0;
      }

      const summary = summarizeToolParamsForTranscript(e.name, e.params);
      setSpinnerMessage(`${e.name} ${summary}`.trim());

      // Tool grouping
      if (shouldGroupTool(e.name) && pendingGroupRef.current?.name === e.name) {
        pendingGroupRef.current.count++;
        if (summary) pendingGroupRef.current.summaries.push(summary);
        return;
      }
      flushGroup();
      if (shouldGroupTool(e.name)) {
        pendingGroupRef.current = { name: e.name, count: 1, summaries: summary ? [summary] : [], totalMs: 0, lastSuccess: true };
      }

      appendTurnPart(e.callId, presentToolBeforeEvent(e, statusRef.current.iteration, statusRef.current.maxIterations));
    }));

    unsubs.push(bus.on("tool:after", (e) => {
      if (e.agentId || e.name === "update_plan" || e.name === "execute_tool_script" || (e.name === "delegate" && e.result.metadata?.["agentMeta"])) return;

      // Collapse failures
      if (collapseFailures && !e.result.success) {
        failureBuffer.count++;
        return;
      }
      if (collapseFailures && failureBuffer.count > 0) {
        appendTurnPart(nextId("fail-batch"), makeStatusPart({ title: "tools", lines: [`${failureBuffer.count} calls failed`], tone: "warning" }));
        failureBuffer.count = 0;
      }

      // Accumulate into group
      if (shouldGroupTool(e.name) && pendingGroupRef.current?.name === e.name) {
        pendingGroupRef.current.totalMs += e.durationMs;
        if (!e.result.success) pendingGroupRef.current.lastSuccess = false;
        if (pendingGroupRef.current.count === 1) {
          addParts(
            `${e.callId}-done`,
            presentToolAfterEvent(e, statusRef.current.iteration, statusRef.current.maxIterations),
          );
        }
        return;
      }

      addParts(
        `${e.callId}-done`,
        presentToolAfterEvent(e, statusRef.current.iteration, statusRef.current.maxIterations),
      );
    }));

    unsubs.push(bus.on("message:tool", (e) => {
      if (e.agentId) return;
      if (e.summaryOnly) {
        appendTurnPart(nextId("tool-msg"), presentSummaryToolMessage(e));
      }
    }));

    let lastPlanStepCount = 0;
    unsubs.push(bus.on("plan:updated", (e) => {
      const steps = e.steps as Array<{ description: string; status: string }>;
      if (steps.length !== lastPlanStepCount) {
        appendTurnPart(nextId("plan"), makePlanPart(steps));
        lastPlanStepCount = steps.length;
      } else {
        const active = steps.find((s) => s.status === "in_progress");
        const completed = steps.filter((s) => s.status === "completed").length;
        const total = steps.length;
        if (compactPlanProgress) {
          const bar = "█".repeat(completed) + "░".repeat(total - completed);
          const costStr = costAccumRef.current > 0 ? ` $${costAccumRef.current.toFixed(4)}` : "";
          const label = active ? active.description : `All completed${costStr}`;
          appendTurnPart(nextId("plan-status"), makeStatusPart({ title: "plan", lines: [`[${bar}] ${completed}/${total} ${label}`], tone: "info" }));
        } else {
          appendTurnPart(nextId("plan-status"), makeStatusPart({ title: "plan", lines: [`${completed}/${total} ${active?.description ?? "All completed"}`], tone: "info" }));
        }
      }
    }));

    unsubs.push(bus.on("context:compacting", (e) => {
      setSpinnerMessage("Compacting context…");
      appendTurnPart(nextId("progress"), presentContextCompactingEvent(e));
    }));
    unsubs.push(bus.on("context:compacted", (e) => {
      setSpinnerMessage(undefined);
      appendTurnPart(nextId("compact"), presentContextCompactedEvent(e));
    }));
    unsubs.push(bus.on("error", (e) => { appendTurnPart(nextId("error"), makeErrorPart(e)); }));

    unsubs.push(bus.on("approval:request", (e) => {
      appendTurnPart(nextId("approval"), presentApprovalRequestEvent(e));
      onApproval?.({ id: e.id, toolName: e.toolName, details: e.details });
    }));
    unsubs.push(bus.on("approval:response", (e) => {
      appendTurnPart(nextId("approval-response"), presentApprovalResponseEvent(e));
    }));

    // Subagents
    unsubs.push(bus.on("subagent:start", (e) => {
      setSubagents((prev) => {
        const next = new Map(prev);
        next.set(e.agentId, { agentId: e.agentId, agentType: String(e.agentType), laneLabel: e.laneLabel, status: "running", iteration: 0, startedAt: Date.now(), activity: "Starting…" });
        return next;
      });
    }));
    unsubs.push(bus.on("subagent:update", (e) => {
      setSubagents((prev) => {
        const existing = prev.get(e.agentId);
        if (!existing) return prev;
        const newIter = e.iteration ?? existing.iteration;
        const newActivity = e.summary ?? e.toolName ?? existing.activity;
        if (newIter === existing.iteration && newActivity === existing.activity) return prev;
        const next = new Map(prev);
        next.set(e.agentId, { ...existing, iteration: newIter, activity: newActivity });
        return next;
      });
    }));
    unsubs.push(bus.on("subagent:end", (e) => {
      setSubagents((prev) => {
        const existing = prev.get(e.agentId);
        if (!existing) return prev;
        const next = new Map(prev);
        next.set(e.agentId, { ...existing, status: "completed", iteration: e.iterations, quality: e.quality ? { score: e.quality.score, completeness: e.quality.completeness } : undefined });
        return next;
      });
    }));
    unsubs.push(bus.on("subagent:error", (e) => {
      setSubagents((prev) => {
        const existing = prev.get(e.agentId);
        if (!existing) return prev;
        const next = new Map(prev);
        next.set(e.agentId, { ...existing, status: "error", error: e.error });
        return next;
      });
      appendTurnPart(nextId("sa-err"), makeErrorPart({ message: `Subagent ${e.agentType} failed: ${e.error}`, code: "SUBAGENT_ERROR" }));
    }));

    unsubs.push(bus.on("session:end", () => { setSubagents(new Map()); }));

    return () => { for (const unsub of unsubs) unsub(); };
  }, [bus, appendTurnPart, addParts, flushThinking, flushGroup, nextId, collapseFailures, compactPlanProgress, onStreamingText, onToolStart, onApproval, shouldGroupTool]);

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
