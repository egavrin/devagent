/**
 * useAgentLog — shared hook for agent event processing.
 *
 * Extracts all common state, refs, callbacks, and bus event wiring
 * used by both App.tsx (interactive) and SingleShotApp.tsx (single-shot).
 */

import { useState, useCallback, useEffect, useRef } from "react";
import type { StatusBarProps } from "./StatusBar.js";
import type { ToolEvent, ToolGroupEvent } from "./ToolDisplay.js";
import type { SubagentState } from "./SubagentPanel.js";
import type { EventBus } from "@devagent/runtime";
import {
  type LogEntry,
  extractToolPreview,
  extractEditDiff,
  summarizeParams,
} from "./shared.js";

// ─── Hook Options ──────────────────────────────────────────

export interface UseAgentLogOptions {
  readonly bus: EventBus;
  readonly model: string;
  readonly approvalMode: string;
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

export interface UseAgentLogResult {
  readonly log: LogEntry[];
  readonly status: StatusBarProps;
  readonly subagents: Map<string, SubagentState>;
  readonly spinnerMessage: string | undefined;
  readonly addLog: (entry: LogEntry) => void;
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

// ─── Hook ──────────────────────────────────────────────────

function stripMarkdown(text: string): string {
  return text.replace(/\*\*([^*]+)\*\*/g, "$1").replace(/`([^`]+)`/g, "$1");
}

export function useAgentLog(options: UseAgentLogOptions): UseAgentLogResult {
  const {
    bus, model, approvalMode,
    onStreamingText, onToolStart, onApproval,
    collapseFailures = false, compactPlanProgress = false,
  } = options;

  const [log, setLog] = useState<LogEntry[]>([]);
  const [status, setStatus] = useState<StatusBarProps>({
    model, cost: 0, inputTokens: 0, maxContextTokens: 0,
    iteration: 0, maxIterations: 0, approvalMode,
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

  const addLog = useCallback((entry: LogEntry) => {
    setLog((prev) => [...prev.slice(-50), entry]);
  }, []);

  const flushGroup = useCallback(() => {
    const group = pendingGroupRef.current;
    if (!group || group.count <= 1) { pendingGroupRef.current = null; return; }
    addLog({
      id: nextId("group"), type: "tool-group",
      data: {
        name: group.name, count: group.count, summaries: group.summaries,
        iteration: statusRef.current.iteration, maxIterations: statusRef.current.maxIterations,
        status: group.lastSuccess ? "success" : "error", totalDurationMs: group.totalMs,
      } satisfies ToolGroupEvent,
    });
    pendingGroupRef.current = null;
  }, [addLog]);

  const flushThinking = useCallback(() => {
    const text = textBufferRef.current.trim();
    if (text) {
      for (const line of text.split("\n").filter((l) => l.trim()).slice(0, 3)) {
        addLog({
          id: nextId("reasoning"), type: "reasoning",
          data: { text: stripMarkdown(line.length > 120 ? line.slice(0, 117) + "..." : line) },
        });
      }
    }
    if (thinkingStartRef.current !== null) {
      const duration = Date.now() - thinkingStartRef.current;
      if (duration > 500) addLog({ id: nextId("think-dur"), type: "thinking-duration", data: { durationMs: duration } });
      thinkingStartRef.current = null;
    }
    textBufferRef.current = "";
  }, [addLog]);

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
        textBufferRef.current += e.content;
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
        addLog({ id: nextId("fail-batch"), type: "info", data: `✗ ${failureBuffer.count} calls failed` });
        failureBuffer.count = 0;
      }

      const summary = summarizeParams(e.name, e.params);
      setSpinnerMessage(`${e.name} ${summary}`.trim());

      // Tool grouping
      if (pendingGroupRef.current?.name === e.name) {
        pendingGroupRef.current.count++;
        if (summary) pendingGroupRef.current.summaries.push(summary);
        return;
      }
      flushGroup();
      pendingGroupRef.current = { name: e.name, count: 1, summaries: summary ? [summary] : [], totalMs: 0, lastSuccess: true };

      addLog({
        id: e.callId, type: "tool",
        data: {
          id: e.callId, name: e.name, summary,
          iteration: statusRef.current.iteration, maxIterations: statusRef.current.maxIterations,
          status: "running",
        } satisfies ToolEvent,
      });
    }));

    unsubs.push(bus.on("tool:after", (e) => {
      if (e.agentId || e.name === "update_plan" || e.name === "execute_tool_script" || (e.name === "delegate" && e.result.metadata?.["agentMeta"])) return;

      // Collapse failures
      if (collapseFailures && !e.result.success) {
        failureBuffer.count++;
        return;
      }
      if (collapseFailures && failureBuffer.count > 0) {
        addLog({ id: nextId("fail-batch"), type: "info", data: `✗ ${failureBuffer.count} calls failed` });
        failureBuffer.count = 0;
      }

      // Accumulate into group
      if (pendingGroupRef.current?.name === e.name) {
        pendingGroupRef.current.totalMs += e.durationMs;
        if (!e.result.success) pendingGroupRef.current.lastSuccess = false;
        if (pendingGroupRef.current.count === 1) {
          const preview = e.result.success ? extractToolPreview(e.name, e.result.output) : undefined;
          const diff = e.result.success ? extractEditDiff(e.name, e.result.output) : undefined;
          addLog({
            id: `${e.callId}-done`, type: "tool",
            data: {
              id: e.callId, name: e.name, summary: "",
              iteration: statusRef.current.iteration, maxIterations: statusRef.current.maxIterations,
              status: e.result.success ? "success" : "error",
              durationMs: e.durationMs, error: e.result.error ?? undefined, preview, diff,
            } satisfies ToolEvent,
          });
        }
        return;
      }

      const preview = e.result.success ? extractToolPreview(e.name, e.result.output) : undefined;
      const diff = e.result.success ? extractEditDiff(e.name, e.result.output) : undefined;
      addLog({
        id: `${e.callId}-done`, type: "tool",
        data: {
          id: e.callId, name: e.name, summary: "",
          iteration: statusRef.current.iteration, maxIterations: statusRef.current.maxIterations,
          status: e.result.success ? "success" : "error",
          durationMs: e.durationMs, error: e.result.error ?? undefined, preview, diff,
        } satisfies ToolEvent,
      });
    }));

    unsubs.push(bus.on("message:tool", (e) => {
      if (e.agentId) return;
      if (e.summaryOnly) {
        let content = e.content;
        content = content.replace(/Subagent \S+ /, "");
        addLog({ id: nextId("tool-msg"), type: "info", data: `  ✓ ${content}` });
      }
    }));

    let lastPlanStepCount = 0;
    unsubs.push(bus.on("plan:updated", (e) => {
      const steps = e.steps as Array<{ description: string; status: string }>;
      if (steps.length !== lastPlanStepCount) {
        addLog({ id: nextId("plan"), type: "plan", data: steps });
        lastPlanStepCount = steps.length;
      } else {
        const active = steps.find((s) => s.status === "in_progress");
        const completed = steps.filter((s) => s.status === "completed").length;
        const total = steps.length;
        if (compactPlanProgress) {
          const bar = "█".repeat(completed) + "░".repeat(total - completed);
          const costStr = costAccumRef.current > 0 ? ` $${costAccumRef.current.toFixed(4)}` : "";
          const label = active ? active.description : `All completed${costStr}`;
          addLog({ id: nextId("plan-status"), type: "info", data: `[${bar}] ${completed}/${total} ${label}` });
        } else {
          addLog({ id: nextId("plan-status"), type: "info", data: `── Plan ${completed}/${total} ── ${active?.description ?? "All completed"}` });
        }
      }
    }));

    unsubs.push(bus.on("context:compacting", () => { setSpinnerMessage("Compacting context…"); }));
    unsubs.push(bus.on("context:compacted", (e) => { setSpinnerMessage(undefined); addLog({ id: nextId("compact"), type: "compaction", data: e }); }));
    unsubs.push(bus.on("error", (e) => { addLog({ id: nextId("error"), type: "error", data: e }); }));

    if (onApproval) {
      unsubs.push(bus.on("approval:request", (e) => {
        onApproval({ id: e.id, toolName: e.toolName, details: e.details });
      }));
    }

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
      addLog({ id: nextId("sa-err"), type: "error", data: { message: `Subagent ${e.agentType} failed: ${e.error}`, code: "SUBAGENT_ERROR" } });
    }));

    unsubs.push(bus.on("session:end", () => { setSubagents(new Map()); }));

    return () => { for (const unsub of unsubs) unsub(); };
  }, [bus, addLog, flushThinking, flushGroup, nextId, collapseFailures, compactPlanProgress, onStreamingText, onToolStart, onApproval]);

  return {
    log, status, subagents, spinnerMessage,
    addLog, flushThinking, flushGroup, nextId,
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
