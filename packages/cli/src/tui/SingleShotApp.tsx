/**
 * SingleShotApp — Ink app for non-interactive (single query) mode.
 *
 * Renders the same components as the interactive App but:
 * - Executes one query immediately on mount
 * - Writes final output to stdout (outside Ink)
 * - Exits when the query completes
 *
 * This unifies the rendering path so both modes use the same
 * TUI components — no parallel ANSI/format.ts code needed.
 */

import React, { useState, useCallback, useEffect, useRef } from "react";
import { Box, Text, Static, useApp } from "ink";
import { StatusBar, type StatusBarProps } from "./StatusBar.js";
import { Spinner } from "./Spinner.js";
import { ToolDisplay, ToolGroupDisplay, type ToolEvent, type ToolGroupEvent } from "./ToolDisplay.js";
import { PlanView, type PlanStep } from "./PlanView.js";
import { ThinkingDuration, ErrorView } from "./MessageView.js";
import { SubagentPanel, type SubagentState } from "./SubagentPanel.js";
import { TurnSummary } from "./TurnSummary.js";
import type { EventBus } from "@devagent/runtime";

// ─── Types ──────────────────────────────────────────────────

export interface SingleShotAppProps {
  readonly bus: EventBus;
  readonly query: string;
  readonly onQuery: (query: string) => Promise<{ iterations: number; toolCalls: number; lastText: string | null }>;
  readonly model: string;
  readonly approvalMode: string;
  /** Called with the final text to write to stdout (outside Ink). */
  readonly onFinalOutput: (text: string) => void;
}

type LogEntryType = "tool" | "tool-group" | "reasoning" | "thinking-duration" | "plan" | "error" | "info" | "compaction" | "turn-summary";

interface LogEntry {
  readonly id: string;
  readonly type: LogEntryType;
  readonly data: unknown;
}

// Reuse helpers from App.tsx
function extractToolPreview(toolName: string, output: string): string | undefined {
  if (!output || output.length < 10) return undefined;
  if (toolName === "search_files") { const m = output.match(/^(\d+) match/); if (m) return output.split("\n")[0]!.slice(0, 80); }
  if (toolName === "run_command") { const l = output.split("\n").filter((x) => x.trim() && !x.startsWith("Exit code:")); if (l.length > 0) return l[0]!.trim().slice(0, 80); }
  if (toolName === "find_files") { const l = output.split("\n").filter((x) => x.trim()); if (l.length > 0) return `${l.length} file(s)`; }
  return undefined;
}

function extractEditDiff(toolName: string, output: string): string | undefined {
  if (toolName !== "replace_in_file" && toolName !== "write_file") return undefined;
  if (!output || output.length < 20) return undefined;
  return output.split("\n").filter((l) => l.trim()).slice(0, 8).join("\n");
}

function summarizeParams(name: string, params: Record<string, unknown>): string {
  const path = params["path"] as string | undefined;
  if (path) return path;
  const command = params["command"] as string | undefined;
  if (command) return command.slice(0, 80);
  const pattern = params["pattern"] as string | undefined;
  if (pattern) return `"${pattern}"`;
  return "";
}

// ─── Component ──────────────────────────────────────────────

export function SingleShotApp({ bus, query, onQuery, model, approvalMode, onFinalOutput }: SingleShotAppProps): React.ReactElement {
  const { exit } = useApp();
  const [log, setLog] = useState<LogEntry[]>([]);
  const [status, setStatus] = useState<StatusBarProps>({
    model, cost: 0, inputTokens: 0, maxContextTokens: 0,
    iteration: 0, maxIterations: 0, approvalMode,
  });
  const [spinnerMessage, setSpinnerMessage] = useState<string | undefined>();
  const [running, setRunning] = useState(true);
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
      data: { name: group.name, count: group.count, summaries: group.summaries, iteration: statusRef.current.iteration, maxIterations: statusRef.current.maxIterations, status: group.lastSuccess ? "success" : "error", totalDurationMs: group.totalMs } satisfies ToolGroupEvent,
    });
    pendingGroupRef.current = null;
  }, [addLog]);

  const flushThinking = useCallback(() => {
    const text = textBufferRef.current.trim();
    if (text) {
      for (const line of text.split("\n").filter((l) => l.trim()).slice(0, 3)) {
        addLog({ id: nextId("r"), type: "reasoning", data: { text: line.slice(0, 120) } });
      }
    }
    if (thinkingStartRef.current !== null) {
      const d = Date.now() - thinkingStartRef.current;
      if (d > 500) addLog({ id: nextId("td"), type: "thinking-duration", data: { durationMs: d } });
      thinkingStartRef.current = null;
    }
    textBufferRef.current = "";
  }, [addLog]);

  // Wire events (same as App.tsx but no approval/input)
  useEffect(() => {
    const unsubs: Array<() => void> = [];

    unsubs.push(bus.on("iteration:start", (e) => {
      if (e.agentId) return;
      setStatus((s) => ({ ...s, iteration: e.iteration, maxIterations: e.maxIterations, inputTokens: e.estimatedTokens, maxContextTokens: e.maxContextTokens }));
    }));
    unsubs.push(bus.on("cost:update", (e) => {
      costAccumRef.current += e.totalCost;
      setStatus((s) => ({ ...s, cost: s.cost + e.totalCost }));
    }));
    unsubs.push(bus.on("message:assistant", (e) => {
      if (e.agentId) return;
      if (e.partial) {
        textBufferRef.current += e.content;
        if (e.chunk?.type === "thinking" && thinkingStartRef.current === null) thinkingStartRef.current = Date.now();
      }
    }));
    unsubs.push(bus.on("tool:before", (e) => {
      if (e.agentId || e.name === "delegate" || e.name === "update_plan" || e.name === "execute_tool_script") return;
      flushThinking();
      turnToolCountRef.current++;
      const summary = summarizeParams(e.name, e.params);
      if (pendingGroupRef.current?.name === e.name) { pendingGroupRef.current.count++; if (summary) pendingGroupRef.current.summaries.push(summary); return; }
      flushGroup();
      pendingGroupRef.current = { name: e.name, count: 1, summaries: summary ? [summary] : [], totalMs: 0, lastSuccess: true };
      addLog({ id: e.callId, type: "tool", data: { id: e.callId, name: e.name, summary, iteration: statusRef.current.iteration, maxIterations: statusRef.current.maxIterations, status: "running" } satisfies ToolEvent });
    }));
    unsubs.push(bus.on("tool:after", (e) => {
      if (e.agentId || e.name === "update_plan" || e.name === "execute_tool_script" || (e.name === "delegate" && e.result.metadata?.["agentMeta"])) return;
      if (pendingGroupRef.current?.name === e.name) {
        pendingGroupRef.current.totalMs += e.durationMs;
        if (!e.result.success) pendingGroupRef.current.lastSuccess = false;
        if (pendingGroupRef.current.count === 1) {
          const preview = e.result.success ? extractToolPreview(e.name, e.result.output) : undefined;
          const diff = e.result.success ? extractEditDiff(e.name, e.result.output) : undefined;
          addLog({ id: `${e.callId}-done`, type: "tool", data: { id: e.callId, name: e.name, summary: "", iteration: statusRef.current.iteration, maxIterations: statusRef.current.maxIterations, status: e.result.success ? "success" : "error", durationMs: e.durationMs, error: e.result.error ?? undefined, preview, diff } satisfies ToolEvent });
        }
        return;
      }
      const preview = e.result.success ? extractToolPreview(e.name, e.result.output) : undefined;
      const diff = e.result.success ? extractEditDiff(e.name, e.result.output) : undefined;
      addLog({ id: `${e.callId}-done`, type: "tool", data: { id: e.callId, name: e.name, summary: "", iteration: statusRef.current.iteration, maxIterations: statusRef.current.maxIterations, status: e.result.success ? "success" : "error", durationMs: e.durationMs, error: e.result.error ?? undefined, preview, diff } satisfies ToolEvent });
    }));
    unsubs.push(bus.on("message:tool", (e) => { if (!e.agentId && e.summaryOnly) addLog({ id: nextId("tm"), type: "info", data: `  ✓ ${e.content}` }); }));
    let lastPlanStepCount2 = 0;
    unsubs.push(bus.on("plan:updated", (e) => {
      const steps = e.steps as Array<{ description: string; status: string }>;
      if (steps.length !== lastPlanStepCount2) {
        addLog({ id: nextId("p"), type: "plan", data: steps });
        lastPlanStepCount2 = steps.length;
      } else {
        const active = steps.find((s) => s.status === "in_progress");
        const completed = steps.filter((s) => s.status === "completed").length;
        addLog({ id: nextId("ps"), type: "info", data: `── Plan ${completed}/${steps.length} ── ${active?.description ?? "All completed"}` });
      }
    }));
    unsubs.push(bus.on("context:compacting", () => { setSpinnerMessage("Compacting…"); }));
    unsubs.push(bus.on("context:compacted", (e) => { setSpinnerMessage(undefined); addLog({ id: nextId("c"), type: "compaction", data: e }); }));
    unsubs.push(bus.on("error", (e) => { addLog({ id: nextId("e"), type: "error", data: e }); }));

    // Subagents
    unsubs.push(bus.on("subagent:start", (e) => { setSubagents((p) => { const n = new Map(p); n.set(e.agentId, { agentId: e.agentId, agentType: String(e.agentType), laneLabel: e.laneLabel, status: "running", iteration: 0, startedAt: Date.now(), activity: "Starting…" }); return n; }); }));
    unsubs.push(bus.on("subagent:update", (e) => { setSubagents((p) => { const x = p.get(e.agentId); if (!x) return p; const newIter = e.iteration ?? x.iteration; const newAct = e.summary ?? e.toolName ?? x.activity; if (newIter === x.iteration && newAct === x.activity) return p; const n = new Map(p); n.set(e.agentId, { ...x, iteration: newIter, activity: newAct }); return n; }); }));
    unsubs.push(bus.on("subagent:end", (e) => { setSubagents((p) => { const x = p.get(e.agentId); if (!x) return p; const n = new Map(p); n.set(e.agentId, { ...x, status: "completed", iteration: e.iterations, quality: e.quality ? { score: e.quality.score, completeness: e.quality.completeness } : undefined }); return n; }); }));
    unsubs.push(bus.on("subagent:error", (e) => { setSubagents((p) => { const x = p.get(e.agentId); if (!x) return p; const n = new Map(p); n.set(e.agentId, { ...x, status: "error", error: e.error }); return n; }); addLog({ id: nextId("sx"), type: "error", data: { message: `Subagent failed: ${e.error}`, code: "SUBAGENT_ERROR" } }); }));

    return () => { for (const u of unsubs) u(); };
  }, [bus, addLog, flushThinking, flushGroup, nextId]);

  // Execute query on mount
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const result = await onQuery(query);
        if (cancelled) return;
        flushThinking();
        flushGroup();

        // Final output → stdout (outside Ink)
        const finalText = textBufferRef.current.trim() || result.lastText;
        if (finalText) onFinalOutput(finalText);

        addLog({
          id: nextId("summary"), type: "turn-summary",
          data: { iterations: result.iterations, toolCalls: turnToolCountRef.current, cost: costAccumRef.current, elapsedMs: Date.now() - turnStartRef.current },
        });
      } catch (err) {
        addLog({ id: nextId("e"), type: "error", data: { message: err instanceof Error ? err.message : String(err), code: "QUERY_ERROR" } });
      }
      setRunning(false);
      // Small delay to let final render flush, then exit
      setTimeout(() => { if (!cancelled) exit(); }, 100);
    })();
    return () => { cancelled = true; };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  let hasActiveSubagents = false;
  for (const a of subagents.values()) { if (a.status === "running") { hasActiveSubagents = true; break; } }

  return (
    <>
      <Static items={log}>
        {(entry) => <LogEntryView key={entry.id} entry={entry} />}
      </Static>
      {hasActiveSubagents && <SubagentPanel agents={subagents} />}
      <Spinner active={running} message={spinnerMessage} suffix={status.cost > 0 ? `$${status.cost.toFixed(4)}` : ""} />
      <StatusBar {...status} />
    </>
  );
}

// Log entry renderer (same as App.tsx)
const LogEntryView = React.memo(function LogEntryView({ entry }: { entry: LogEntry }): React.ReactElement | null {
  switch (entry.type) {
    case "tool": return <ToolDisplay event={entry.data as ToolEvent} />;
    case "tool-group": return <ToolGroupDisplay event={entry.data as ToolGroupEvent} />;
    case "reasoning": return <Text dimColor>  ℹ {(entry.data as { text: string }).text}</Text>;
    case "thinking-duration": return <ThinkingDuration durationMs={(entry.data as { durationMs: number }).durationMs} />;
    case "plan": return <PlanView steps={entry.data as PlanStep[]} />;
    case "error": { const e = entry.data as { message: string; code: string }; return <ErrorView message={e.message} code={e.code} />; }
    case "turn-summary": { const s = entry.data as { iterations: number; toolCalls: number; cost: number; elapsedMs: number }; return <TurnSummary iterations={s.iterations} toolCalls={s.toolCalls} cost={s.cost} elapsedMs={s.elapsedMs} />; }
    case "info": return <Text dimColor>{String(entry.data)}</Text>;
    case "compaction": { const e = entry.data as { tokensBefore: number; estimatedTokens: number }; const p = e.tokensBefore > 0 ? Math.round(((e.tokensBefore - e.estimatedTokens) / e.tokensBefore) * 100) : 0; return <Text dimColor>[context] {Math.round(e.tokensBefore / 1000)}k → {Math.round(e.estimatedTokens / 1000)}k ({p}%)</Text>; }
    default: return null;
  }
});
