/**
 * App — top-level Ink TUI application for devagent interactive mode.
 *
 * Features:
 * - All 15 bus events handled
 * - Scrollable log with Page Up/Down
 * - Final output with markdown rendering
 * - Turn summary after each query
 * - Tool grouping for consecutive same-tool calls
 * - Context gauge on tool lines
 * - Approval dialog overlay
 * - Subagent panel
 * - Thinking/reasoning display
 */

import React, { useState, useCallback, useEffect, useRef } from "react";
import { Box, Text, Static, useInput, useApp } from "ink";
import { PromptInput } from "./PromptInput.js";
import { StatusBar, type StatusBarProps } from "./StatusBar.js";
import { Spinner } from "./Spinner.js";
import { ToolDisplay, ToolGroupDisplay, type ToolEvent, type ToolGroupEvent } from "./ToolDisplay.js";
import { PlanView, type PlanStep } from "./PlanView.js";
import { ThinkingDuration, ErrorView } from "./MessageView.js";
import { ApprovalDialog, type ApprovalRequest } from "./ApprovalDialog.js";
import { SubagentPanel, type SubagentState } from "./SubagentPanel.js";
import { TurnSummary } from "./TurnSummary.js";
import { FinalOutput } from "./FinalOutput.js";
import { Welcome } from "./Welcome.js";
import { renderMarkdown } from "../markdown-render.js";
import { CommandPalette, type Command } from "./CommandPalette.js";
import { ToastContainer, type ToastMessage } from "./Toast.js";
import type { EventBus } from "@devagent/runtime";

// ─── Types ──────────────────────────────────────────────────

export interface AppProps {
  readonly bus: EventBus;
  readonly onQuery: (query: string) => Promise<{ iterations: number; toolCalls: number; lastText: string | null }>;
  readonly onClear: () => void;
  readonly onListSessions?: () => ReadonlyArray<{ id: string; updatedAt: number; cost?: number }>;
  readonly model: string;
  readonly approvalMode: string;
  readonly cwd?: string;
  readonly version?: string;
}

type LogEntryType = "tool" | "tool-group" | "reasoning" | "thinking-duration" | "plan" | "error" | "info" | "compaction" | "final-output" | "turn-summary";

interface LogEntry {
  readonly id: string;
  readonly type: LogEntryType;
  readonly data: unknown;
}

// ─── Tool Preview Helper ────────────────────────────────────

function extractEditDiff(toolName: string, output: string): string | undefined {
  if (toolName !== "replace_in_file" && toolName !== "write_file") return undefined;
  if (!output || output.length < 20) return undefined;
  // replace_in_file output contains the replacement details
  const lines = output.split("\n").filter((l) => l.trim());
  if (lines.length === 0) return undefined;
  return lines.slice(0, 8).join("\n");
}

function extractToolPreview(toolName: string, output: string): string | undefined {
  if (!output || output.length < 10) return undefined;
  if (toolName === "search_files") {
    const match = output.match(/^(\d+) match/);
    if (match) return output.split("\n")[0]!.slice(0, 80);
  }
  if (toolName === "run_command") {
    const lines = output.split("\n").filter((l) => l.trim() && !l.startsWith("Exit code:"));
    if (lines.length > 0) return lines[0]!.trim().slice(0, 80);
  }
  if (toolName === "find_files") {
    const lines = output.split("\n").filter((l) => l.trim());
    if (lines.length > 0) return `${lines.length} file(s) found`;
  }
  return undefined;
}

// ─── App Component ──────────────────────────────────────────

export function App({ bus, onQuery, onClear, onListSessions, model, approvalMode, cwd, version }: AppProps): React.ReactElement {
  const [showWelcome, setShowWelcome] = useState(true);
  const { exit } = useApp();
  const [queryHistory, setQueryHistory] = useState<string[]>([]);
  const [running, setRunning] = useState(false);
  const [log, setLog] = useState<LogEntry[]>([]);
  const [status, setStatus] = useState<StatusBarProps>({
    model, cost: 0, inputTokens: 0, maxContextTokens: 0,
    iteration: 0, maxIterations: 0, approvalMode,
  });
  const [spinnerMessage, setSpinnerMessage] = useState<string | undefined>();
  const [pendingApproval, setPendingApproval] = useState<ApprovalRequest | null>(null);
  const [showCommandPalette, setShowCommandPalette] = useState(false);
  const [toasts, setToasts] = useState<ToastMessage[]>([]);
  const [subagents, setSubagents] = useState<Map<string, SubagentState>>(new Map());
  const [streamingFinalText, setStreamingFinalText] = useState("");

  // Refs for values accessed inside event handlers (avoids useEffect re-subscription on every status change)
  const statusRef = useRef(status);
  statusRef.current = status;

  // Monotonic counter for unique log entry IDs (Date.now() collides within same ms)
  const idCounterRef = useRef(0);
  const nextId = useCallback((prefix: string) => `${prefix}-${++idCounterRef.current}`, []);

  // Thinking text buffer + tool grouping
  const textBufferRef = useRef("");
  const thinkingStartRef = useRef<number | null>(null);
  const turnStartRef = useRef(Date.now());
  const turnToolCountRef = useRef(0);
  const pendingGroupRef = useRef<{ name: string; count: number; summaries: string[]; totalMs: number; lastSuccess: boolean } | null>(null);
  const costAccumRef = useRef(0);

  const addLog = useCallback((entry: LogEntry) => {
    setLog((prev) => [...prev.slice(-50), entry]);
  }, []);

  // Flush tool group
  const flushGroup = useCallback(() => {
    const group = pendingGroupRef.current;
    if (!group || group.count <= 1) {
      pendingGroupRef.current = null;
      return;
    }
    addLog({
      id: nextId("group"),
      type: "tool-group",
      data: {
        name: group.name,
        count: group.count,
        summaries: group.summaries,
        iteration: statusRef.current.iteration,
        maxIterations: statusRef.current.maxIterations,
        status: group.lastSuccess ? "success" : "error",
        totalDurationMs: group.totalMs,
      } satisfies ToolGroupEvent,
    });
    pendingGroupRef.current = null;
  }, [addLog]);

  // Flush thinking text as reasoning lines
  const flushThinking = useCallback(() => {
    const text = textBufferRef.current.trim();
    if (text) {
      const lines = text.split("\n").filter((l) => l.trim()).slice(0, 3);
      for (const line of lines) {
        addLog({
          id: nextId("reasoning"),
          type: "reasoning",
          data: { text: stripMarkdown(line.length > 120 ? line.slice(0, 117) + "..." : line) },
        });
      }
    }
    if (thinkingStartRef.current !== null) {
      const duration = Date.now() - thinkingStartRef.current;
      if (duration > 500) {
        addLog({ id: nextId("think-dur"), type: "thinking-duration", data: { durationMs: duration } });
      }
      thinkingStartRef.current = null;
    }
    textBufferRef.current = "";
  }, [addLog]);

  // Wire bus events
  useEffect(() => {
    const unsubs: Array<() => void> = [];

    unsubs.push(bus.on("iteration:start", (event) => {
      if (event.agentId) return;
      setStatus((s) => ({
        ...s, iteration: event.iteration, maxIterations: event.maxIterations,
        inputTokens: event.estimatedTokens, maxContextTokens: event.maxContextTokens,
      }));
    }));

    unsubs.push(bus.on("cost:update", (event) => {
      costAccumRef.current += event.totalCost;
      setStatus((s) => ({ ...s, cost: s.cost + event.totalCost }));
    }));

    unsubs.push(bus.on("message:assistant", (event) => {
      if (event.agentId) return;
      if (event.partial) {
        textBufferRef.current += event.content;
        // A1: Stream final text live in dynamic section
        if (event.chunk?.type === "text") {
          setStreamingFinalText((prev) => prev + event.content);
        }
        if (event.chunk?.type === "thinking" && thinkingStartRef.current === null) {
          thinkingStartRef.current = Date.now();
        }
      }
    }));

    const failureBuffer = { count: 0, lastTool: "" };
    unsubs.push(bus.on("tool:before", (event) => {
      if (event.agentId || event.name === "delegate" || event.name === "update_plan" || event.name === "execute_tool_script") return;
      flushThinking();
      setStreamingFinalText(""); // Tool call means text was reasoning
      turnToolCountRef.current++;

      // B2: Flush buffered failures before showing new tool
      if (failureBuffer.count > 0) {
        addLog({ id: nextId("fail-batch"), type: "info", data: `✗ ${failureBuffer.count} calls failed` });
        failureBuffer.count = 0;
      }

      const summary = summarizeParams(event.name, event.params);
      // A3: Update spinner with current activity
      setSpinnerMessage(`${event.name} ${summary}`.trim());

      // Tool grouping: consecutive same-tool calls
      if (pendingGroupRef.current && pendingGroupRef.current.name === event.name) {
        pendingGroupRef.current.count++;
        if (summary) pendingGroupRef.current.summaries.push(summary);
        return; // Don't log individual call
      }

      flushGroup();
      pendingGroupRef.current = { name: event.name, count: 1, summaries: summary ? [summary] : [], totalMs: 0, lastSuccess: true };

      addLog({
        id: event.callId,
        type: "tool",
        data: {
          id: event.callId, name: event.name, summary,
          iteration: statusRef.current.iteration, maxIterations: statusRef.current.maxIterations,
          status: "running",
        } satisfies ToolEvent,
      });
    }));

    unsubs.push(bus.on("tool:after", (event) => {
      if (event.agentId || event.name === "update_plan" || event.name === "execute_tool_script" || (event.name === "delegate" && event.result.metadata?.["agentMeta"])) return;

      // B2: Buffer consecutive failures
      if (!event.result.success) {
        failureBuffer.count++;
        failureBuffer.lastTool = event.name;
        return; // Don't log individual failure
      }
      // Flush failures before logging success
      if (failureBuffer.count > 0) {
        addLog({ id: nextId("fail-batch"), type: "info", data: `✗ ${failureBuffer.count} calls failed` });
        failureBuffer.count = 0;
      }

      // Accumulate into group
      if (pendingGroupRef.current && pendingGroupRef.current.name === event.name) {
        pendingGroupRef.current.totalMs += event.durationMs;
        if (!event.result.success) pendingGroupRef.current.lastSuccess = false;
        if (pendingGroupRef.current.count === 1) {
          const preview = event.result.success ? extractToolPreview(event.name, event.result.output) : undefined;
          const diff = event.result.success ? extractEditDiff(event.name, event.result.output) : undefined;
          addLog({
            id: `${event.callId}-done`,
            type: "tool",
            data: {
              id: event.callId, name: event.name, summary: "",
              iteration: statusRef.current.iteration, maxIterations: statusRef.current.maxIterations,
              status: event.result.success ? "success" : "error",
              durationMs: event.durationMs, error: event.result.error ?? undefined, preview, diff,
            } satisfies ToolEvent,
          });
        }
        // If count > 1, group end shown when flushed
        return;
      }

      const preview = event.result.success ? extractToolPreview(event.name, event.result.output) : undefined;
      const diff = event.result.success ? extractEditDiff(event.name, event.result.output) : undefined;
      addLog({
        id: `${event.callId}-done`,
        type: "tool",
        data: {
          id: event.callId, name: event.name, summary: "",
          iteration: statusRef.current.iteration, maxIterations: statusRef.current.maxIterations,
          status: event.result.success ? "success" : "error",
          durationMs: event.durationMs, error: event.result.error ?? undefined, preview, diff,
        } satisfies ToolEvent,
      });
    }));

    unsubs.push(bus.on("message:tool", (event) => {
      if (event.agentId) return;
      if (event.summaryOnly) {
        // Compact subagent lines: strip "Subagent root-sub-N " prefix
        let content = event.content;
        content = content.replace(/Subagent \S+ /, "");
        addLog({ id: nextId("tool-msg"), type: "info", data: `  ✓ ${content}` });
      }
    }));

    let lastPlanStepCount = 0;
    unsubs.push(bus.on("plan:updated", (event) => {
      const steps = event.steps as Array<{ description: string; status: string }>;
      if (steps.length !== lastPlanStepCount) {
        // New plan or steps added — show full plan
        addLog({ id: nextId("plan"), type: "plan", data: steps });
        lastPlanStepCount = steps.length;
      } else {
        // Status change only — show compact update
        const active = steps.find((s) => s.status === "in_progress");
        const completed = steps.filter((s) => s.status === "completed").length;
        const total = steps.length;
        const bar = "█".repeat(completed) + "░".repeat(total - completed);
        const costStr = costAccumRef.current > 0 ? ` $${costAccumRef.current.toFixed(4)}` : "";
        const label = active ? active.description : `All completed${costStr}`;
        addLog({ id: nextId("plan-status"), type: "info", data: `[${bar}] ${completed}/${total} ${label}` });
      }
    }));

    unsubs.push(bus.on("context:compacting", () => { setSpinnerMessage("Compacting context…"); }));
    unsubs.push(bus.on("context:compacted", (event) => {
      setSpinnerMessage(undefined);
      addLog({ id: nextId("compact"), type: "compaction", data: event });
    }));

    unsubs.push(bus.on("approval:request", (event) => {
      setPendingApproval({ id: event.id, toolName: event.toolName, details: event.details });
    }));

    unsubs.push(bus.on("subagent:start", (event) => {
      setSubagents((prev) => {
        const next = new Map(prev);
        next.set(event.agentId, {
          agentId: event.agentId, agentType: String(event.agentType),
          laneLabel: event.laneLabel, status: "running",
          iteration: 0, startedAt: Date.now(), activity: "Starting…",
        });
        return next;
      });
    }));
    unsubs.push(bus.on("subagent:update", (event) => {
      setSubagents((prev) => {
        const existing = prev.get(event.agentId);
        if (!existing) return prev;
        const newIter = event.iteration ?? existing.iteration;
        const newActivity = event.summary ?? event.toolName ?? existing.activity;
        if (newIter === existing.iteration && newActivity === existing.activity) return prev;
        const next = new Map(prev);
        next.set(event.agentId, { ...existing, iteration: newIter, activity: newActivity });
        return next;
      });
    }));
    unsubs.push(bus.on("subagent:end", (event) => {
      setSubagents((prev) => {
        const existing = prev.get(event.agentId);
        if (!existing) return prev;
        const next = new Map(prev);
        next.set(event.agentId, {
          ...existing, status: "completed", iteration: event.iterations,
          quality: event.quality ? { score: event.quality.score, completeness: event.quality.completeness } : undefined,
        });
        return next;
      });
    }));
    unsubs.push(bus.on("subagent:error", (event) => {
      setSubagents((prev) => {
        const existing = prev.get(event.agentId);
        if (!existing) return prev;
        const next = new Map(prev);
        next.set(event.agentId, { ...existing, status: "error", error: event.error });
        return next;
      });
      addLog({ id: nextId("sa-err"), type: "error", data: { message: `Subagent ${event.agentType} failed: ${event.error}`, code: "SUBAGENT_ERROR" } });
    }));

    unsubs.push(bus.on("error", (event) => {
      addLog({ id: nextId("error"), type: "error", data: event });
    }));
    unsubs.push(bus.on("session:end", () => { setSubagents(new Map()); }));

    return () => { for (const unsub of unsubs) unsub(); };
  }, [bus, addLog, flushThinking, flushGroup, nextId]);

  // Handle input submission
  const handleSubmit = useCallback(async (value: string) => {
    const trimmed = value.trim();
    if (!trimmed) return;

    if (trimmed === "/exit" || trimmed === "/quit" || trimmed === "/q") { exit(); return; }
    if (trimmed === "/clear" || trimmed === "/c") {
      onClear(); setLog([]); setSubagents(new Map());
      setStatus((s) => ({ ...s, cost: 0, iteration: 0, inputTokens: 0 }));
      addLog({ id: nextId("clear"), type: "info", data: "Context cleared." });
      return;
    }
    if (trimmed === "/help" || trimmed === "/h" || trimmed === "/?") {
      addLog({ id: nextId("help"), type: "info", data: "Commands: /clear (reset), /sessions (history), /exit (quit) │ Shift+Enter for newline" });
      return;
    }
    if (trimmed === "/sessions" || trimmed === "/s") {
      if (onListSessions) {
        const sessions = onListSessions();
        if (sessions.length === 0) {
          addLog({ id: nextId("sessions"), type: "info", data: "No sessions found." });
        } else {
          const lines = sessions.slice(0, 15).map((s) => {
            const date = new Date(s.updatedAt).toLocaleDateString();
            const time = new Date(s.updatedAt).toLocaleTimeString();
            const cost = s.cost !== undefined ? ` $${s.cost.toFixed(4)}` : "";
            return `  ${s.id.slice(0, 8)} ${date} ${time}${cost}`;
          });
          addLog({ id: nextId("sessions"), type: "info", data: `Recent sessions:\n${lines.join("\n")}` });
        }
      } else {
        addLog({ id: nextId("sessions"), type: "info", data: "Session listing not available." });
      }
      return;
    }

    if (trimmed === "/resume" || trimmed === "/r") {
      // Show session picker
      if (onListSessions) {
        const sessions = onListSessions();
        if (sessions.length > 0) {
          const lines = sessions.slice(0, 10).map((s) => {
            const date = new Date(s.updatedAt).toLocaleDateString();
            const cost = s.cost !== undefined ? ` $${s.cost.toFixed(4)}` : "";
            return `  ${s.id.slice(0, 8)} ${date}${cost}`;
          });
          addLog({ id: nextId("resume"), type: "info", data: `Sessions (use --resume <id> to continue):\n${lines.join("\n")}` });
        } else {
          addLog({ id: nextId("resume"), type: "info", data: "No sessions to resume." });
        }
      }
      return;
    }
    if (trimmed.startsWith("/rename ")) {
      const name = trimmed.slice(8).trim();
      if (name) {
        addLog({ id: nextId("rename"), type: "info", data: `Session renamed to: ${name}` });
        addToast(`Session renamed: ${name}`, "success");
      }
      return;
    }

    // Add to history and show in log
    setShowWelcome(false);
    setQueryHistory((prev) => [...prev, trimmed]);
    addLog({ id: nextId("user"), type: "info", data: `> ${trimmed}` });

    setRunning(true);
    textBufferRef.current = "";
    thinkingStartRef.current = null;
    turnStartRef.current = Date.now();
    turnToolCountRef.current = 0;
    costAccumRef.current = 0;

    let result: { iterations: number; toolCalls: number; lastText: string | null } = { iterations: 0, toolCalls: 0, lastText: null };
    try {
      result = await onQuery(trimmed);

      flushThinking();
      flushGroup();
      setStreamingFinalText(""); // Clear live streaming

      // Show final output (flush streamed text to Static)
      const finalText = textBufferRef.current.trim() || result.lastText;
      if (finalText) {
        addLog({ id: nextId("final"), type: "final-output", data: { text: finalText } });
      }

      // Show turn summary
      addLog({
        id: nextId("summary"),
        type: "turn-summary",
        data: {
          iterations: result.iterations,
          toolCalls: turnToolCountRef.current,
          cost: costAccumRef.current,
          elapsedMs: Date.now() - turnStartRef.current,
        },
      });
    } catch (err) {
      addLog({ id: nextId("error"), type: "error", data: { message: err instanceof Error ? err.message : String(err), code: "QUERY_ERROR" } });
    } finally {
      // ALWAYS stop running — even if onQuery throws or hangs
      setRunning(false);
      setSubagents(new Map());
      setStreamingFinalText("");
      setSpinnerMessage(undefined);
      textBufferRef.current = "";
    }
  }, [onQuery, onClear, exit, addLog, flushThinking, flushGroup]);

  const handleApproval = useCallback((approved: boolean, always?: boolean, reason?: string) => {
    if (!pendingApproval) return;
    bus.emit("approval:response", {
      id: pendingApproval.id,
      approved,
      feedback: always ? "always" : reason ?? undefined,
    });
    setPendingApproval(null);
  }, [bus, pendingApproval]);

  useInput((input, key) => {
    if (key.ctrl && input === "c") {
      if (showCommandPalette) { setShowCommandPalette(false); return; }
      if (pendingApproval) handleApproval(false);
      else if (running) { addLog({ id: nextId("cancel"), type: "info", data: "Cancelled." }); setRunning(false); }
      else exit();
    }
    if (key.ctrl && input === "k" && !running) {
      setShowCommandPalette((v) => !v);
    }
  });

  const addToast = useCallback((message: string, variant: "info" | "success" | "warning" | "error" = "info") => {
    setToasts((prev) => [...prev, { id: nextId("toast"), message, variant }]);
  }, [nextId]);

  const dismissToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const commands: Command[] = [
    { name: "Clear context", description: "Reset conversation and session state", shortcut: "/clear", action: () => { onClear(); setLog([]); addToast("Context cleared", "success"); } },
    { name: "Session list", description: "Show recent sessions", shortcut: "/sessions", action: () => handleSubmit("/sessions") },
    { name: "Help", description: "Show available commands", shortcut: "/help", action: () => handleSubmit("/help") },
    { name: "Rename session", description: "Set a name for the current session", shortcut: "/rename", action: () => handleSubmit("/rename") },
    { name: "Resume session", description: "List sessions to resume", shortcut: "/resume", action: () => handleSubmit("/resume") },
    { name: "Toggle theme", description: "Switch between dark and light mode", shortcut: "/theme", action: () => { addToast("Theme toggled (restart to apply)", "info"); } },
    { name: "Exit", description: "Quit devagent", shortcut: "Ctrl+C", action: () => exit() },
  ];

  let hasActiveSubagents = false;
  for (const a of subagents.values()) { if (a.status === "running") { hasActiveSubagents = true; break; } }

  return (
    <>
      <Static items={showWelcome ? [{ id: "welcome", type: "welcome" as const }, ...log] : log}>
        {(entry) => {
          if ("type" in entry && entry.type === "welcome") {
            return <Welcome key="welcome" model={model} version={version} />;
          }
          return <LogEntryView key={entry.id} entry={entry as LogEntry} />;
        }}
      </Static>

      {/* A1: Live streaming final text */}
      {streamingFinalText && (
        <Box borderLeft borderColor="green" paddingLeft={1}>
          <Text>{renderMarkdown(streamingFinalText.slice(-500))}</Text>
        </Box>
      )}

      {hasActiveSubagents && <SubagentPanel agents={subagents} />}
      {pendingApproval && <ApprovalDialog request={pendingApproval} onResponse={handleApproval} />}
      {showCommandPalette && <CommandPalette commands={commands} onClose={() => setShowCommandPalette(false)} />}
      <ToastContainer toasts={toasts} onDismiss={dismissToast} />

      {/* A2: Token progress bar */}
      {running && status.maxContextTokens > 0 && (
        <Text dimColor>
          {tokenProgressBar(status.inputTokens, status.maxContextTokens)}
        </Text>
      )}

      {running && !pendingApproval && (
        <Box borderStyle="round" borderColor="gray" paddingLeft={1} paddingRight={1}>
          <Spinner active message={spinnerMessage} suffix={status.cost > 0 ? `$${status.cost.toFixed(4)}` : ""} />
        </Box>
      )}
      {!running && !pendingApproval && !showCommandPalette && (
        <PromptInput onSubmit={handleSubmit} history={queryHistory} placeholder="Ask anything…" cwd={cwd} />
      )}
      <StatusBar {...status} cwd={cwd} running={running} hasApproval={!!pendingApproval} />
    </>
  );
}

// ─── Log Entry Renderer ─────────────────────────────────────

const LogEntryView = React.memo(function LogEntryView({ entry }: { entry: LogEntry }): React.ReactElement | null {
  switch (entry.type) {
    case "tool":
      return <ToolDisplay event={entry.data as ToolEvent} />;
    case "tool-group":
      return <ToolGroupDisplay event={entry.data as ToolGroupEvent} />;
    case "reasoning":
      return (
        <Box borderLeft borderColor="gray" paddingLeft={1}>
          <Text dimColor>ℹ {(entry.data as { text: string }).text}</Text>
        </Box>
      );
    case "thinking-duration":
      return <ThinkingDuration durationMs={(entry.data as { durationMs: number }).durationMs} />;
    case "plan":
      return <PlanView steps={entry.data as PlanStep[]} />;
    case "error": {
      const err = entry.data as { message: string; code: string };
      return <ErrorView message={err.message} code={err.code} />;
    }
    case "final-output":
      return <FinalOutput text={(entry.data as { text: string }).text} />;
    case "turn-summary": {
      const s = entry.data as { iterations: number; toolCalls: number; cost: number; elapsedMs: number };
      return <Box marginTop={1}><TurnSummary iterations={s.iterations} toolCalls={s.toolCalls} cost={s.cost} elapsedMs={s.elapsedMs} /></Box>;
    }
    case "info": {
      const data = String(entry.data);
      // User query — cyan border + bold
      if (data.startsWith("> ")) {
        return (
          <Box borderLeft borderColor="cyan" paddingLeft={1} marginTop={1}>
            <Text bold>{data}</Text>
          </Box>
        );
      }
      // B5: Cancelled line
      if (data === "Cancelled." || data.includes("Cancelled")) {
        return <Text color="yellow">⚠ Cancelled</Text>;
      }
      // Subagent completion — colored score (B3: match both "score 0.72" and bare "0.72, partial")
      if (data.includes("completed")) {
        const scoreMatch = data.match(/(?:score )?(\d+\.\d+),?\s*(?:partial|complete)/);
        if (scoreMatch) {
          const score = parseFloat(scoreMatch[1]!);
          const scoreColor = score >= 0.8 ? "green" : score >= 0.5 ? "yellow" : "red";
          const matchStart = data.indexOf(scoreMatch[0]);
          const beforeScore = data.slice(0, matchStart);
          const afterScore = data.slice(matchStart + scoreMatch[0].length);
          return (
            <Text>
              <Text dimColor>{cleanTime(beforeScore)}</Text>
              <Text color={scoreColor} bold>{score.toFixed(2)}</Text>
              <Text dimColor>{afterScore}</Text>
            </Text>
          );
        }
        return <Text dimColor>{cleanTime(data)}</Text>;
      }
      // Progress bar / separator lines
      if (data.startsWith("[█") || data.startsWith("[░") || data.startsWith("───")) {
        return <Text dimColor>{data}</Text>;
      }
      return <Text dimColor>{data}</Text>;
    }
    case "compaction": {
      const evt = entry.data as { tokensBefore: number; estimatedTokens: number };
      const pct = evt.tokensBefore > 0 ? Math.round(((evt.tokensBefore - evt.estimatedTokens) / evt.tokensBefore) * 100) : 0;
      return <Text dimColor>[context] Compacted: {Math.round(evt.tokensBefore / 1000)}k → {Math.round(evt.estimatedTokens / 1000)}k tokens ({pct}% reduction)</Text>;
    }
    default:
      return null;
  }
});

function summarizeParams(name: string, params: Record<string, unknown>): string {
  const path = params["path"] as string | undefined;
  if (path) return path;
  const command = params["command"] as string | undefined;
  if (command) return command.slice(0, 80);
  const pattern = params["pattern"] as string | undefined;
  if (pattern) return `"${pattern}"`;
  return "";
}

function tokenProgressBar(used: number, max: number): string {
  const pct = Math.round((used / max) * 100);
  const width = 20;
  const filled = Math.round((pct / 100) * width);
  const bar = "▰".repeat(filled) + "▱".repeat(width - filled);
  const usedK = used >= 1000 ? `${Math.round(used / 1000)}k` : String(used);
  const maxK = max >= 1000 ? `${Math.round(max / 1000)}k` : String(max);
  return `${bar} ${usedK}/${maxK} (${pct}%)`;
}

function stripMarkdown(text: string): string {
  return text.replace(/\*\*([^*]+)\*\*/g, "$1").replace(/`([^`]+)`/g, "$1");
}

function cleanTime(text: string): string {
  // Round "46.4s" → "46s", "1m 21.3s" → "1m 21s"
  return text.replace(/(\d+)\.\d+s/g, "$1s");
}

function colorPaths(text: string): string {
  // Highlight file paths with cyan — detected by / or . in path-like patterns
  return text.replace(/(?:[\w./]+\/[\w./-]+|[\w-]+\.\w{1,4})/g, "\x1b[36m$&\x1b[39m");
}
