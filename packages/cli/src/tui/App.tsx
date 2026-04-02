/**
 * App — top-level Ink TUI application for devagent interactive mode.
 */

import React, { useState, useCallback, useRef, useEffect } from "react";
import { Box, Text, Static, useInput, useApp } from "ink";
import { PromptInput } from "./PromptInput.js";
import { StatusBar } from "./StatusBar.js";
import { Spinner } from "./Spinner.js";
import { ApprovalDialog, type ApprovalRequest } from "./ApprovalDialog.js";
import { SubagentPanel } from "./SubagentPanel.js";
import { CommandPalette, type Command } from "./CommandPalette.js";
import { ToastContainer, type ToastMessage } from "./Toast.js";
import { Welcome } from "./Welcome.js";
import { LogEntryView } from "./LogEntryView.js";
import { useAgentLog } from "./useAgentLog.js";
import { tokenProgressBar } from "./shared.js";
import type { LogEntry } from "./shared.js";
import type { EventBus } from "@devagent/runtime";

// Show last N lines of streaming text without expensive markdown parsing
function renderStreamingPreview(text: string): string {
  const lines = text.split("\n");
  return (lines.length > 20 ? lines.slice(-20) : lines).join("\n");
}

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

// ─── App Component ──────────────────────────────────────────

export function App({ bus, onQuery, onClear, onListSessions, model, approvalMode, cwd, version }: AppProps): React.ReactElement {
  const [showWelcome, setShowWelcome] = useState(true);
  const { exit } = useApp();
  const [queryHistory, setQueryHistory] = useState<string[]>([]);
  const [running, setRunning] = useState(false);
  const [pendingApproval, setPendingApproval] = useState<ApprovalRequest | null>(null);
  const [showCommandPalette, setShowCommandPalette] = useState(false);
  const [toasts, setToasts] = useState<ToastMessage[]>([]);
  const [streamingFinalText, setStreamingFinalText] = useState("");
  const [streamingDone, setStreamingDone] = useState(false);

  // Throttle streaming updates: accumulate in ref, flush to state every 50ms
  const streamingBufferRef = useRef("");
  const throttleRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const startStreamingThrottle = useCallback(() => {
    streamingBufferRef.current = "";
    if (throttleRef.current) clearInterval(throttleRef.current);
    throttleRef.current = setInterval(() => {
      setStreamingFinalText(streamingBufferRef.current);
    }, 50);
  }, []);

  const stopStreamingThrottle = useCallback(() => {
    if (throttleRef.current) {
      clearInterval(throttleRef.current);
      throttleRef.current = null;
    }
    streamingBufferRef.current = "";
  }, []);

  // Cleanup on unmount
  useEffect(() => () => { if (throttleRef.current) clearInterval(throttleRef.current); }, []);

  const {
    log, status, subagents, spinnerMessage,
    addLog, flushThinking, flushGroup, nextId,
    setStatus, setSubagents, setSpinnerMessage,
    refs,
  } = useAgentLog({
    bus, model, approvalMode,
    collapseFailures: true,
    compactPlanProgress: true,
    onStreamingText: (content) => { streamingBufferRef.current += content; },
    onToolStart: () => { stopStreamingThrottle(); setStreamingFinalText(""); },
    onApproval: (e) => setPendingApproval({ id: e.id, toolName: e.toolName, details: e.details }),
  });

  // Handle input submission
  const handleSubmit = useCallback(async (value: string) => {
    const trimmed = value.trim();
    if (!trimmed) return;

    if (trimmed === "/exit" || trimmed === "/quit" || trimmed === "/q") { exit(); return; }
    if (trimmed === "/clear" || trimmed === "/c") {
      onClear(); setSubagents(new Map());
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
    refs.textBuffer.current = "";
    refs.thinkingStart.current = null;
    refs.turnStart.current = Date.now();
    refs.turnToolCount.current = 0;
    refs.costAccum.current = 0;
    startStreamingThrottle();

    let result: { iterations: number; toolCalls: number; lastText: string | null } = { iterations: 0, toolCalls: 0, lastText: null };
    try {
      result = await onQuery(trimmed);
      flushThinking();
      flushGroup();
      stopStreamingThrottle();

      // Hide streaming preview and add Static entry atomically
      setStreamingDone(true);
      const finalText = refs.textBuffer.current.trim() || result.lastText;
      if (finalText) {
        addLog({ id: nextId("final"), type: "final-output", data: { text: finalText } });
      }

      addLog({
        id: nextId("summary"), type: "turn-summary",
        data: {
          iterations: result.iterations, toolCalls: refs.turnToolCount.current,
          cost: refs.costAccum.current, elapsedMs: Date.now() - refs.turnStart.current,
        },
      });
    } catch (err) {
      addLog({ id: nextId("error"), type: "error", data: { message: err instanceof Error ? err.message : String(err), code: "QUERY_ERROR" } });
    } finally {
      stopStreamingThrottle();
      setRunning(false);
      setSubagents(new Map());
      setStreamingFinalText("");
      setStreamingDone(false);
      setSpinnerMessage(undefined);
      refs.textBuffer.current = "";
    }
  }, [onQuery, onClear, exit, addLog, flushThinking, flushGroup, nextId, refs, setStatus, setSubagents, setSpinnerMessage, startStreamingThrottle, stopStreamingThrottle]);

  const handleApproval = useCallback((approved: boolean, always?: boolean, reason?: string) => {
    if (!pendingApproval) return;
    bus.emit("approval:response", {
      id: pendingApproval.id, approved,
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
    { name: "Clear context", description: "Reset conversation and session state", shortcut: "/clear", action: () => { onClear(); setSubagents(new Map()); addToast("Context cleared", "success"); } },
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

      {streamingFinalText && !streamingDone && (
        <Box borderLeft borderColor="green" paddingLeft={1}>
          <Text>{renderStreamingPreview(streamingFinalText)}</Text>
        </Box>
      )}

      {hasActiveSubagents && <SubagentPanel agents={subagents} />}
      {pendingApproval && <ApprovalDialog request={pendingApproval} onResponse={handleApproval} />}
      {showCommandPalette && <CommandPalette commands={commands} onClose={() => setShowCommandPalette(false)} />}
      <ToastContainer toasts={toasts} onDismiss={dismissToast} />

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
