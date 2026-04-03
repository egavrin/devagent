/**
 * App — top-level Ink TUI application for devagent interactive mode.
 */

import React, { useState, useCallback } from "react";
import { Box, Text, Static, useInput, useApp } from "ink";
import { SafetyMode } from "@devagent/runtime";
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
import { cycleApprovalMode, getApprovalModeColor, type InteractiveQueryResult, tokenProgressBar } from "./shared.js";
import type { LogEntry } from "./shared.js";
import type { EventBus } from "@devagent/runtime";

export const TUI_HELP_MESSAGE = "Commands: /clear (reset), /continue (resume work), /sessions (history), /exit (quit) │ Embedded shortcuts can appear anywhere: /review, /simplify │ Shift+Enter for newline │ Shift+Tab toggles safety mode";
export const ITERATION_LIMIT_NOTICE = "Iteration limit exhausted. Type /continue to proceed.";

// ─── Types ──────────────────────────────────────────────────

export interface AppProps {
  readonly bus: EventBus;
  readonly onQuery: (query: string) => Promise<InteractiveQueryResult>;
  readonly onClear: () => void;
  readonly onCycleApprovalMode: (mode: SafetyMode) => void;
  readonly onListSessions?: () => ReadonlyArray<{ id: string; updatedAt: number; cost?: number }>;
  readonly model: string;
  readonly approvalMode: string;
  readonly cwd?: string;
  readonly version?: string;
}

export interface TranscriptViewProps {
  readonly showWelcome: boolean;
  readonly log: ReadonlyArray<LogEntry>;
  readonly model: string;
  readonly version?: string;
}

export function TranscriptView({ showWelcome, log, model, version }: TranscriptViewProps): React.ReactElement {
  return (
    <>
      {showWelcome && <Welcome model={model} version={version} />}
      <Static items={[...log]}>
        {(entry) => <LogEntryView key={entry.id} entry={entry} />}
      </Static>
    </>
  );
}

// ─── App Component ──────────────────────────────────────────

export function App({ bus, onQuery, onClear, onCycleApprovalMode, onListSessions, model, approvalMode, cwd, version }: AppProps): React.ReactElement {
  const [showWelcome, setShowWelcome] = useState(true);
  const { exit } = useApp();
  const [queryHistory, setQueryHistory] = useState<string[]>([]);
  const [running, setRunning] = useState(false);
  const [pendingApproval, setPendingApproval] = useState<ApprovalRequest | null>(null);
  const [showCommandPalette, setShowCommandPalette] = useState(false);
  const [toasts, setToasts] = useState<ToastMessage[]>([]);
  const [currentApprovalMode, setCurrentApprovalMode] = useState(approvalMode);

  const {
    log, status, subagents, spinnerMessage,
    addLog, flushThinking, flushGroup, nextId,
    setStatus, setSubagents, setSpinnerMessage,
    refs,
  } = useAgentLog({
    bus, model, approvalMode: currentApprovalMode,
    collapseFailures: true,
    compactPlanProgress: true,
    onApproval: (e) => setPendingApproval({ id: e.id, toolName: e.toolName, details: e.details }),
  });

  const addToast = useCallback((message: string, variant: "info" | "success" | "warning" | "error" = "info") => {
    setToasts((prev) => [...prev, { id: nextId("toast"), message, variant }]);
  }, [nextId]);

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
    if (trimmed === "/continue") {
      return handleSubmit("continue");
    }
    if (trimmed === "/help" || trimmed === "/h" || trimmed === "/?") {
      addLog({ id: nextId("help"), type: "info", data: TUI_HELP_MESSAGE });
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

    let result: InteractiveQueryResult = { iterations: 0, toolCalls: 0, lastText: null, status: "success" };
    try {
      result = await onQuery(trimmed);
      flushThinking();
      flushGroup();
      const finalText = result.lastText;
      if (finalText) {
        addLog({ id: nextId("final"), type: "final-output", data: { text: finalText } });
      }

      addLog({
        id: nextId("summary"), type: "turn-summary",
        data: {
          iterations: result.iterations, toolCalls: result.toolCalls,
          cost: refs.costAccum.current, elapsedMs: Date.now() - refs.turnStart.current,
        },
      });

      if (result.status === "budget_exceeded") {
        addLog({ id: nextId("budget"), type: "info", data: ITERATION_LIMIT_NOTICE });
        addToast(ITERATION_LIMIT_NOTICE, "warning");
      }
    } catch (err) {
      addLog({ id: nextId("error"), type: "error", data: { message: err instanceof Error ? err.message : String(err), code: "QUERY_ERROR" } });
    } finally {
      setRunning(false);
      setSubagents(new Map());
      setSpinnerMessage(undefined);
      refs.textBuffer.current = "";
    }
  }, [onQuery, onClear, exit, addLog, addToast, flushThinking, flushGroup, nextId, refs, setStatus, setSubagents, setSpinnerMessage, currentApprovalMode]);

  const handleCycleApprovalMode = useCallback(() => {
    if (running || pendingApproval || showCommandPalette) {
      return;
    }
    const nextMode = cycleApprovalMode(currentApprovalMode);
    setCurrentApprovalMode(nextMode);
    setStatus((s) => ({ ...s, approvalMode: nextMode }));
    onCycleApprovalMode(nextMode);
    addToast(`Safety: ${nextMode}`, "info");
  }, [addToast, currentApprovalMode, onCycleApprovalMode, pendingApproval, running, setStatus, showCommandPalette]);

  const handleApproval = useCallback((approved: boolean, session?: boolean, reason?: string) => {
    if (!pendingApproval) return;
    bus.emit("approval:response", {
      id: pendingApproval.id, approved,
      feedback: session ? "session" : reason ?? undefined,
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

  const dismissToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const commands: Command[] = [
    { name: "Clear context", description: "Reset conversation and session state", shortcut: "/clear", action: () => { onClear(); setSubagents(new Map()); addToast("Context cleared", "success"); } },
    { name: "Continue", description: "Continue the current session after a pause or budget limit", shortcut: "/continue", action: () => handleSubmit("/continue") },
    { name: "Session list", description: "Show recent sessions", shortcut: "/sessions", action: () => handleSubmit("/sessions") },
    { name: "Help", description: "Show available commands", shortcut: "/help", action: () => handleSubmit("/help") },
    { name: "Rename session", description: "Set a name for the current session", shortcut: "/rename", action: () => handleSubmit("/rename") },
    { name: "Review local changes", description: "Insert or run the embedded /review command anywhere in a prompt", shortcut: "/review", action: () => handleSubmit("/review") },
    { name: "Simplify local changes", description: "Insert or run the embedded /simplify command anywhere in a prompt", shortcut: "/simplify", action: () => handleSubmit("/simplify") },
    { name: "Resume session", description: "List sessions to resume", shortcut: "/resume", action: () => handleSubmit("/resume") },
    { name: "Safety mode", description: "Toggle default and autopilot", shortcut: "Shift+Tab", action: handleCycleApprovalMode },
    { name: "Exit", description: "Quit devagent", shortcut: "Ctrl+C", action: () => exit() },
  ];

  let hasActiveSubagents = false;
  for (const a of subagents.values()) { if (a.status === "running") { hasActiveSubagents = true; break; } }

  return (
    <>
      <TranscriptView showWelcome={showWelcome} log={log} model={model} version={version} />
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
        <Box borderStyle="round" borderColor={getApprovalModeColor(currentApprovalMode)} paddingLeft={1} paddingRight={1}>
          <Spinner active message={spinnerMessage} suffix={status.cost > 0 ? `$${status.cost.toFixed(4)}` : ""} />
        </Box>
      )}
      {!running && !pendingApproval && !showCommandPalette && (
        <PromptInput
          onSubmit={handleSubmit}
          onCycleApprovalMode={handleCycleApprovalMode}
          history={queryHistory}
          placeholder="Ask anything…"
          cwd={cwd}
          approvalMode={currentApprovalMode}
        />
      )}
      <StatusBar {...status} cwd={cwd} running={running} hasApproval={!!pendingApproval} />
    </>
  );
}
