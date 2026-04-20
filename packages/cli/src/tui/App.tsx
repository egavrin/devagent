/**
 * App — top-level Ink TUI application for devagent interactive mode.
 */

import { Box, Static, Text, useApp, useInput } from "ink";
import React, { useCallback, useState } from "react";

import { ApprovalDialog, type ApprovalRequest } from "./ApprovalDialog.js";
import { CommandPalette, type Command } from "./CommandPalette.js";
import { LogEntryView } from "./LogEntryView.js";
import { PromptInput } from "./PromptInput.js";
import {
  cycleApprovalMode,
  getApprovalModeColor,
  type InteractiveQueryResult,
  type TranscriptNode,
} from "./shared.js";
import { Spinner } from "./Spinner.js";
import { StatusBar } from "./StatusBar.js";
import { SubagentPanel } from "./SubagentPanel.js";
import { ToastContainer, type ToastMessage } from "./Toast.js";
import { useAgentLog } from "./useAgentLog.js";
import { Welcome } from "./Welcome.js";
import { renderSessionPreview, type SessionPreview } from "../session-preview.js";
import type { PresentedTurn } from "../transcript-composer.js";
import {
  makeInfoPart,
  makeErrorPart,
  makeFinalOutputPart,
  makeTurnSummaryPart,
} from "../transcript-presenter.js";
import type { EventBus, SafetyMode } from "@devagent/runtime";

export const TUI_HELP_MESSAGE = "Commands: /clear (reset), /continue (resume work), /sessions (history), /exit (quit) │ Embedded shortcuts can appear anywhere: /review, /simplify │ Shift+Enter or Option+Enter for newline │ Shift+Tab toggles safety mode";
export const ITERATION_LIMIT_NOTICE = "Iteration limit exhausted. Type /continue to proceed.";

// ─── Types ──────────────────────────────────────────────────

export interface AppProps {
  readonly bus: EventBus;
  readonly onQuery: (query: string) => Promise<InteractiveQueryResult>;
  readonly onClear: () => void;
  readonly onCycleApprovalMode: (mode: SafetyMode) => void;
  readonly onListSessions?: () => ReadonlyArray<SessionPreview>;
  readonly model: string;
  readonly approvalMode: string;
  readonly cwd?: string;
  readonly version?: string;
}

interface TranscriptViewProps {
  readonly showWelcome: boolean;
  readonly transcriptNodes: ReadonlyArray<TranscriptNode>;
  readonly model: string;
  readonly version?: string;
  readonly keepLatestNodeLive?: boolean;
}

function isRunningTurnNode(node: TranscriptNode | undefined): node is Extract<TranscriptNode, { readonly kind: "turn" }> {
  return node?.kind === "turn" && node.turn.status === "running";
}

export function TranscriptView({
  showWelcome,
  transcriptNodes,
  model,
  version,
  keepLatestNodeLive = false,
}: TranscriptViewProps): React.ReactElement {
  const lastNode = transcriptNodes.at(-1);
  const liveTailNode = lastNode && (keepLatestNodeLive || isRunningTurnNode(lastNode))
    ? lastNode
    : null;
  const staticNodes = liveTailNode ? transcriptNodes.slice(0, -1) : transcriptNodes;

  return (
    <>
      {showWelcome && <Welcome model={model} version={version} />}
      <Static items={[...staticNodes]}>
        {(node) => <TranscriptNodeView key={node.id} node={node} />}
      </Static>
      {liveTailNode ? <TranscriptNodeView node={liveTailNode} /> : null}
    </>
  );
}

function TranscriptNodeView({ node }: { readonly node: TranscriptNode }): React.ReactElement {
  if (node.kind === "part") {
    return <LogEntryView entry={{ id: node.id, part: node.part }} />;
  }
  return <TurnView turn={node.turn} />;
}

function TurnView({ turn }: { readonly turn: PresentedTurn }): React.ReactElement {
  const statusColor = turn.status === "error"
    ? "red"
    : turn.status === "budget_exceeded"
      ? "yellow"
      : turn.status === "running"
        ? "cyan"
        : "green";
  const statusLabel = turn.status === "running"
    ? "running"
    : turn.status === "budget_exceeded"
      ? "budget"
      : turn.status;

  return (
    <Box flexDirection="column" marginTop={1}>
      <Text>
        <Text dimColor>  ╭─ </Text>
        <Text color={statusColor}>{statusLabel}</Text>
        <Text> </Text>
        <Text bold>{turn.userText}</Text>
      </Text>
      {turn.entries.map((entry) => (
        <LogEntryView key={entry.id} entry={{ id: entry.id, part: entry.part }} />
      ))}
    </Box>
  );
}

export function renderSessionsCommandOutput(sessions: ReadonlyArray<SessionPreview>): string {
  if (sessions.length === 0) {
    return "No sessions found.";
  }

  const lines = sessions.slice(0, 15).map((s) => renderSessionPreview(s));
  return `Recent sessions:\n${lines.join("\n")}`;
}

export function renderResumeCommandOutput(sessions: ReadonlyArray<SessionPreview>): string {
  if (sessions.length === 0) {
    return "No sessions to resume.";
  }

  const lines = sessions.slice(0, 10).map((s) => renderSessionPreview(s));
  return `Sessions (use --resume <id> to continue):\n${lines.join("\n")}`;
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
    transcriptNodes, status, subagents, spinnerMessage,
    appendStandalonePart, startTurn, appendTurnPart, completeTurn, flushThinking, flushGroup, nextId,
    setStatus, setSubagents, setSpinnerMessage,
    refs,
  } = useAgentLog({
    bus, model,
    collapseFailures: true,
    compactPlanProgress: true,
    onApproval: (e) => setPendingApproval({ id: e.id, toolName: e.toolName, details: e.details }),
  });

  const addToast = useCallback((message: string, variant: "info" | "success" | "warning" | "error" = "info") => {
    setToasts((prev) => [...prev, { id: nextId("toast"), message, variant }]);
  }, [nextId]);

  const clearSubagents = useCallback(() => {
    setSubagents((prev) => (prev.size === 0 ? prev : new Map()));
  }, [setSubagents]);

  // Handle input submission
  const handleSubmit = useCallback(async (value: string) => {
    const trimmed = value.trim();
    if (!trimmed) return;

    if (trimmed === "/exit" || trimmed === "/quit" || trimmed === "/q") { exit(); return; }
    if (trimmed === "/clear" || trimmed === "/c") {
      onClear(); clearSubagents();
      setStatus((s) => ({ ...s, cost: 0, iteration: 0, inputTokens: 0 }));
      appendStandalonePart(nextId("clear"), makeInfoPart("info", ["Context cleared."]));
      return;
    }
    if (trimmed === "/continue") {
      return handleSubmit("continue");
    }
    if (trimmed === "/help" || trimmed === "/h" || trimmed === "/?") {
      appendStandalonePart(nextId("help"), makeInfoPart("info", [TUI_HELP_MESSAGE]));
      return;
    }
    if (trimmed === "/sessions" || trimmed === "/s") {
      if (onListSessions) {
        appendStandalonePart(nextId("sessions"), makeInfoPart("sessions", renderSessionsCommandOutput(onListSessions()).split("\n")));
      } else {
        appendStandalonePart(nextId("sessions"), makeInfoPart("sessions", ["Session listing not available."]));
      }
      return;
    }
    if (trimmed === "/resume" || trimmed === "/r") {
      if (onListSessions) {
        appendStandalonePart(nextId("resume"), makeInfoPart("sessions", renderResumeCommandOutput(onListSessions()).split("\n")));
      }
      return;
    }

    // Add to history and show in log
    setShowWelcome(false);
    setQueryHistory((prev) => [...prev, trimmed]);

    setRunning(true);
    refs.textBuffer.current = "";
    refs.thinkingStart.current = null;
    refs.turnStart.current = Date.now();
    refs.turnToolCount.current = 0;
    refs.costAccum.current = 0;
    startTurn(nextId("turn"), trimmed, refs.turnStart.current);

    let result: InteractiveQueryResult = { iterations: 0, toolCalls: 0, lastText: null, status: "success" };
    try {
      result = await onQuery(trimmed);
      flushThinking();
      flushGroup();
      const finalText = result.lastText;
      if (finalText) {
        appendTurnPart(nextId("final"), makeFinalOutputPart(finalText));
      }

      if (result.status === "budget_exceeded") {
        appendTurnPart(nextId("budget"), makeInfoPart("status", [ITERATION_LIMIT_NOTICE]));
        addToast(ITERATION_LIMIT_NOTICE, "warning");
      }

      completeTurn(
        nextId("summary"),
        makeTurnSummaryPart({
          iterations: result.iterations, toolCalls: result.toolCalls,
          cost: refs.costAccum.current, elapsedMs: Date.now() - refs.turnStart.current,
        }),
        { status: result.status === "budget_exceeded" ? "budget_exceeded" : "completed", finishedAt: Date.now() },
      );
    } catch (err) {
      appendTurnPart(nextId("error"), makeErrorPart({ message: err instanceof Error ? err.message : String(err), code: "QUERY_ERROR" }));
      completeTurn(
        nextId("summary"),
        makeTurnSummaryPart({
          iterations: 0, toolCalls: refs.turnToolCount.current,
          cost: refs.costAccum.current, elapsedMs: Date.now() - refs.turnStart.current,
        }),
        { status: "error", finishedAt: Date.now() },
      );
    } finally {
      setRunning(false);
      clearSubagents();
      setSpinnerMessage(undefined);
      refs.textBuffer.current = "";
    }
  }, [onQuery, onClear, exit, clearSubagents, appendStandalonePart, startTurn, appendTurnPart, completeTurn, addToast, flushThinking, flushGroup, nextId, refs, setStatus, setSpinnerMessage, currentApprovalMode]);

  const handleCycleApprovalMode = useCallback(() => {
    if (running || pendingApproval || showCommandPalette) {
      return;
    }
    const nextMode = cycleApprovalMode(currentApprovalMode);
    setCurrentApprovalMode(nextMode);
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
      else if (running) { appendStandalonePart(nextId("cancel"), makeInfoPart("status", ["Cancelled."])); setRunning(false); }
      else exit();
    }
    if (key.ctrl && input === "k" && !running) {
      setShowCommandPalette((v) => !v);
    }
  });

  const dismissToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const runCommand = useCallback((command: string) => {
    void handleSubmit(command);
  }, [handleSubmit]);

  const showPrompt = !running && !pendingApproval && !showCommandPalette;

  const commands: Command[] = [
    { name: "Clear context", description: "Reset conversation and session state", shortcut: "/clear", action: () => { onClear(); clearSubagents(); addToast("Context cleared", "success"); } },
    { name: "Continue", description: "Continue the current session after a pause or budget limit", shortcut: "/continue", action: () => runCommand("/continue") },
    { name: "Session list", description: "Show recent sessions", shortcut: "/sessions", action: () => runCommand("/sessions") },
    { name: "Help", description: "Show available commands", shortcut: "/help", action: () => runCommand("/help") },
    { name: "Review local changes", description: "Insert or run the embedded /review command anywhere in a prompt", shortcut: "/review", action: () => runCommand("/review") },
    { name: "Simplify local changes", description: "Insert or run the embedded /simplify command anywhere in a prompt", shortcut: "/simplify", action: () => runCommand("/simplify") },
    { name: "Resume session", description: "List sessions to resume", shortcut: "/resume", action: () => runCommand("/resume") },
    { name: "Safety mode", description: "Toggle default and autopilot", shortcut: "Shift+Tab", action: handleCycleApprovalMode },
    { name: "Exit", description: "Quit devagent", shortcut: "Ctrl+C", action: () => exit() },
  ];

  let hasActiveSubagents = false;
  for (const a of subagents.values()) { if (a.status === "running") { hasActiveSubagents = true; break; } }

  return (
    <>
      <TranscriptView
        showWelcome={showWelcome}
        transcriptNodes={transcriptNodes}
        model={model}
        version={version}
        keepLatestNodeLive={showPrompt}
      />
      {hasActiveSubagents && <SubagentPanel agents={subagents} />}
      {pendingApproval && <ApprovalDialog request={pendingApproval} onResponse={handleApproval} />}
      {showCommandPalette && <CommandPalette commands={commands} onClose={() => setShowCommandPalette(false)} />}
      <ToastContainer toasts={toasts} onDismiss={dismissToast} />

      {running && !pendingApproval && (
        <Box borderStyle="round" borderColor={getApprovalModeColor(currentApprovalMode)} paddingLeft={1} paddingRight={1}>
          <Spinner active message={spinnerMessage} suffix={status.cost > 0 ? `$${status.cost.toFixed(4)}` : ""} />
        </Box>
      )}
      {showPrompt && (
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
