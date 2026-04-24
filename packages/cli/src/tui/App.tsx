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
const EMPTY_RESPONSE_NOTICE = "Model returned no final response. Type /continue to retry, or switch provider/model if it repeats.";
const ABORTED_NOTICE = "Run stopped before completion. Type /continue to retry from the current session.";

// ─── Types ──────────────────────────────────────────────────

export interface AppProps {
  readonly bus: EventBus;
  readonly onQuery: (query: string) => Promise<InteractiveQueryResult>;
  readonly onCancelQuery?: () => void;
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

type BuiltinCommandResult = "handled" | "run-continue" | "not-builtin";
type BuiltinCommandAction = "clear" | "exit" | "help" | "resume" | "run-continue" | "sessions";

interface BuiltinCommandContext {
  readonly exit: () => void;
  readonly clearSession: () => void;
  readonly onClear: () => void;
  readonly clearSubagents: () => void;
  readonly onListSessions: (() => ReadonlyArray<SessionPreview>) | undefined;
  readonly appendStandalonePart: (id: string, part: ReturnType<typeof makeInfoPart>) => void;
  readonly nextId: (prefix: string) => string;
  readonly setStatus: React.Dispatch<React.SetStateAction<ReturnType<typeof useAgentLog>["status"]>>;
}

interface QueryRunContext {
  readonly onQuery: (query: string) => Promise<InteractiveQueryResult>;
  readonly refs: ReturnType<typeof useAgentLog>["refs"];
  readonly startTurn: ReturnType<typeof useAgentLog>["startTurn"];
  readonly appendTurnPart: ReturnType<typeof useAgentLog>["appendTurnPart"];
  readonly completeTurn: ReturnType<typeof useAgentLog>["completeTurn"];
  readonly flushThinking: ReturnType<typeof useAgentLog>["flushThinking"];
  readonly flushGroup: ReturnType<typeof useAgentLog>["flushGroup"];
  readonly nextId: (prefix: string) => string;
  readonly addToast: (message: string, variant?: "info" | "success" | "warning" | "error") => void;
  readonly clearSubagents: () => void;
  readonly setShowWelcome: React.Dispatch<React.SetStateAction<boolean>>;
  readonly setQueryHistory: React.Dispatch<React.SetStateAction<string[]>>;
  readonly setRunning: React.Dispatch<React.SetStateAction<boolean>>;
  readonly setCancelPending: React.Dispatch<React.SetStateAction<boolean>>;
  readonly setSpinnerMessage: React.Dispatch<React.SetStateAction<string | undefined>>;
}

const BUILTIN_COMMANDS: Readonly<Record<string, BuiltinCommandAction>> = {
  "/?": "help",
  "/c": "clear",
  "/clear": "clear",
  "/continue": "run-continue",
  "/exit": "exit",
  "/h": "help",
  "/help": "help",
  "/q": "exit",
  "/quit": "exit",
  "/r": "resume",
  "/resume": "resume",
  "/s": "sessions",
  "/sessions": "sessions",
};

const BUILTIN_COMMAND_HANDLERS: Record<BuiltinCommandAction, (context: BuiltinCommandContext) => BuiltinCommandResult> = {
  clear: (context) => {
    context.clearSession();
    return "handled";
  },
  exit: (context) => {
    context.exit();
    return "handled";
  },
  help: (context) => {
    context.appendStandalonePart(context.nextId("help"), makeInfoPart("info", [TUI_HELP_MESSAGE]));
    return "handled";
  },
  resume: (context) => {
    appendResumeList(context);
    return "handled";
  },
  "run-continue": () => "run-continue",
  sessions: (context) => {
    appendSessionList(context);
    return "handled";
  },
};

function handleBuiltinCommand(command: string, context: BuiltinCommandContext): BuiltinCommandResult {
  const action = BUILTIN_COMMANDS[command];
  return action ? BUILTIN_COMMAND_HANDLERS[action](context) : "not-builtin";
}

interface ClearSessionContext {
  readonly onClear: () => void;
  readonly clearSubagents: () => void;
  readonly setStatus: React.Dispatch<React.SetStateAction<ReturnType<typeof useAgentLog>["status"]>>;
  readonly appendStandalonePart: (id: string, part: ReturnType<typeof makeInfoPart>) => void;
  readonly nextId: (prefix: string) => string;
}

function clearInteractiveSession(context: ClearSessionContext): void {
  context.onClear();
  context.clearSubagents();
  context.setStatus((status) => ({ ...status, cost: 0, iteration: 0, inputTokens: 0 }));
  context.appendStandalonePart(context.nextId("clear"), makeInfoPart("info", ["Context cleared."]));
}

function appendSessionList(context: BuiltinCommandContext): void {
  const lines = context.onListSessions
    ? renderSessionsCommandOutput(context.onListSessions()).split("\n")
    : ["Session listing not available."];
  context.appendStandalonePart(context.nextId("sessions"), makeInfoPart("sessions", lines));
}

function appendResumeList(context: BuiltinCommandContext): void {
  if (!context.onListSessions) {
    return;
  }
  context.appendStandalonePart(
    context.nextId("resume"),
    makeInfoPart("sessions", renderResumeCommandOutput(context.onListSessions()).split("\n")),
  );
}

async function runQueryTurn(query: string, context: QueryRunContext): Promise<void> {
  prepareQueryTurn(query, context);
  try {
    await completeSuccessfulQueryTurn(query, context);
  } catch (err) {
    completeFailedQueryTurn(err, context);
  } finally {
    finishQueryTurn(context);
  }
}

function prepareQueryTurn(query: string, context: QueryRunContext): void {
  context.setShowWelcome(false);
  context.setQueryHistory((prev) => [...prev, query]);
  context.setRunning(true);
  context.refs.textBuffer.current = "";
  context.refs.thinkingStart.current = null;
  context.refs.turnStart.current = Date.now();
  context.refs.turnToolCount.current = 0;
  context.refs.costAccum.current = 0;
  context.startTurn(context.nextId("turn"), query, context.refs.turnStart.current);
}

async function completeSuccessfulQueryTurn(query: string, context: QueryRunContext): Promise<void> {
  const result = await context.onQuery(query);
  context.flushThinking();
  context.flushGroup();
  appendQueryResult(result, context);
  context.completeTurn(
    context.nextId("summary"),
    makeTurnSummaryPart({
      iterations: result.iterations,
      toolCalls: result.toolCalls,
      cost: context.refs.costAccum.current,
      elapsedMs: Date.now() - context.refs.turnStart.current,
    }),
    { status: turnStatusForQueryStatus(result.status), finishedAt: Date.now() },
  );
}

function appendQueryResult(result: InteractiveQueryResult, context: QueryRunContext): void {
  if (result.lastText) {
    context.appendTurnPart(context.nextId("final"), makeFinalOutputPart(result.lastText));
  }
  const notice = noticeForQueryStatus(result.status);
  if (notice) {
    context.appendTurnPart(context.nextId("status"), makeInfoPart("status", [notice]));
    context.addToast(notice, "warning");
  }
}

export function turnStatusForQueryStatus(status: InteractiveQueryResult["status"]): "completed" | "budget_exceeded" | "error" {
  if (status === "success") return "completed";
  if (status === "budget_exceeded") return "budget_exceeded";
  return "error";
}

export function noticeForQueryStatus(status: InteractiveQueryResult["status"]): string | null {
  if (status === "budget_exceeded") return ITERATION_LIMIT_NOTICE;
  if (status === "empty_response") return EMPTY_RESPONSE_NOTICE;
  if (status === "aborted") return ABORTED_NOTICE;
  return null;
}

function completeFailedQueryTurn(err: unknown, context: QueryRunContext): void {
  context.appendTurnPart(
    context.nextId("error"),
    makeErrorPart({ message: err instanceof Error ? err.message : String(err), code: "QUERY_ERROR" }),
  );
  context.completeTurn(
    context.nextId("summary"),
    makeTurnSummaryPart({
      iterations: 0,
      toolCalls: context.refs.turnToolCount.current,
      cost: context.refs.costAccum.current,
      elapsedMs: Date.now() - context.refs.turnStart.current,
    }),
    { status: "error", finishedAt: Date.now() },
  );
}

function finishQueryTurn(context: QueryRunContext): void {
  context.setRunning(false);
  context.setCancelPending(false);
  context.clearSubagents();
  context.setSpinnerMessage(undefined);
  context.refs.textBuffer.current = "";
}

interface SubmitHandlerOptions extends BuiltinCommandContext, QueryRunContext {}

function useSubmitHandler(options: SubmitHandlerOptions): (value: string) => Promise<void> {
  return useCallback(async (value: string) => {
    const trimmed = value.trim();
    if (!trimmed) return;

    const commandResult = handleBuiltinCommand(trimmed, options);
    if (commandResult === "handled") return;
    await runQueryTurn(commandResult === "run-continue" ? "continue" : trimmed, options);
  }, [options]);
}

function useTuiKeyboardShortcuts(options: {
  readonly appendStandalonePart: ReturnType<typeof useAgentLog>["appendStandalonePart"];
  readonly cancelPending: boolean;
  readonly exit: () => void;
  readonly handleApproval: (approved: boolean, session?: boolean, reason?: string) => void;
  readonly nextId: (prefix: string) => string;
  readonly onCancelQuery: (() => void) | undefined;
  readonly pendingApproval: ApprovalRequest | null;
  readonly running: boolean;
  readonly setCancelPending: React.Dispatch<React.SetStateAction<boolean>>;
  readonly setRunning: React.Dispatch<React.SetStateAction<boolean>>;
  readonly setShowCommandPalette: React.Dispatch<React.SetStateAction<boolean>>;
  readonly setSpinnerMessage: React.Dispatch<React.SetStateAction<string | undefined>>;
  readonly showCommandPalette: boolean;
}): void {
  useInput((input, key) => {
    if ((key.ctrl && input === "c") || input === "\x03") {
      handleCancelShortcut(options);
    }
    if (key.ctrl && input === "k" && !options.running) {
      options.setShowCommandPalette((value) => !value);
    }
  });
}

export function handleCancelShortcut(options: {
  readonly appendStandalonePart: ReturnType<typeof useAgentLog>["appendStandalonePart"];
  readonly cancelPending: boolean;
  readonly exit: () => void;
  readonly handleApproval: (approved: boolean, session?: boolean, reason?: string) => void;
  readonly nextId: (prefix: string) => string;
  readonly onCancelQuery: (() => void) | undefined;
  readonly pendingApproval: ApprovalRequest | null;
  readonly running: boolean;
  readonly setCancelPending: React.Dispatch<React.SetStateAction<boolean>>;
  readonly setRunning: React.Dispatch<React.SetStateAction<boolean>>;
  readonly setShowCommandPalette: React.Dispatch<React.SetStateAction<boolean>>;
  readonly setSpinnerMessage: React.Dispatch<React.SetStateAction<string | undefined>>;
  readonly showCommandPalette: boolean;
}): void {
  if (options.showCommandPalette) {
    options.setShowCommandPalette(false);
  } else if (options.pendingApproval) {
    options.handleApproval(false);
  } else if (options.running) {
    if (options.cancelPending) return;
    options.onCancelQuery?.();
    options.setCancelPending(true);
    options.setSpinnerMessage("Cancelling...");
  } else {
    options.exit();
  }
}

function buildCommands(options: {
  readonly addToast: (message: string, variant?: "info" | "success" | "warning" | "error") => void;
  readonly clearSession: () => void;
  readonly exit: () => void;
  readonly handleCycleApprovalMode: () => void;
  readonly runCommand: (command: string) => void;
}): Command[] {
  return [
    { name: "Clear context", description: "Reset conversation and session state", shortcut: "/clear", action: () => { options.clearSession(); options.addToast("Context cleared", "success"); } },
    { name: "Continue", description: "Continue the current session after a pause or budget limit", shortcut: "/continue", action: () => options.runCommand("/continue") },
    { name: "Session list", description: "Show recent sessions", shortcut: "/sessions", action: () => options.runCommand("/sessions") },
    { name: "Help", description: "Show available commands", shortcut: "/help", action: () => options.runCommand("/help") },
    { name: "Review local changes", description: "Insert or run the embedded /review command anywhere in a prompt", shortcut: "/review", action: () => options.runCommand("/review") },
    { name: "Simplify local changes", description: "Insert or run the embedded /simplify command anywhere in a prompt", shortcut: "/simplify", action: () => options.runCommand("/simplify") },
    { name: "Resume session", description: "List sessions to resume", shortcut: "/resume", action: () => options.runCommand("/resume") },
    { name: "Safety mode", description: "Toggle default and autopilot", shortcut: "Shift+Tab", action: options.handleCycleApprovalMode },
    { name: "Exit", description: "Quit devagent", shortcut: "Ctrl+C", action: () => options.exit() },
  ];
}

function hasRunningSubagents(subagents: Map<string, { readonly status: string }>): boolean {
  for (const agent of subagents.values()) {
    if (agent.status === "running") return true;
  }
  return false;
}

interface AppViewProps {
  readonly approvalMode: string;
  readonly commands: Command[];
  readonly cwd: string | undefined;
  readonly handleApproval: (approved: boolean, session?: boolean, reason?: string) => void;
  readonly handleCycleApprovalMode: () => void;
  readonly handleSubmit: (value: string) => Promise<void>;
  readonly hasActiveSubagents: boolean;
  readonly pendingApproval: ApprovalRequest | null;
  readonly queryHistory: ReadonlyArray<string>;
  readonly running: boolean;
  readonly setShowCommandPalette: React.Dispatch<React.SetStateAction<boolean>>;
  readonly showCommandPalette: boolean;
  readonly showPrompt: boolean;
  readonly showWelcome: boolean;
  readonly spinnerMessage: string | undefined;
  readonly status: ReturnType<typeof useAgentLog>["status"];
  readonly subagents: ReturnType<typeof useAgentLog>["subagents"];
  readonly toasts: ReadonlyArray<ToastMessage>;
  readonly transcriptNodes: ReadonlyArray<TranscriptNode>;
  readonly dismissToast: (id: string) => void;
  readonly model: string;
  readonly version: string | undefined;
}

function AppView(props: AppViewProps): React.ReactElement {
  return (
    <>
      <TranscriptView
        showWelcome={props.showWelcome}
        transcriptNodes={props.transcriptNodes}
        model={props.model}
        version={props.version}
        keepLatestNodeLive={props.showPrompt}
      />
      {props.hasActiveSubagents && <SubagentPanel agents={props.subagents} />}
      {props.pendingApproval && <ApprovalDialog request={props.pendingApproval} onResponse={props.handleApproval} />}
      {props.showCommandPalette && <CommandPalette commands={props.commands} onClose={() => props.setShowCommandPalette(false)} />}
      <ToastContainer toasts={props.toasts} onDismiss={props.dismissToast} />

      {props.running && !props.pendingApproval && (
        <Box borderStyle="round" borderColor={getApprovalModeColor(props.approvalMode)} paddingLeft={1} paddingRight={1}>
          <Spinner active message={props.spinnerMessage} suffix={props.status.cost > 0 ? `$${props.status.cost.toFixed(4)}` : ""} />
        </Box>
      )}
      {props.showPrompt && (
        <PromptInput
          onSubmit={(value) => { void props.handleSubmit(value); }}
          onCycleApprovalMode={props.handleCycleApprovalMode}
          history={props.queryHistory}
          placeholder="Ask anything…"
          cwd={props.cwd}
          approvalMode={props.approvalMode}
        />
      )}
      <StatusBar {...props.status} cwd={props.cwd} running={props.running} hasApproval={!!props.pendingApproval} />
    </>
  );
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
export function App({ bus, onQuery, onCancelQuery, onClear, onCycleApprovalMode, onListSessions, model, approvalMode, cwd, version }: AppProps): React.ReactElement {
  const [showWelcome, setShowWelcome] = useState(true);
  const { exit } = useApp();
  const [queryHistory, setQueryHistory] = useState<string[]>([]);
  const [running, setRunning] = useState(false);
  const [cancelPending, setCancelPending] = useState(false);
  const [pendingApproval, setPendingApproval] = useState<ApprovalRequest | null>(null);
  const [showCommandPalette, setShowCommandPalette] = useState(false);
  const [toasts, setToasts] = useState<ToastMessage[]>([]);
  const [currentApprovalMode, setCurrentApprovalMode] = useState(approvalMode);

  const {
    transcriptNodes, status, subagents, spinnerMessage, appendStandalonePart, startTurn, appendTurnPart,
    completeTurn, flushThinking, flushGroup, nextId, setStatus, setSubagents, setSpinnerMessage, refs,
  } = useAgentLog({ bus, model, collapseFailures: true, compactPlanProgress: true, onApproval: (e) => setPendingApproval({ id: e.id, toolName: e.toolName, details: e.details }) });

  const addToast = useCallback((message: string, variant: "info" | "success" | "warning" | "error" = "info") => {
    setToasts((prev) => [...prev, { id: nextId("toast"), message, variant }]);
  }, [nextId]);

  const clearSubagents = useCallback(() => {
    setSubagents((prev) => (prev.size === 0 ? prev : new Map()));
  }, [setSubagents]);

  const clearSession = useCallback(() => {
    clearInteractiveSession({ onClear, clearSubagents, setStatus, appendStandalonePart, nextId });
  }, [appendStandalonePart, clearSubagents, nextId, onClear, setStatus]);

  const handleSubmit = useSubmitHandler({
    exit, clearSession, onClear, clearSubagents, onListSessions, appendStandalonePart, nextId, setStatus,
    onQuery, refs, startTurn, appendTurnPart, completeTurn, flushThinking, flushGroup,
    addToast, setShowWelcome, setQueryHistory, setRunning, setCancelPending, setSpinnerMessage,
  });

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

  useTuiKeyboardShortcuts({
    appendStandalonePart, cancelPending, exit, handleApproval, nextId, onCancelQuery, pendingApproval,
    running, setCancelPending, setRunning, setShowCommandPalette, setSpinnerMessage, showCommandPalette,
  });

  const dismissToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const runCommand = useCallback((command: string) => {
    void handleSubmit(command);
  }, [handleSubmit]);

  const showPrompt = !running && !pendingApproval && !showCommandPalette;

  const commands = buildCommands({ addToast, clearSession, exit, handleCycleApprovalMode, runCommand });
  const hasActiveSubagents = hasRunningSubagents(subagents);

  return <AppView {...{
    approvalMode: currentApprovalMode, commands, cwd, dismissToast, handleApproval, handleCycleApprovalMode,
    handleSubmit, hasActiveSubagents, model, pendingApproval, queryHistory, running, setShowCommandPalette,
    showCommandPalette, showPrompt, showWelcome, spinnerMessage, status, subagents, toasts, transcriptNodes, version,
  }} />;
}
