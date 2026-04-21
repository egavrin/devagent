import {
  EventLogger,
  MessageRole,
  SessionState,
  SessionStore,
} from "@devagent/runtime";
import { basename } from "node:path";

import { createCrashSessionReporter } from "./cli-crash-session.js";
import { dim, yellow } from "./format.js";
import type {
  CrashSessionReporter,
  SetupSessionPersistenceArgs,
  SessionPersistenceResult,
  Verbosity,
} from "./main-types.js";
import { deriveSessionTitle, formatResumeCandidate } from "./session-preview.js";
import type {
  DevAgentConfig,
  EventBus,
  Session,
  SessionStateJSON,
  SessionStatePersistence,
  TurnBriefing,
} from "@devagent/runtime";

interface ResumeState {
  readonly initialMessages: import("@devagent/runtime").Message[] | undefined;
  readonly resumeBriefing: TurnBriefing | undefined;
  readonly prevSession: Session | null;
  readonly resumeTargetMissing: boolean;
}

interface SessionLifecycle {
  readonly activateSession: (query?: string) => Session;
  readonly deactivateSession: (reason?: "completed" | "cancelled" | "error" | "budget_exceeded") => void;
  readonly hasActiveSession: () => boolean;
  readonly getActiveSession: () => Session | null;
  readonly getActiveSessionStartTime: () => number | null;
  readonly printActiveSessionId: () => void;
}

export function resolveResumeTarget(sessionStore: SessionStore, resumeIdOrPrefix: string): Session | null {
  const exact = sessionStore.getSession(resumeIdOrPrefix);
  if (exact) return exact;
  if (resumeIdOrPrefix.length < 8) {
    throw new Error(`Session prefix "${resumeIdOrPrefix}" is too short. Use at least 8 characters or the full session ID.`);
  }
  const matches = collectResumePrefixMatches(sessionStore, resumeIdOrPrefix);
  if (matches.length === 1) return matches[0]!;
  if (matches.length > 1) {
    throw new Error(
      `Ambiguous session prefix "${resumeIdOrPrefix}". Matching sessions:\n${matches.map((session) => `- ${formatResumeCandidate(session)}`).join("\n")}`,
    );
  }
  throw new Error(`No session found for "${resumeIdOrPrefix}".`);
}

function collectResumePrefixMatches(sessionStore: SessionStore, prefix: string): Session[] {
  const matches: Session[] = [];
  for (let offset = 0; ; offset += 100) {
    const batch = sessionStore.listSessions(100, offset);
    if (batch.length === 0) break;
    matches.push(...batch.filter((session) => session.id.startsWith(prefix)));
    if (matches.length > 1) break;
  }
  return matches;
}

function resolveResumeState(sessionStore: SessionStore, cliArgs: SetupSessionPersistenceArgs[1]): ResumeState {
  if (!cliArgs.resume && !cliArgs.continue_) {
    return { initialMessages: undefined, resumeBriefing: undefined, prevSession: null, resumeTargetMissing: false };
  }
  const prevSession = cliArgs.resume
    ? resolveResumeTarget(sessionStore, cliArgs.resume)
    : sessionStore.listSessions(1)[0] ?? null;
  if (prevSession) {
    if (cliArgs.verbosity !== "quiet") {
      process.stderr.write(dim(`[session] Resuming ${prevSession.id} (${prevSession.messages.length} messages)`) + "\n");
    }
    return { initialMessages: [...prevSession.messages], resumeBriefing: undefined, prevSession, resumeTargetMissing: false };
  }
  if (cliArgs.continue_) {
    process.stderr.write(yellow(`[session] No session found: ${cliArgs.resume ?? "most recent"}`) + "\n");
    return { initialMessages: undefined, resumeBriefing: undefined, prevSession: null, resumeTargetMissing: true };
  }
  return { initialMessages: undefined, resumeBriefing: undefined, prevSession: null, resumeTargetMissing: false };
}

function createSessionStatePersistence(sessionStore: SessionStore): SessionStatePersistence {
  return {
    save: (id, state) => sessionStore.saveSessionState(id, state),
    load: (id) => sessionStore.loadSessionState(id) as SessionStateJSON | null,
  };
}

function restoreEffectiveSessionState(options: {
  readonly cliArgs: SetupSessionPersistenceArgs[1];
  readonly config: DevAgentConfig;
  readonly persistence: SessionStatePersistence;
  readonly prevSession: Session | null;
  readonly sessionState: SessionState;
}): SessionState {
  if ((!options.cliArgs.resume && !options.cliArgs.continue_) || !options.prevSession) return options.sessionState;
  const prevData = options.persistence.load(options.prevSession.id);
  if (!prevData) return options.sessionState;
  if (options.cliArgs.verbosity !== "quiet") {
    process.stderr.write(dim(`[session-state] Restored from prior session (${prevData.plan?.length ?? 0} plan steps, ${prevData.modifiedFiles?.length ?? 0} files)`) + "\n");
  }
  return SessionState.fromJSON(prevData, options.config.sessionState);
}

function createSessionLifecycle(options: {
  readonly bus: EventBus;
  readonly cliArgs: SetupSessionPersistenceArgs[1];
  readonly config: DevAgentConfig;
  readonly createCrashReporter: (sessionId: string, verbosity: Verbosity) => CrashSessionReporter;
  readonly projectRoot: string;
  readonly sessionState: SessionState;
  readonly sessionStatePersistence: SessionStatePersistence;
  readonly sessionStore: SessionStore;
}): SessionLifecycle {
  let activeSession: Session | null = null;
  let activeSessionStartTime: number | null = null;
  let eventLogger: EventLogger | null = null;
  let crashSessionReporter: CrashSessionReporter | null = null;
  let logsRotated = false;
  const activateSession = (query?: string): Session => {
    if (activeSession) return activeSession;
    activeSession = createSessionRecord(options, query);
    options.sessionState.bind(activeSession.id, options.sessionStatePersistence);
    if (options.config.logging?.enabled !== false) {
      logsRotated = rotateLogsOnce(options.config, options.cliArgs, logsRotated);
      eventLogger = new EventLogger(activeSession.id, options.config.logging?.logDir);
      eventLogger.attach(options.bus);
    }
    crashSessionReporter = options.createCrashReporter(activeSession.id, options.cliArgs.verbosity);
    activeSessionStartTime = Date.now();
    options.bus.emit("session:start", { sessionId: activeSession.id });
    return activeSession;
  };
  const deactivateSession = (reason: "completed" | "cancelled" | "error" | "budget_exceeded" = "completed"): void => {
    if (!activeSession) return;
    options.bus.emit("session:end", { sessionId: activeSession.id, reason });
    crashSessionReporter?.dispose();
    eventLogger?.close();
    crashSessionReporter = null; eventLogger = null; activeSession = null; activeSessionStartTime = null;
  };
  return {
    activateSession,
    deactivateSession,
    hasActiveSession: () => activeSession !== null,
    getActiveSession: () => activeSession,
    getActiveSessionStartTime: () => activeSessionStartTime,
    printActiveSessionId: () => { crashSessionReporter?.printSessionId(); },
  };
}

function createSessionRecord(options: {
  readonly cliArgs: SetupSessionPersistenceArgs[1];
  readonly config: DevAgentConfig;
  readonly projectRoot: string;
  readonly sessionStore: SessionStore;
}, query?: string): Session {
  const initialQuery = query ?? options.cliArgs.query ?? "(interactive query)";
  return options.sessionStore.createSession({
    query: initialQuery,
    title: deriveSessionTitle(initialQuery),
    repoLabel: basename(options.projectRoot) || options.projectRoot || "unknown repo",
    repoRoot: options.projectRoot,
    provider: options.config.provider,
    model: options.config.model,
    mode: "act",
  });
}

function rotateLogsOnce(config: DevAgentConfig, cliArgs: SetupSessionPersistenceArgs[1], logsRotated: boolean): boolean {
  if (logsRotated) return true;
  try {
    const deleted = EventLogger.rotate(config.logging?.retentionDays ?? 30, config.logging?.logDir);
    if (deleted > 0 && cliArgs.verbosity === "verbose") process.stderr.write(dim(`[logging] Rotated ${deleted} old log file(s)`) + "\n");
  } catch {
    // Non-fatal: documented exception to fail-fast.
  }
  return true;
}

function registerSessionPersistenceEvents(bus: EventBus, sessionStore: SessionStore, getActiveSession: () => Session | null): void {
  bus.on("message:user", (event) => {
    const session = event.agentId ? null : getActiveSession();
    if (session) sessionStore.addMessage(session.id, { role: MessageRole.USER, content: event.content });
  });
  bus.on("message:assistant", (event) => {
    const session = event.agentId || event.partial ? null : getActiveSession();
    if (session) sessionStore.addMessage(session.id, { role: MessageRole.ASSISTANT, content: event.content, toolCalls: event.toolCalls });
  });
  bus.on("message:tool", (event) => {
    const session = event.agentId ? null : getActiveSession();
    if (session) sessionStore.addMessage(session.id, { role: MessageRole.TOOL, content: event.content, toolCallId: event.toolCallId });
  });
  bus.on("cost:update", (event) => {
    const session = getActiveSession();
    if (session) sessionStore.addCostRecord(session.id, { inputTokens: event.inputTokens, outputTokens: event.outputTokens, cacheReadTokens: 0, cacheWriteTokens: 0, totalCost: event.totalCost });
  });
  bus.on("context:compacted", (event) => {
    const session = getActiveSession();
    if (session) sessionStore.saveCompactionEvent(session.id, { tokensBefore: event.tokensBefore, tokensAfter: event.estimatedTokens, removedCount: event.removedCount });
  });
}

export async function setupSessionPersistence(...args: SetupSessionPersistenceArgs): Promise<SessionPersistenceResult> {
  const [config, cliArgs, projectRoot, _provider, bus, sessionState, options = {}] = args;
  void _provider;
  const sessionStore = options.sessionStore ?? new SessionStore();
  const createCrashReporter = options.createCrashReporter
    ?? ((sessionId: string, verbosity: Verbosity) => createCrashSessionReporter(sessionId, verbosity));
  const resume = resolveResumeState(sessionStore, cliArgs);
  const sessionStatePersistence = createSessionStatePersistence(sessionStore);
  const effectiveSessionState = restoreEffectiveSessionState({
    cliArgs, config, persistence: sessionStatePersistence, prevSession: resume.prevSession, sessionState,
  });
  const lifecycle = createSessionLifecycle({
    bus, cliArgs, config, createCrashReporter, projectRoot,
    sessionState: effectiveSessionState, sessionStatePersistence, sessionStore,
  });
  registerSessionPersistenceEvents(bus, sessionStore, lifecycle.getActiveSession);

  return {
    sessionStore,
    initialMessages: resume.initialMessages,
    resumeBriefing: resume.resumeBriefing,
    resumeTargetMissing: resume.resumeTargetMissing,
    sessionState: effectiveSessionState,
    ...lifecycle,
    close: () => {
      lifecycle.deactivateSession();
      sessionStore.close();
    },
  };
}
