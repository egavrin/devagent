import type { LSPRouter } from "./double-check-wiring.js";
import type { OutputState } from "./output-state.js";
import type { StatusLine } from "./status-line.js";
import type { createDefaultRegistry } from "@devagent/providers";
import type {
  ApprovalGate,
  ContextManager,
  DoubleCheck,
  EventBus,
  LLMProvider,
  Message,
  Session,
  SessionState,
  SkillRegistry,
  SkillResolver,
  TaskLoopOptions,
  TaskLoopResult,
  TaskMode,
  ToolRegistry,
  TurnBriefing,
  VerbosityConfig,
} from "@devagent/runtime";

export type Verbosity = "quiet" | "normal" | "verbose";

export interface RunOptions {
  readonly provider: LLMProvider;
  readonly toolRegistry: ToolRegistry;
  readonly bus: EventBus;
  readonly gate: ApprovalGate;
  readonly config: import("@devagent/runtime").DevAgentConfig;
  readonly repoRoot: string;
  readonly mode: TaskMode;
  readonly skills: SkillRegistry;
  readonly contextManager: ContextManager;
  readonly doubleCheck: DoubleCheck;
  readonly initialMessages: Message[] | undefined;
  readonly verbosity: Verbosity;
  readonly verbosityConfig: VerbosityConfig;
  readonly sessionState: SessionState;
  readonly briefing?: TurnBriefing;
}

export interface RunSingleQueryOptions extends RunOptions {
  readonly query: string;
}

export interface InteractiveSystemPromptOptions {
  readonly repoRoot: string;
  readonly skills: SkillRegistry;
  readonly toolRegistry: ToolRegistry;
  readonly config: import("@devagent/runtime").DevAgentConfig;
  readonly safetyMode: import("@devagent/runtime").SafetyMode;
  readonly mode: TaskMode;
  readonly briefing?: TurnBriefing;
}

export interface InteractiveResumeSeed {
  initialMessages: Message[] | undefined;
  briefing: TurnBriefing | undefined;
}

export interface ReviewArgs {
  patchFile: string;
  ruleFile: string | null;
  jsonOutput: boolean;
  help: boolean;
}

export interface CliSubcommand {
  name: "help" | "doctor" | "config" | "configure" | "setup" | "init" | "update" | "completions" | "install-lsp";
  args: string[];
}

export interface CliArgs {
  query: string | null;
  file: string | null;
  safetyMode: import("@devagent/runtime").SafetyMode | null;
  modeParseError: string | null;
  usageError: string | null;
  provider: string | null;
  model: string | null;
  maxIterations: number | null;
  reasoning: "low" | "medium" | "high" | null;
  verbosity: Verbosity;
  verboseCategories: string | undefined;
  authCommand: { subcommand: string; args: string[] } | null;
  sessionsCommand: boolean;
  resume: string | null;
  continue_: boolean;
  review: ReviewArgs | null;
  subcommand: CliSubcommand | null;
}

export interface CrashSessionReporter {
  printSessionId: () => void;
  dispose: () => void;
}

export interface CrashSessionReporterProcess {
  stderr: {
    write: (chunk: string) => boolean;
    destroyed?: boolean;
    writableEnded?: boolean;
    writableFinished?: boolean;
  };
  once: (event: "SIGINT" | "uncaughtException" | "unhandledRejection", listener: (...args: any[]) => void) => void;
  off: (event: "SIGINT" | "uncaughtException" | "unhandledRejection", listener: (...args: any[]) => void) => void;
  exit: (code?: number) => never;
}

export interface ConfigSetupResult {
  readonly config: import("@devagent/runtime").DevAgentConfig;
  readonly projectRoot: string;
}

export interface ProviderSetupResult {
  readonly provider: LLMProvider;
  readonly providerRegistry: ReturnType<typeof createDefaultRegistry>;
}

export interface ToolsSetupResult {
  readonly toolRegistry: ToolRegistry;
  readonly bus: EventBus;
  readonly gate: ApprovalGate;
  readonly verbosityConfig: VerbosityConfig;
  readonly lspToolCounts: Map<string, number>;
  readonly statusLine: StatusLine | null;
  readonly trackInternalLSPDiagnostics: () => void;
  sessionState: SessionState;
  readonly skills: SkillRegistry;
  readonly skillResolver: SkillResolver;
  readonly doubleCheck: DoubleCheck;
  readonly contextManager: ContextManager;
  readonly delegateAmbientContext: {
    approvalMode: string;
  };
}

export interface LSPSetupResult {
  lspRouter: LSPRouter | null;
  hasLSPDiagnostics: boolean;
}

export interface SessionPersistenceResult {
  readonly sessionStore: import("@devagent/runtime").SessionStore;
  readonly initialMessages: Message[] | undefined;
  readonly resumeBriefing: TurnBriefing | undefined;
  readonly resumeTargetMissing: boolean;
  sessionState: SessionState;
  readonly activateSession: (query?: string) => Session;
  readonly deactivateSession: (reason?: "completed" | "cancelled" | "error" | "budget_exceeded") => void;
  readonly hasActiveSession: () => boolean;
  readonly getActiveSession: () => Session | null;
  readonly getActiveSessionStartTime: () => number | null;
  readonly printActiveSessionId: () => void;
  readonly close: () => void;
}

export interface SessionPersistenceSetupOptions {
  readonly sessionStore?: import("@devagent/runtime").SessionStore;
  readonly createCrashReporter?: (sessionId: string, verbosity: Verbosity) => CrashSessionReporter;
}

export type SetupSessionPersistenceArgs = [
  config: import("@devagent/runtime").DevAgentConfig,
  cliArgs: CliArgs,
  projectRoot: string,
  provider: LLMProvider,
  bus: EventBus,
  sessionState: SessionState,
  options?: SessionPersistenceSetupOptions,
];

export type InitialTuiLoopOptions = Pick<
  TaskLoopOptions,
  "systemPrompt" | "initialMessages" | "injectSessionStateOnFirstTurn"
>;

export type DelegatedWorkSummary = ReturnType<OutputState["buildDelegatedWorkSummary"]>;
export type TaskLoopRunResult = TaskLoopResult;
