export type ValidationSuite = "smoke" | "full";
export type ValidationSafetyMode = "default" | "autopilot";
export type TargetRepoId =
  | "arkcompiler_ets_frontend"
  | "arkcompiler_runtime_core"
  | "arkcompiler_runtime_core_docs";
export type ValidationSurface = "execute" | "cli";
export type ValidationTaskShape = "readonly" | "review" | "implement" | "repair";
export type IsolationMode = "temp-copy" | "worktree";
export type FailureClass = "setup" | "provider" | "runtime" | "verification" | "assertion";

export interface ToolCallRequirement {
  readonly tool: string;
  readonly minCalls: number;
}

export interface ToolBatchRequirement {
  readonly tool: string;
  readonly minBatches: number;
  readonly minBatchSize: number;
}

export interface ScenarioWriteFileStep {
  readonly kind: "write-file";
  readonly path: string;
  readonly content?: string;
  readonly templateFile?: string;
  readonly executable?: boolean;
}

export interface ScenarioRunCommandStep {
  readonly kind: "run-command";
  readonly command: string;
  readonly cwd?: "repo" | "linter";
}

export type ScenarioPreSetupStep =
  | ScenarioWriteFileStep
  | ScenarioRunCommandStep;

export interface ExecuteScenarioInvocation {
  readonly type: "execute";
  readonly taskType: "triage" | "plan" | "implement" | "review" | "repair";
  readonly workItemTitle: string;
  readonly summary: string;
  readonly issueBody?: string;
  readonly extraInstructions?: ReadonlyArray<string>;
  readonly maxIterations?: number;
  readonly reasoning?: "low" | "medium" | "high";
}

export interface CliScenarioInvocation {
  readonly type: "cli";
  readonly query: string;
  readonly maxIterations?: number;
  readonly safetyMode?: ValidationSafetyMode;
  readonly reasoning?: "low" | "medium" | "high";
  readonly extraArgs?: ReadonlyArray<string>;
}

export interface CliCommandScenarioInvocation {
  readonly type: "cli-command";
  readonly args: ReadonlyArray<string>;
}

export type ValidationScenarioInvocation =
  | ExecuteScenarioInvocation
  | CliScenarioInvocation
  | CliCommandScenarioInvocation;

export interface ContainsAssertion {
  readonly type: "contains";
  readonly source: "stdout" | "stderr" | "repoDiff" | "repoStatus" | "events" | "artifact";
  readonly path?: string;
  readonly value: string;
}

export interface MatchesAssertion {
  readonly type: "matches";
  readonly source: "stdout" | "stderr" | "repoDiff" | "repoStatus" | "events" | "artifact";
  readonly path?: string;
  readonly pattern: string;
}

export type ValidationAssertion =
  | ContainsAssertion
  | MatchesAssertion;

export interface VerificationCommand {
  readonly command: string;
  readonly cwd?: "repo" | "linter";
}

export interface ValidationScenario {
  readonly id: string;
  readonly description: string;
  readonly suites: ReadonlyArray<ValidationSuite>;
  readonly targetRepo: TargetRepoId;
  readonly surface: ValidationSurface;
  readonly taskShape: ValidationTaskShape;
  readonly isolationMode: IsolationMode;
  readonly preSetup?: ReadonlyArray<ScenarioPreSetupStep>;
  readonly invocation: ValidationScenarioInvocation;
  readonly expectedArtifacts: ReadonlyArray<string>;
  readonly assertions: ReadonlyArray<ValidationAssertion>;
  readonly verificationCommands: ReadonlyArray<VerificationCommand>;
  readonly cleanupPolicy: "destroy";
  readonly variables?: Readonly<Record<string, string>>;
  readonly commandEnv?: Readonly<Record<string, string>>;
  readonly baselineAfterSetup?: boolean;
  readonly requiresAuth?: boolean;
  readonly requiredProvider?: string;
  readonly requiresArktsLinter?: boolean;
  readonly timeoutMs?: number;
  readonly requiredToolCalls?: ReadonlyArray<ToolCallRequirement>;
  readonly requiredToolBatches?: ReadonlyArray<ToolBatchRequirement>;
  readonly expectedExitCode?: number;
}

export interface AssertionResult {
  readonly type: ValidationAssertion["type"];
  readonly source: string;
  readonly passed: boolean;
  readonly message: string;
}

export interface ToolCallAssertionResult {
  readonly tool: string;
  readonly minCalls: number;
  readonly observedCalls: number;
  readonly passed: boolean;
  readonly message: string;
}

export interface ToolBatchAssertionResult {
  readonly tool: string;
  readonly minBatches: number;
  readonly minBatchSize: number;
  readonly observedBatches: number;
  readonly observedMaxBatchSize: number;
  readonly passed: boolean;
  readonly message: string;
}

export interface ObservedToolBatch {
  readonly batchCount: number;
  readonly maxBatchSize: number;
}

export interface ArtifactValidationCheck {
  readonly name: string;
  readonly passed: boolean;
  readonly message: string;
}

export interface VerificationCommandResult {
  readonly command: string;
  readonly cwd: string;
  readonly exitCode: number;
  readonly passed: boolean;
  readonly stdout: string;
  readonly stderr: string;
}

export interface ValidationScenarioReport {
  readonly scenarioId: string;
  readonly description: string;
  readonly targetRepo: TargetRepoId;
  readonly surface: ValidationSurface;
  readonly taskShape: ValidationTaskShape;
  readonly provider: string;
  readonly model: string;
  readonly status: "passed" | "failed";
  readonly failureClass?: FailureClass;
  readonly failureMessage?: string;
  readonly startedAt: string;
  readonly finishedAt: string;
  readonly durationMs: number;
  readonly sourceRepoPath: string;
  readonly isolationPath: string;
  readonly outputDir: string;
  readonly command: {
    readonly executable: string;
    readonly args: ReadonlyArray<string>;
    readonly exitCode: number;
  };
  readonly artifactValidation: {
    readonly passed: boolean;
    readonly checks: ReadonlyArray<ArtifactValidationCheck>;
  };
  readonly assertionResults: ReadonlyArray<AssertionResult>;
  readonly toolCallAssertionResults: ReadonlyArray<ToolCallAssertionResult>;
  readonly toolBatchAssertionResults: ReadonlyArray<ToolBatchAssertionResult>;
  readonly observedToolCalls?: Readonly<Record<string, number>>;
  readonly observedToolBatches?: Readonly<Record<string, ObservedToolBatch>>;
  readonly eventsSourcePath?: string;
  readonly verificationResults: ReadonlyArray<VerificationCommandResult>;
  readonly timing: {
    readonly durationMs: number;
  };
  readonly cost: {
    readonly inputTokens?: number;
    readonly outputTokens?: number;
    readonly totalCost?: number;
  };
  readonly rawOutputs: {
    readonly stdoutPath?: string;
    readonly stderrPath?: string;
    readonly repoDiffPath?: string;
    readonly repoStatusPath?: string;
    readonly eventsPath?: string;
  };
}

export interface AggregateValidationSummary {
  readonly provider: string;
  readonly model: string;
  readonly suite: ValidationSuite | "scenario";
  readonly total: number;
  readonly passed: number;
  readonly failed: number;
  readonly reports: ReadonlyArray<ValidationScenarioReport>;
}

export interface AuthStatusSummary {
  readonly configuredProviders: ReadonlyArray<string>;
  readonly expiredProviders?: ReadonlyArray<string>;
}

export interface IsolationWorkspace {
  readonly mode: IsolationMode;
  readonly path: string;
  readonly sourceRoot: string;
}

export interface RunValidationScenarioOptions {
  readonly devagentRoot: string;
  readonly sourceRepoRoots?: Partial<Record<TargetRepoId, string>>;
  readonly provider: string;
  readonly model: string;
  readonly outputRoot: string;
  readonly command?: {
    readonly executable: string;
    readonly baseArgs: ReadonlyArray<string>;
  };
  readonly authStatusOverride?: AuthStatusSummary;
}
