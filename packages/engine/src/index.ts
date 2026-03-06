/**
 * @devagent/engine — Task loop, agents, orchestration.
 */

export { TaskLoop, truncateToolOutput } from "./task-loop.js";
export type {
  TaskMode,
  TaskCompletionStatus,
  TaskLoopOptions,
  TaskLoopResult,
  MidpointCallback,
} from "./task-loop.js";

export { StagnationDetector } from "./stagnation-detector.js";
export type {
  StagnationToolCall,
  StagnationDetectorOptions,
} from "./stagnation-detector.js";


export { AgentRegistry, runAgent } from "./agents.js";
export type {
  AgentDefinition,
  AgentRunOptions,
  AgentRunResult,
} from "./agents.js";

export { CheckpointManager } from "./checkpoints.js";
export type {
  Checkpoint,
  CheckpointManagerOptions,
} from "./checkpoints.js";

export { createDelegateTool } from "./delegate-tool.js";
export type { DelegateToolContext } from "./delegate-tool.js";

export { createPlanTool } from "./plan-tool.js";
export type { PlanStep, Plan } from "./plan-tool.js";

// Plan quality judge
export { judgePlanQuality } from "./plan-judge.js";
export type { PlanJudgeResult } from "./plan-judge.js";

// Compaction quality judge
export { judgeCompactionQuality } from "./compaction-judge.js";
export type { CompactionJudgeResult } from "./compaction-judge.js";

// Sub-agent validation judge
export { judgeSubagentOutput } from "./subagent-judge.js";
export type { SubagentJudgeResult } from "./subagent-judge.js";

// Error recovery classification judge
export { classifyError } from "./error-judge.js";
export type { ErrorClassification } from "./error-judge.js";

// Knowledge extraction (pre-compaction domain knowledge capture)
export { extractPreCompactionKnowledge } from "./knowledge-extractor.js";
export type { KnowledgeExtractionResult, KnowledgeExtractionEntry } from "./knowledge-extractor.js";

export { SessionState, extractEnvFact, DEFAULT_SESSION_STATE_CONFIG, SESSION_STATE_MARKER, PRUNED_MARKER_PREFIX, SUPERSEDED_MARKER_PREFIX, KNOWLEDGE_CONTENT_MAX_CHARS } from "./session-state.js";
export type {
  EnvFact,
  ToolResultSummary,
  Finding,
  KnowledgeEntry,
  SessionStateJSON,
  SessionStatePersistence,
  SessionStateConfig,
} from "./session-state.js";

export { DoubleCheck, DEFAULT_DOUBLE_CHECK_OPTIONS, parseTestOutput } from "./double-check.js";
export type {
  DoubleCheckOptions,
  DoubleCheckResult,
  TestSummary,
  DiagnosticProvider,
  DiagnosticBaseline,
  TestRunner,
} from "./double-check.js";

// Turn briefing (context synthesis for turn isolation)
export { synthesizeBriefing, formatBriefing, findLastUserContent } from "./briefing.js";
export type { TurnBriefing, BriefingStrategy, SynthesizeBriefingOptions } from "./briefing.js";

// Memory tools (LLM-callable cross-session memory)
export { createMemoryTools } from "./memory-tools.js";
export type { MemoryToolOptions } from "./memory-tools.js";

// Finding tool (LLM-callable analysis persistence)
export { createFindingTool } from "./finding-tool.js";

// Tool script — batched readonly tool execution
export { ToolScriptEngine } from "./tool-script.js";
export type {
  ToolScriptStep,
  ToolScript,
  StepResult,
  ToolScriptResult,
  ToolScriptEngineOptions,
} from "./tool-script.js";
export { createToolScriptTool } from "./tool-script-tool.js";
export type { ToolScriptToolContext } from "./tool-script-tool.js";

// Review pipeline (rule-based patch review)
export { runReviewPipeline, VIOLATION_SCHEMA } from "./review/index.js";
export type {
  ReviewPipelineOptions,
  ReviewPipelineInput,
  ReviewConfig,
  ReviewResult,
  Violation,
  ReviewSummary,
  Severity,
  ReviewChangeType,
  PatchReviewData,
  ContextItem,
  ContextProvider,
} from "./review/index.js";

// Built-in plugins
export {
  createCommitPlugin,
  createReviewPlugin,
  createFeatureDevPlugin,
  createBuiltinPlugins,
} from "./plugins/index.js";
