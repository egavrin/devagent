/**
 * @devagent/engine — Task loop, agents, orchestration.
 */

export { TaskLoop } from "./task-loop.js";
export type {
  TaskMode,
  TaskCompletionStatus,
  TaskLoopOptions,
  TaskLoopResult,
} from "./task-loop.js";

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

export { createPlanTool, getCurrentPlan, clearPlan } from "./plan-tool.js";
export type { PlanStep, Plan } from "./plan-tool.js";

export { DoubleCheck, DEFAULT_DOUBLE_CHECK_OPTIONS } from "./double-check.js";
export type {
  DoubleCheckOptions,
  DoubleCheckResult,
  DiagnosticProvider,
  TestRunner,
} from "./double-check.js";

// Memory tools (LLM-callable cross-session memory)
export { createMemoryTools } from "./memory-tools.js";

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

// Built-in plugins
export {
  createCommitPlugin,
  createReviewPlugin,
  createFeatureDevPlugin,
  createBuiltinPlugins,
  getFeaturePhases,
} from "./plugins/index.js";
export type { FeaturePhase } from "./plugins/index.js";
