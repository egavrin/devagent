/**
 * @devagent/engine — Task loop, agents, orchestration.
 */

export { TaskLoop } from "./task-loop.js";
export type {
  TaskMode,
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

export { DoubleCheck, DEFAULT_DOUBLE_CHECK_OPTIONS } from "./double-check.js";
export type {
  DoubleCheckOptions,
  DoubleCheckResult,
  DiagnosticProvider,
  TestRunner,
} from "./double-check.js";
