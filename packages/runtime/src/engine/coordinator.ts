/**
 * Coordinator Mode — central orchestrator that restricts the main agent
 * to readonly + delegation + messaging tools, while spawning concurrent
 * worker agents for actual implementation.
 *
 * Inspired by claude-code-src coordinatorMode.ts pattern.
 *
 * Usage:
 *   When config.coordinator.enabled is true, the TaskLoop's available tools
 *   are restricted to the coordinator's tool set (readonly, delegate, task management).
 *   Workers are spawned via the delegate tool and communicate via the EventBus.
 */

import type { ToolSpec } from "../core/index.js";
import type { ToolRegistry } from "../tools/index.js";

/** Task lifecycle states. */
type WorkerTaskStatus = "pending" | "running" | "completed" | "failed" | "killed";

interface WorkerTask {
  readonly taskId: string;
  readonly agentId: string | null;
  readonly objective: string;
  status: WorkerTaskStatus;
  readonly createdAt: number;
  completedAt: number | null;
  result: string | null;
  error: string | null;
}

// ─── Coordinator ────────────────────────────────────────────

/**
 * Manages the coordinator's worker task pool.
 * Tracks task lifecycle and enforces concurrency limits.
 */
export class CoordinatorTaskPool {
  private readonly tasks = new Map<string, WorkerTask>();
  private readonly maxWorkers: number;
  private taskCounter = 0;
  private runningCounter = 0;

  constructor(maxWorkers: number = 3) {
    this.maxWorkers = maxWorkers;
  }

  /**
   * Create a new task in the pool. Returns the task ID.
   */
  createTask(objective: string): string {
    const taskId = `task-${++this.taskCounter}`;
    this.tasks.set(taskId, {
      taskId,
      agentId: null,
      objective,
      status: "pending",
      createdAt: Date.now(),
      completedAt: null,
      result: null,
      error: null,
    });
    return taskId;
  }

  /**
   * Assign an agent to a task and mark it as running.
   */
  startTask(taskId: string, agentId: string): boolean {
    const task = this.tasks.get(taskId);
    if (!task || task.status !== "pending") return false;
    if (this.runningCounter >= this.maxWorkers) return false;

    task.status = "running";
    (task as { agentId: string | null }).agentId = agentId;
    this.runningCounter++;
    return true;
  }

  /**
   * Mark a task as completed with a result.
   */
  completeTask(taskId: string, result: string): boolean {
    const task = this.tasks.get(taskId);
    if (!task || task.status !== "running") return false;

    task.status = "completed";
    task.completedAt = Date.now();
    task.result = result;
    this.runningCounter--;
    return true;
  }

  /**
   * Mark a task as failed with an error.
   */
  failTask(taskId: string, error: string): boolean {
    const task = this.tasks.get(taskId);
    if (!task || task.status !== "running") return false;

    task.status = "failed";
    task.completedAt = Date.now();
    task.error = error;
    this.runningCounter--;
    return true;
  }

  /**
   * Kill a running or pending task.
   */
  killTask(taskId: string): boolean {
    const task = this.tasks.get(taskId);
    if (!task || (task.status !== "running" && task.status !== "pending")) return false;

    if (task.status === "running") this.runningCounter--;
    task.status = "killed";
    task.completedAt = Date.now();
    return true;
  }

  /**
   * Get a task by ID.
   */
  getTask(taskId: string): WorkerTask | undefined {
    return this.tasks.get(taskId);
  }

  /**
   * List all tasks.
   */
  listTasks(): ReadonlyArray<WorkerTask> {
    return [...this.tasks.values()];
  }

  /**
   * Count of currently running tasks.
   */
  getRunningCount(): number {
    return this.runningCounter;
  }

  /**
   * True if more workers can be started.
   */
  hasCapacity(): boolean {
    return this.getRunningCount() < this.maxWorkers;
  }
}

// ─── Tool Filtering ─────────────────────────────────────────

/** Tool categories allowed for the coordinator agent. */
const COORDINATOR_ALLOWED_CATEGORIES = new Set(["readonly", "workflow", "state"]);

/** Specific tool names blocked even within allowed categories. */
const COORDINATOR_BLOCKED_TOOLS = new Set([
  "write_file", "replace_in_file", "run_command",
]);

/**
 * Filter a ToolRegistry to only include coordinator-allowed tools.
 * The coordinator can read files, delegate to workers, and manage tasks,
 * but cannot directly mutate files or run commands.
 */
export function filterCoordinatorTools(registry: ToolRegistry): ReadonlyArray<ToolSpec> {
  return registry.getAll().filter((tool) => {
    if (COORDINATOR_BLOCKED_TOOLS.has(tool.name)) return false;
    return COORDINATOR_ALLOWED_CATEGORIES.has(tool.category);
  });
}

// ─── Task Management Tools ──────────────────────────────────

/**
 * Create task management tools for the coordinator.
 * These tools are registered in addition to the coordinator's filtered tool set.
 */
export function createTaskManagementTools(pool: CoordinatorTaskPool): ReadonlyArray<ToolSpec> {
  return [
    createTaskCreateTool(pool),
    createTaskListTool(pool),
    createTaskGetTool(pool),
    createTaskStopTool(pool),
  ];
}

function createTaskCreateTool(pool: CoordinatorTaskPool): ToolSpec {
  return {
    name: "task_create",
    description: "Create a new worker task. The task starts as pending and can be assigned to a worker via the delegate tool.",
    category: "state",
    paramSchema: {
      type: "object",
      properties: { objective: { type: "string", description: "The task objective for the worker agent." } },
      required: ["objective"],
    },
    resultSchema: { type: "object", properties: { task_id: { type: "string" } } },
    handler: async (params) => createCoordinatorTask(pool, params),
  };
}

function createTaskListTool(pool: CoordinatorTaskPool): ToolSpec {
  return {
    name: "task_list",
    description: "List all worker tasks with their current status.",
    category: "state",
    paramSchema: { type: "object", properties: {} },
    resultSchema: { type: "object", properties: { tasks: { type: "array" } } },
    handler: async () => listCoordinatorTasks(pool),
  };
}

function createTaskGetTool(pool: CoordinatorTaskPool): ToolSpec {
  return {
    name: "task_get",
    description: "Get details of a specific worker task.",
    category: "state",
    paramSchema: {
      type: "object",
      properties: { task_id: { type: "string", description: "Task ID" } },
      required: ["task_id"],
    },
    resultSchema: { type: "object" },
    handler: async (params) => getCoordinatorTask(pool, params),
  };
}

function createTaskStopTool(pool: CoordinatorTaskPool): ToolSpec {
  return {
    name: "task_stop",
    description: "Stop (kill) a running or pending worker task.",
    category: "state",
    paramSchema: {
      type: "object",
      properties: { task_id: { type: "string", description: "Task ID to stop" } },
      required: ["task_id"],
    },
    resultSchema: { type: "object" },
    handler: async (params) => stopCoordinatorTask(pool, params),
  };
}

function createCoordinatorTask(pool: CoordinatorTaskPool, params: Record<string, unknown>) {
  const objective = params["objective"] as string;
  if (!objective?.trim()) {
    return { success: false, output: "", error: "Objective is required", artifacts: [] };
  }
  const taskId = pool.createTask(objective);
  return { success: true, output: `Created task ${taskId}: ${objective}`, error: null, artifacts: [] };
}

function listCoordinatorTasks(pool: CoordinatorTaskPool) {
  const tasks = pool.listTasks();
  if (tasks.length === 0) return { success: true, output: "No tasks.", error: null, artifacts: [] };
  return { success: true, output: tasks.map(formatCoordinatorTaskSummary).join("\n"), error: null, artifacts: [] };
}

function formatCoordinatorTaskSummary(t: WorkerTask) {
  const duration = t.completedAt ? ` (${Math.round((t.completedAt - t.createdAt) / 1000)}s)` : "";
  return `- ${t.taskId} [${t.status}]${duration}: ${t.objective}`;
}

function getCoordinatorTask(pool: CoordinatorTaskPool, params: Record<string, unknown>) {
  const taskId = params["task_id"] as string;
  const task = pool.getTask(taskId);
  if (!task) return { success: false, output: "", error: `Task ${taskId} not found`, artifacts: [] };
  return { success: true, output: formatCoordinatorTaskDetails(task), error: null, artifacts: [] };
}

function formatCoordinatorTaskDetails(task: WorkerTask) {
  return [
    `Task: ${task.taskId}`,
    `Status: ${task.status}`,
    `Objective: ${task.objective}`,
    task.agentId ? `Agent: ${task.agentId}` : null,
    task.result ? `Result: ${task.result.slice(0, 500)}` : null,
    task.error ? `Error: ${task.error}` : null,
  ].filter(Boolean).join("\n");
}

function stopCoordinatorTask(pool: CoordinatorTaskPool, params: Record<string, unknown>) {
  const taskId = params["task_id"] as string;
  const killed = pool.killTask(taskId);
  if (!killed) {
    return { success: false, output: "", error: `Cannot stop task ${taskId} (not running/pending)`, artifacts: [] };
  }
  return { success: true, output: `Task ${taskId} stopped.`, error: null, artifacts: [] };
}
