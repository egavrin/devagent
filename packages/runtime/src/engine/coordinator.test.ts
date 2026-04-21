import { describe, it, expect, beforeEach } from "vitest";

import {
  CoordinatorTaskPool,
  filterCoordinatorTools,
  createTaskManagementTools,
} from "./coordinator.js";
import type { ToolSpec, ToolCategory } from "../core/index.js";
import { ToolRegistry } from "../tools/registry.js";

// ─── Helpers ────────────────────────────────────────────────

function makeTool(name: string, category: ToolCategory): ToolSpec {
  return {
    name,
    description: `${name} tool`,
    category,
    paramSchema: { type: "object", properties: {} },
    resultSchema: { type: "object", properties: {} },
    handler: async () => ({ success: true, output: "ok", error: null, artifacts: [] }),
  };
}

// ─── CoordinatorTaskPool ────────────────────────────────────
let pool: CoordinatorTaskPool;

beforeEach(() => {
  pool = new CoordinatorTaskPool(2);
});

describe("createTask", () => {
  it("creates a task with pending status", () => {
    const taskId = pool.createTask("Build feature X");
    const task = pool.getTask(taskId);

    expect(task).toBeDefined();
    expect(task!.status).toBe("pending");
    expect(task!.objective).toBe("Build feature X");
    expect(task!.agentId).toBeNull();
    expect(task!.result).toBeNull();
    expect(task!.error).toBeNull();
  });

  it("assigns unique task IDs", () => {
    const id1 = pool.createTask("Task 1");
    const id2 = pool.createTask("Task 2");
    expect(id1).not.toBe(id2);
  });
});

describe("startTask", () => {
  it("transitions pending task to running", () => {
    const taskId = pool.createTask("Do work");
    const started = pool.startTask(taskId, "agent-1");

    expect(started).toBe(true);
    const task = pool.getTask(taskId);
    expect(task!.status).toBe("running");
    expect(task!.agentId).toBe("agent-1");
  });

  it("returns false for non-existent task", () => {
    expect(pool.startTask("bogus", "agent-1")).toBe(false);
  });

  it("returns false for already running task", () => {
    const taskId = pool.createTask("Work");
    pool.startTask(taskId, "agent-1");
    expect(pool.startTask(taskId, "agent-2")).toBe(false);
  });

  it("rejects start when maxWorkers reached", () => {
    const t1 = pool.createTask("Task 1");
    const t2 = pool.createTask("Task 2");
    const t3 = pool.createTask("Task 3");

    pool.startTask(t1, "a1");
    pool.startTask(t2, "a2");

    // maxWorkers is 2, so third should be rejected
    expect(pool.startTask(t3, "a3")).toBe(false);
    expect(pool.getTask(t3)!.status).toBe("pending");
  });
});

describe("completeTask", () => {
  it("transitions running task to completed", () => {
    const taskId = pool.createTask("Work");
    pool.startTask(taskId, "agent-1");

    const completed = pool.completeTask(taskId, "All done");
    expect(completed).toBe(true);

    const task = pool.getTask(taskId);
    expect(task!.status).toBe("completed");
    expect(task!.result).toBe("All done");
    expect(task!.completedAt).not.toBeNull();
  });

  it("returns false for pending task", () => {
    const taskId = pool.createTask("Work");
    expect(pool.completeTask(taskId, "done")).toBe(false);
  });
});

describe("failTask", () => {
  it("transitions running task to failed", () => {
    const taskId = pool.createTask("Work");
    pool.startTask(taskId, "agent-1");

    const failed = pool.failTask(taskId, "Something broke");
    expect(failed).toBe(true);

    const task = pool.getTask(taskId);
    expect(task!.status).toBe("failed");
    expect(task!.error).toBe("Something broke");
    expect(task!.completedAt).not.toBeNull();
  });

  it("returns false for pending task", () => {
    const taskId = pool.createTask("Work");
    expect(pool.failTask(taskId, "err")).toBe(false);
  });
});

describe("killTask", () => {
  it("kills a running task", () => {
    const taskId = pool.createTask("Work");
    pool.startTask(taskId, "agent-1");

    expect(pool.killTask(taskId)).toBe(true);
    expect(pool.getTask(taskId)!.status).toBe("killed");
  });

  it("kills a pending task", () => {
    const taskId = pool.createTask("Work");
    expect(pool.killTask(taskId)).toBe(true);
    expect(pool.getTask(taskId)!.status).toBe("killed");
  });

  it("returns false for completed task", () => {
    const taskId = pool.createTask("Work");
    pool.startTask(taskId, "agent-1");
    pool.completeTask(taskId, "done");

    expect(pool.killTask(taskId)).toBe(false);
  });

  it("returns false for failed task", () => {
    const taskId = pool.createTask("Work");
    pool.startTask(taskId, "agent-1");
    pool.failTask(taskId, "err");

    expect(pool.killTask(taskId)).toBe(false);
  });
});

describe("listTasks and getTask", () => {
  it("lists all tasks", () => {
    pool.createTask("Task A");
    pool.createTask("Task B");

    const tasks = pool.listTasks();
    expect(tasks).toHaveLength(2);
    expect(tasks.map((t) => t.objective)).toEqual(["Task A", "Task B"]);
  });

  it("getTask returns undefined for missing task", () => {
    expect(pool.getTask("nope")).toBeUndefined();
  });
});

describe("hasCapacity", () => {
  it("returns true when under limit", () => {
    expect(pool.hasCapacity()).toBe(true);

    const t1 = pool.createTask("Task 1");
    pool.startTask(t1, "a1");
    expect(pool.hasCapacity()).toBe(true);
  });

  it("returns false when at limit", () => {
    const t1 = pool.createTask("Task 1");
    const t2 = pool.createTask("Task 2");
    pool.startTask(t1, "a1");
    pool.startTask(t2, "a2");

    expect(pool.hasCapacity()).toBe(false);
  });

  it("regains capacity after task completes", () => {
    const t1 = pool.createTask("Task 1");
    const t2 = pool.createTask("Task 2");
    pool.startTask(t1, "a1");
    pool.startTask(t2, "a2");

    expect(pool.hasCapacity()).toBe(false);

    pool.completeTask(t1, "done");
    expect(pool.hasCapacity()).toBe(true);
  });
});


// ─── filterCoordinatorTools ─────────────────────────────────

describe("filterCoordinatorTools", () => {
  it("allows readonly, workflow, and state tools", () => {
    const registry = new ToolRegistry();
    registry.register(makeTool("read_file", "readonly"));
    registry.register(makeTool("delegate", "workflow"));
    registry.register(makeTool("session_state", "state"));

    const filtered = filterCoordinatorTools(registry);
    expect(filtered.map((t) => t.name)).toEqual(["read_file", "delegate", "session_state"]);
  });

  it("blocks mutating and external tools", () => {
    const registry = new ToolRegistry();
    registry.register(makeTool("read_file", "readonly"));
    registry.register(makeTool("run_shell", "mutating"));
    registry.register(makeTool("web_fetch", "external"));

    const filtered = filterCoordinatorTools(registry);
    expect(filtered.map((t) => t.name)).toEqual(["read_file"]);
  });

  it("blocks specific tools even within allowed categories", () => {
    const registry = new ToolRegistry();
    // write_file is in the blocked list regardless of its category
    registry.register(makeTool("write_file", "readonly"));
    registry.register(makeTool("replace_in_file", "readonly"));
    registry.register(makeTool("run_command", "workflow"));
    registry.register(makeTool("search_files", "readonly"));

    const filtered = filterCoordinatorTools(registry);
    expect(filtered.map((t) => t.name)).toEqual(["search_files"]);
  });
});

// ─── createTaskManagementTools ──────────────────────────────

describe("createTaskManagementTools", () => {
  let pool: CoordinatorTaskPool;
  let tools: ReadonlyArray<ToolSpec>;

  beforeEach(() => {
    pool = new CoordinatorTaskPool(3);
    tools = createTaskManagementTools(pool);
  });

  it("creates 4 task management tools", () => {
    expect(tools).toHaveLength(4);
    expect(tools.map((t) => t.name)).toEqual([
      "task_create", "task_list", "task_get", "task_stop",
    ]);
  });

  it("task_create handler creates a task", async () => {
    const tool = tools.find((t) => t.name === "task_create")!;
    const result = await tool.handler({ objective: "Build feature" }, {} as any);

    expect(result.success).toBe(true);
    expect(result.output).toContain("Created task");
    expect(pool.listTasks()).toHaveLength(1);
  });

  it("task_create handler rejects empty objective", async () => {
    const tool = tools.find((t) => t.name === "task_create")!;
    const result = await tool.handler({ objective: "  " }, {} as any);

    expect(result.success).toBe(false);
    expect(result.error).toContain("Objective is required");
  });

  it("task_list handler lists tasks", async () => {
    pool.createTask("Task A");
    pool.createTask("Task B");

    const tool = tools.find((t) => t.name === "task_list")!;
    const result = await tool.handler({}, {} as any);

    expect(result.success).toBe(true);
    expect(result.output).toContain("Task A");
    expect(result.output).toContain("Task B");
  });

  it("task_list handler returns empty message when no tasks", async () => {
    const tool = tools.find((t) => t.name === "task_list")!;
    const result = await tool.handler({}, {} as any);

    expect(result.success).toBe(true);
    expect(result.output).toBe("No tasks.");
  });

  it("task_get handler returns task details", async () => {
    const taskId = pool.createTask("My objective");
    pool.startTask(taskId, "agent-x");

    const tool = tools.find((t) => t.name === "task_get")!;
    const result = await tool.handler({ task_id: taskId }, {} as any);

    expect(result.success).toBe(true);
    expect(result.output).toContain("My objective");
    expect(result.output).toContain("running");
    expect(result.output).toContain("agent-x");
  });

  it("task_get handler returns error for missing task", async () => {
    const tool = tools.find((t) => t.name === "task_get")!;
    const result = await tool.handler({ task_id: "bogus" }, {} as any);

    expect(result.success).toBe(false);
    expect(result.error).toContain("not found");
  });

  it("task_stop handler kills a running task", async () => {
    const taskId = pool.createTask("Work");
    pool.startTask(taskId, "agent-1");

    const tool = tools.find((t) => t.name === "task_stop")!;
    const result = await tool.handler({ task_id: taskId }, {} as any);

    expect(result.success).toBe(true);
    expect(result.output).toContain("stopped");
    expect(pool.getTask(taskId)!.status).toBe("killed");
  });

  it("task_stop handler returns error for completed task", async () => {
    const taskId = pool.createTask("Work");
    pool.startTask(taskId, "agent-1");
    pool.completeTask(taskId, "done");

    const tool = tools.find((t) => t.name === "task_stop")!;
    const result = await tool.handler({ task_id: taskId }, {} as any);

    expect(result.success).toBe(false);
    expect(result.error).toContain("Cannot stop");
  });
});
