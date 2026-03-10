import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { describe, expect, it } from "vitest";
import { PROTOCOL_VERSION, type TaskExecutionEvent, type TaskExecutionRequest } from "@devagent-sdk/types";
import {
  artifactInfoForTask,
  buildTaskQuery,
  executeTask,
  executeVerifyCommands,
  extractArtifactBody,
  loadTaskExecutionRequest,
  parseExecuteArgs,
  readFakeTaskResponse,
  resolveRequestedSkills,
  validateExecutionCapabilities,
} from "./index.js";

function createRequest(taskType: TaskExecutionRequest["taskType"]): TaskExecutionRequest {
  return {
    protocolVersion: PROTOCOL_VERSION,
    taskId: `task-${taskType}`,
    taskType,
    project: { id: "p1", name: "repo" },
    workItem: { kind: "github-issue", externalId: "42", title: "Refactor runner" },
    workspace: {
      sourceRepoPath: "/tmp/repo",
      workBranch: `devagent/${taskType}/task-${taskType}`,
      isolation: "temp-copy",
    },
    executor: {
      executorId: "devagent",
      provider: "chatgpt",
      model: "gpt-5.4",
      approvalMode: "full-auto",
      reasoning: "medium",
    },
    constraints: {
      maxIterations: 5,
    },
    context: {
      summary: `Handle ${taskType}`,
      skills: [],
    },
    expectedArtifacts: [artifactInfoForTask(taskType).kind],
  };
}

async function createRepoRoot(): Promise<string> {
  return mkdtemp(join(tmpdir(), "devagent-executor-test-"));
}

describe("parseExecuteArgs", () => {
  it("parses execute args", () => {
    expect(parseExecuteArgs(["node", "devagent", "execute", "--request", "request.json", "--artifact-dir", "artifacts"]))
      .toEqual({
        requestPath: expect.stringContaining("request.json"),
        artifactDir: expect.stringContaining("artifacts"),
      });
  });
});

describe("request validation", () => {
  it("loads a valid SDK request", async () => {
    const dir = await createRepoRoot();
    const path = join(dir, "request.json");
    await writeFile(path, JSON.stringify(createRequest("plan")));

    await expect(loadTaskExecutionRequest(path)).resolves.toMatchObject({
      taskId: "task-plan",
      taskType: "plan",
    });

    await rm(dir, { recursive: true, force: true });
  });

  it("fails on invalid request payloads", async () => {
    const dir = await createRepoRoot();
    const path = join(dir, "request.json");
    await writeFile(path, JSON.stringify({ protocolVersion: PROTOCOL_VERSION, taskId: "missing-fields" }));

    await expect(loadTaskExecutionRequest(path)).rejects.toThrow(/Invalid TaskExecutionRequest/);

    await rm(dir, { recursive: true, force: true });
  });
});

describe("capability validation", () => {
  it("rejects unsupported executor routing", () => {
    const request = createRequest("plan");
    request.executor.executorId = "codex";
    expect(() => validateExecutionCapabilities(request)).toThrow(/Unsupported executor/);
  });

  it("rejects unsupported network restrictions", () => {
    const request = createRequest("plan");
    request.constraints.allowNetwork = false;
    expect(() => validateExecutionCapabilities(request)).toThrow(/allowNetwork=false/);
  });
});

describe("skills", () => {
  it("resolves requested skills through the registry", async () => {
    const repoRoot = await createRepoRoot();
    const skillDir = join(repoRoot, ".devagent", "skills", "testing");
    await mkdir(skillDir, { recursive: true });
    await writeFile(
      join(skillDir, "SKILL.md"),
      "---\nname: testing\ndescription: Run tests carefully\n---\nCheck the test suite.\n",
    );

    const resolved = await resolveRequestedSkills(repoRoot, ["testing"], "session-1");
    expect(resolved).toHaveLength(1);
    expect(resolved[0]).toMatchObject({
      name: "testing",
      description: "Run tests carefully",
    });
    expect(resolved[0]?.instructions).toContain("Check the test suite.");

    await rm(repoRoot, { recursive: true, force: true });
  });

  it("injects resolved skill instructions into the task query", () => {
    const query = buildTaskQuery(createRequest("plan"), [
      {
        name: "testing",
        description: "Run tests carefully",
        source: "project",
        instructions: "Check the test suite.",
      },
    ]);
    expect(query).toContain("Requested skills");
    expect(query).toContain("Check the test suite.");
  });
});

describe("verify commands", () => {
  it("executes verify commands and returns a markdown report", async () => {
    const repoRoot = await createRepoRoot();
    const result = await executeVerifyCommands(
      [
        `${process.execPath} -e "process.stdout.write('ok')"` ,
        `${process.execPath} -e "process.stderr.write('warn')"` ,
      ],
      repoRoot,
    );

    expect(result.success).toBe(true);
    expect(result.report).toContain("Overall result: pass");
    expect(result.report).toContain("ok");
    expect(result.report).toContain("warn");

    await rm(repoRoot, { recursive: true, force: true });
  });
});

describe("artifact body extraction", () => {
  it("extracts the plain result body from task-loop envelopes", () => {
    const body = extractArtifactBody(JSON.stringify({
      subtype: "success",
      result: "## Plan\n\nKeep the change set small.",
    }));
    expect(body).toBe("## Plan\n\nKeep the change set small.");
  });

  it("extracts the plain result body from heading-wrapped task-loop envelopes", () => {
    const body = extractArtifactBody(
      '# Plan\n\n{"result":"# Plan\\n\\nKeep the change set small."}',
    );
    expect(body).toBe("# Plan\n\nKeep the change set small.");
  });
});

describe("task execution", () => {
  it("uses fake executor responses for non-verify tasks when configured", async () => {
    const repoRoot = await createRepoRoot();
    const artifactDir = join(repoRoot, "artifacts");
    const request = createRequest("plan");
    const original = process.env["DEVAGENT_EXECUTOR_FAKE_RESPONSE_PLAN"];
    process.env["DEVAGENT_EXECUTOR_FAKE_RESPONSE_PLAN"] = "Fake baseline response";

    try {
      expect(readFakeTaskResponse("plan")).toBe("Fake baseline response");
      const result = await executeTask({
        request,
        artifactDir,
        repoRoot,
        runQuery: async () => {
          throw new Error("runQuery should not be called when fake responses are configured");
        },
        emit: () => {},
      });

      expect(result.status).toBe("success");
      const artifactText = await readFile(result.artifacts[0]!.path, "utf-8");
      expect(artifactText).toContain("Fake baseline response");
    } finally {
      if (original === undefined) {
        delete process.env["DEVAGENT_EXECUTOR_FAKE_RESPONSE_PLAN"];
      } else {
        process.env["DEVAGENT_EXECUTOR_FAKE_RESPONSE_PLAN"] = original;
      }
      await rm(repoRoot, { recursive: true, force: true });
    }
  });

  for (const taskType of ["triage", "plan", "implement", "verify", "review", "repair"] as const) {
    it(`emits artifacts and events for ${taskType}`, async () => {
      const repoRoot = await createRepoRoot();
      const artifactDir = join(repoRoot, "artifacts");
      const request = createRequest(taskType);
      if (taskType === "verify") {
        request.constraints.verifyCommands = [`${process.execPath} -e "process.stdout.write('verify-pass')"`];
      } else {
        request.context.skills = ["testing"];
        const skillDir = join(repoRoot, ".devagent", "skills", "testing");
        await mkdir(skillDir, { recursive: true });
        await writeFile(
          join(skillDir, "SKILL.md"),
          "---\nname: testing\ndescription: Run tests carefully\n---\nCheck the test suite.\n",
        );
      }

      const events: TaskExecutionEvent[] = [];
      const result = await executeTask({
        request,
        artifactDir,
        repoRoot,
        runQuery: async ({ query }) => ({
          success: true,
          responseText: `Handled ${taskType}\n\n${query}`,
          iterations: 1,
        }),
        emit: (event) => {
          events.push(event);
        },
      });

      expect(result.status).toBe("success");
      expect(result.artifacts).toHaveLength(1);
      expect(result.artifacts[0]?.kind).toBe(artifactInfoForTask(taskType).kind);
      expect(events.map((event) => event.type)).toContain("started");
      expect(events.map((event) => event.type)).toContain("artifact");
      expect(events.at(-1)?.type).toBe("completed");

      const artifactText = await readFile(result.artifacts[0]!.path, "utf-8");
      if (taskType === "verify") {
        expect(artifactText).toContain("Overall result: pass");
        expect(events.some((event) => event.type === "log")).toBe(true);
      } else {
        expect(artifactText).toContain(`Handled ${taskType}`);
      }

      await rm(repoRoot, { recursive: true, force: true });
    });
  }

  it("returns a failed result when verification commands fail", async () => {
    const repoRoot = await createRepoRoot();
    const artifactDir = join(repoRoot, "artifacts");
    const request = createRequest("verify");
    request.constraints.verifyCommands = [`${process.execPath} -e "process.exit(2)"`];
    const events: TaskExecutionEvent[] = [];

    const result = await executeTask({
      request,
      artifactDir,
      repoRoot,
      runQuery: async () => ({
        success: true,
        responseText: "unused",
        iterations: 1,
      }),
      emit: (event) => {
        events.push(event);
      },
    });

    expect(result.status).toBe("failed");
    expect(result.error?.message).toMatch(/verification commands failed/i);
    expect(events.at(-1)).toMatchObject({ type: "completed", status: "failed" });

    await rm(repoRoot, { recursive: true, force: true });
  });
});
