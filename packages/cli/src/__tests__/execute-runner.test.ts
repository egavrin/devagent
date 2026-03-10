import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { spawn } from "node:child_process";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { afterEach, describe, expect, it } from "vitest";
import { buildWorkflowInput, formatPhaseReport } from "../execute-runner.js";
import type { TaskExecutionRequest } from "@devagent-sdk/types";

const tempPaths: string[] = [];

afterEach(async () => {
  await Promise.all(tempPaths.splice(0).map((path) => rm(path, { recursive: true, force: true })));
});

type RequestOverrides = {
  taskType?: "triage" | "plan" | "implement" | "verify" | "review" | "repair";
  verifyCommands?: string[];
  allowNetwork?: boolean;
  expectedArtifacts?: string[];
};

async function createRequest(overrides: RequestOverrides = {}): Promise<{
  root: string;
  requestPath: string;
  artifactDir: string;
}> {
  const root = await mkdtemp(join(tmpdir(), "devagent-execute-test-"));
  tempPaths.push(root);
  const artifactDir = join(root, "artifacts");
  await mkdir(artifactDir, { recursive: true });
  const requestPath = join(root, "request.json");
  const taskType = overrides.taskType ?? "plan";
  await writeFile(requestPath, JSON.stringify({
    protocolVersion: "0.1",
    taskId: "task-123",
    taskType,
    project: {
      id: "org/repo",
      name: "repo",
    },
    workItem: {
      kind: "github-issue",
      externalId: "42",
      title: "Test execute",
    },
    workspace: {
      sourceRepoPath: root,
      workBranch: "test/execute",
      isolation: "temp-copy",
    },
    executor: {
      executorId: "devagent",
      provider: "chatgpt",
      model: "gpt-5.4",
      approvalMode: "full-auto",
    },
    constraints: {
      verifyCommands: overrides.verifyCommands,
      allowNetwork: overrides.allowNetwork ?? true,
    },
    context: {
      summary: "Test execute path",
    },
    expectedArtifacts: overrides.expectedArtifacts
      ?? [taskType === "verify" ? "verification-report" : "plan"],
  }, null, 2));
  return { root, requestPath, artifactDir };
}

function makeRequest(overrides: Partial<TaskExecutionRequest> = {}): TaskExecutionRequest {
  return {
    protocolVersion: "0.1",
    taskId: "task-123",
    taskType: "plan",
    project: {
      id: "org/repo",
      name: "repo",
    },
    workItem: {
      kind: "github-issue",
      externalId: "42",
      title: "Test execute",
    },
    workspace: {
      sourceRepoPath: "/tmp/repo",
      workBranch: "test/execute",
      isolation: "temp-copy",
    },
    executor: {
      executorId: "devagent",
      provider: "chatgpt",
      model: "gpt-5.4",
      approvalMode: "full-auto",
    },
    constraints: {
      allowNetwork: true,
    },
    context: {
      summary: "Test execute path",
    },
    expectedArtifacts: ["plan"],
    ...overrides,
  };
}

async function runExecute(
  requestPath: string,
  artifactDir: string,
  cwd: string,
  env: NodeJS.ProcessEnv = {},
): Promise<{ code: number; stdout: string; stderr: string }> {
  return await new Promise((resolve) => {
    const child = spawn("bun", [
      join(process.cwd(), "src/index.ts"),
      "execute",
      "--request",
      requestPath,
      "--artifact-dir",
      artifactDir,
    ], {
      cwd,
      env: { ...process.env, ...env },
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk: Buffer) => {
      stdout += chunk.toString();
    });
    child.stderr.on("data", (chunk: Buffer) => {
      stderr += chunk.toString();
    });
    child.once("close", (code) => {
      resolve({ code: code ?? 1, stdout, stderr });
    });
  });
}

describe("execute-runner", () => {
  it("builds implement workflow input from the accepted plan instruction", () => {
    const input = buildWorkflowInput(makeRequest({
      taskType: "implement",
      expectedArtifacts: ["implementation-summary"],
      context: {
        summary: "implement for issue #42",
        extraInstructions: ["Accepted plan:\n1. Update src/app.ts\n2. Add tests"],
      },
    }));

    expect(input.acceptedPlan).toBe("1. Update src/app.ts\n2. Add tests");
  });

  it("does not emit the review pass sentinel when findings are present", () => {
    const report = formatPhaseReport("review", {
      summary: "Found issues.",
      findings: [{ file: "src/app.ts", line: 7, severity: "major", message: "Bug", category: "bug" }],
    });

    expect(report).not.toContain("No defects found.");
    expect(report).toContain("## Findings");
  });

  it("does not emit the review pass sentinel for an explicit block verdict", () => {
    const report = formatPhaseReport("review", {
      summary: "Blocked pending investigation.",
      verdict: "block",
      blockingCount: 0,
      findings: [],
    });

    expect(report).not.toContain("No defects found.");
    expect(report).toContain("**Verdict**: block");
  });

  it("writes protocol events and result files for fake responses", async () => {
    const { root, requestPath, artifactDir } = await createRequest();
    const result = await runExecute(requestPath, artifactDir, root, {
      DEVAGENT_EXECUTOR_FAKE_RESPONSE_PLAN: "# Plan\n\nShip it",
    });

    expect(result.code).toBe(0);
    const resultJson = JSON.parse(await readFile(join(artifactDir, "result.json"), "utf-8")) as {
      status: string;
      artifacts: Array<{ path: string }>;
    };
    expect(resultJson.status).toBe("success");
    expect(await readFile(join(artifactDir, "plan.md"), "utf-8")).toContain("Ship it");

    const eventTypes = result.stdout
      .trim()
      .split("\n")
      .map((line) => JSON.parse(line).type);
    expect(eventTypes).toEqual(["started", "artifact", "completed"]);
    expect(resultJson.artifacts[0]?.path).toBe(join(artifactDir, "plan.md"));
  });

  it("runs verify commands and writes a verification report", async () => {
    const { root, requestPath, artifactDir } = await createRequest({
      taskType: "verify",
      verifyCommands: [`${process.execPath} -e "process.stdout.write('ok')"`],
      expectedArtifacts: ["verification-report"],
    });
    const result = await runExecute(requestPath, artifactDir, root);

    expect(result.code).toBe(0);
    const report = await readFile(join(artifactDir, "verification-report.md"), "utf-8");
    expect(report).toContain("Overall result: pass");
    expect(report).toContain("ok");
  });

  it("fails fast when verification commands fail", async () => {
    const { root, requestPath, artifactDir } = await createRequest({
      taskType: "verify",
      verifyCommands: [`${process.execPath} -e "process.exit(2)"`],
      expectedArtifacts: ["verification-report"],
    });
    const result = await runExecute(requestPath, artifactDir, root);

    expect(result.code).toBe(1);
    const resultJson = JSON.parse(await readFile(join(artifactDir, "result.json"), "utf-8")) as {
      status: string;
      error?: { message: string };
    };
    expect(resultJson.status).toBe("failed");
    expect(resultJson.error?.message).toContain("verification commands failed");
  });
});
