import { spawn } from "node:child_process";
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";

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
  const root = await mkdtemp(join(tmpdir(), "devagent-execute-command-test-"));
  tempPaths.push(root);
  const artifactDir = join(root, "artifacts");
  await mkdir(artifactDir, { recursive: true });
  const requestPath = join(root, "request.json");
  const taskType = overrides.taskType ?? "plan";
  const workspaceId = "workspace-1";
  const repositoryId = "repo-1";
  await writeFile(requestPath, JSON.stringify({
    protocolVersion: "0.1",
    taskId: "task-123",
    taskType,
    workspaceRef: {
      id: workspaceId,
      name: "repo",
      provider: "github",
      primaryRepositoryId: repositoryId,
    },
    repositories: [{
      id: repositoryId,
      workspaceId,
      alias: "primary",
      name: "repo",
      repoRoot: root,
      repoFullName: "org/repo",
      defaultBranch: "main",
      provider: "github",
    }],
    workItem: {
      id: "issue-42",
      kind: "github-issue",
      externalId: "42",
      title: "Test execute",
      repositoryId,
    },
    execution: {
      primaryRepositoryId: repositoryId,
      repositories: [{
        repositoryId,
        alias: "primary",
        sourceRepoPath: root,
        workBranch: "test/execute",
        isolation: "temp-copy",
      }],
    },
    targetRepositoryIds: [repositoryId],
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
    capabilities: {
      canSyncTasks: true,
      canCreateTask: true,
      canComment: true,
      canReview: true,
      canMerge: true,
      canOpenReviewable: true,
    },
    context: {
      summary: "Test execute path",
    },
    expectedArtifacts: overrides.expectedArtifacts
      ?? [taskType === "verify" ? "verification-report" : "plan"],
  }, null, 2));
  return { root, requestPath, artifactDir };
}

async function runCli(
  args: string[],
  cwd: string,
  env: NodeJS.ProcessEnv = {},
): Promise<{ code: number; stdout: string; stderr: string }> {
  return await new Promise((resolve) => {
    const child = spawn("bun", [
      join(process.cwd(), "src/index.ts"),
      ...args,
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

describe("execute command", () => {
  it("routes execute through the canonical executor path", async () => {
    const { root, requestPath, artifactDir } = await createRequest();
    const result = await runCli(
      ["execute", "--request", requestPath, "--artifact-dir", artifactDir],
      root,
      { DEVAGENT_EXECUTOR_FAKE_RESPONSE_PLAN: "# Plan\n\nShip it" },
    );

    expect(result.code).toBe(0);
    expect(await readFile(join(artifactDir, "plan.md"), "utf-8")).toContain("Ship it");

    const eventTypes = result.stdout
      .trim()
      .split("\n")
      .map((line) => JSON.parse(line).type);
    expect(eventTypes).toEqual(["started", "progress", "artifact", "completed"]);
  });

  it("fails when verification commands fail", async () => {
    const { root, requestPath, artifactDir } = await createRequest({
      taskType: "verify",
      verifyCommands: [`${process.execPath} -e "process.exit(2)"`],
      expectedArtifacts: ["verification-report"],
    });

    const result = await runCli(
      ["execute", "--request", requestPath, "--artifact-dir", artifactDir],
      root,
    );

    expect(result.code).toBe(1);
    const resultJson = JSON.parse(await readFile(join(artifactDir, "result.json"), "utf-8")) as {
      status: string;
      error?: { message: string };
    };
    expect(resultJson.status).toBe("failed");
    expect(resultJson.error?.message).toContain("verification commands failed");
  });

  it("removes the legacy workflow subcommand dispatch from the CLI entrypoint", async () => {
    const mainSource = await readFile(join(process.cwd(), "src/main.ts"), "utf-8");

    expect(mainSource).not.toContain('process.argv[2] === "workflow"');
    expect(mainSource).not.toContain('import("./workflow-runner.js")');
    expect(mainSource).not.toContain("runInteractive(");
    expect(mainSource).not.toContain('arg === "chat"');
  });
});
