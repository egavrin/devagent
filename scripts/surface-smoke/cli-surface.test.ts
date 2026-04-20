import { afterEach, describe, expect, it } from "bun:test";
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { cp, mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const ROOT = resolve(import.meta.dirname, "..", "..");
const CLI_ENTRYPOINT = join(ROOT, "packages", "cli", "src", "index.ts");
const FIXTURE_REPO = resolve(import.meta.dirname, "fixtures", "sample-repo");
const tempPaths: string[] = [];

afterEach(async () => {
  await Promise.all(tempPaths.splice(0).map((path) => rm(path, { recursive: true, force: true })));
});

async function makeTempDir(prefix: string): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), prefix));
  tempPaths.push(dir);
  return dir;
}

async function runDevagent(
  args: string[],
  options: {
    cwd?: string;
    env?: NodeJS.ProcessEnv;
  } = {},
): Promise<{ code: number; stdout: string; stderr: string }> {
  return await new Promise((resolvePromise) => {
    const child = spawn("bun", [CLI_ENTRYPOINT, ...args], {
      cwd: options.cwd ?? ROOT,
      env: { ...process.env, DEVAGENT_DISABLE_UPDATE_CHECK: "1", ...options.env },
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
      resolvePromise({ code: code ?? 1, stdout, stderr });
    });
  });
}

async function createExecuteRequest(root: string, artifactDir: string): Promise<string> {
  const requestPath = join(root, "request.json");
  const repositoryId = "repo-1";
  await writeFile(requestPath, JSON.stringify({
    protocolVersion: "0.1",
    taskId: "surface-plan",
    taskType: "plan",
    workspaceRef: {
      id: "workspace-1",
      name: "surface-fixture",
      provider: "github",
      primaryRepositoryId: repositoryId,
    },
    repositories: [{
      id: repositoryId,
      workspaceId: "workspace-1",
      alias: "primary",
      name: "surface-fixture",
      repoRoot: root,
      repoFullName: "org/surface-fixture",
      defaultBranch: "main",
      provider: "github",
    }],
    workItem: {
      id: "issue-1",
      kind: "github-issue",
      externalId: "surface-plan",
      title: "Create a concise implementation plan",
      repositoryId,
    },
    execution: {
      primaryRepositoryId: repositoryId,
      repositories: [{
        repositoryId,
        alias: "primary",
        sourceRepoPath: root,
        workBranch: "devagent/surface-plan",
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
      allowNetwork: true,
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
      summary: "Use README.md as the planning context.",
    },
    expectedArtifacts: ["plan"],
  }, null, 2));
  await mkdir(artifactDir, { recursive: true });
  return requestPath;
}

describe("surface smoke", () => {
  it("prints top-level help through the real CLI entrypoint", async () => {
    const result = await runDevagent(["--help"]);

    expect(result.code).toBe(0);
    expect(result.stdout).toContain("Usage:");
    expect(result.stdout).toContain("devagent execute --request <file> --artifact-dir <dir>");
    expect(result.stdout).toContain("devagent sessions");
  });

  it("lists sessions successfully when no sessions exist", async () => {
    const home = await makeTempDir("devagent-surface-home-");
    const result = await runDevagent(["sessions"], {
      env: { HOME: home },
    });

    expect(result.code).toBe(0);
    expect(result.stdout).toBe("");
    expect(result.stderr).toContain("No sessions found.");
  });

  it("executes the public machine contract hermetically and writes artifacts", async () => {
    const workspaceRoot = await makeTempDir("devagent-surface-workspace-");
    const repoRoot = join(workspaceRoot, "repo");
    const artifactDir = join(workspaceRoot, "artifacts");
    await cp(FIXTURE_REPO, repoRoot, { recursive: true });

    const requestPath = await createExecuteRequest(repoRoot, artifactDir);
    const result = await runDevagent(
      ["execute", "--request", requestPath, "--artifact-dir", artifactDir],
      {
        cwd: repoRoot,
        env: {
          DEVAGENT_EXECUTOR_FAKE_RESPONSE_PLAN: "# Plan\n\n- Inspect README.md\n- Draft the implementation plan around the documented scope",
        },
      },
    );

    expect(result.code).toBe(0);
    const eventTypes = result.stdout.trim().split("\n").map((line) => JSON.parse(line).type);
    expect(eventTypes[0]).toBe("started");
    expect(eventTypes).toContain("artifact");
    expect(eventTypes.at(-1)).toBe("completed");

    const planPath = join(artifactDir, "plan.md");
    const resultJsonPath = join(artifactDir, "result.json");
    expect(existsSync(planPath)).toBe(true);
    expect(existsSync(resultJsonPath)).toBe(true);

    const plan = await readFile(planPath, "utf-8");
    expect(plan).toContain("Inspect README.md");

    const resultJson = JSON.parse(await readFile(resultJsonPath, "utf-8")) as {
      status: string;
      artifacts: Array<{ path: string; kind: string }>;
    };
    expect(resultJson.status).toBe("success");
    expect(resultJson.artifacts).toHaveLength(1);
    expect(resultJson.artifacts[0]?.kind).toBe("plan");
    expect(resultJson.artifacts[0]?.path).toBe(planPath);

    const eventsPath = join(artifactDir, "engine-events.jsonl");
    expect(existsSync(eventsPath)).toBe(false);
  });
});
