import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import { chmodSync, existsSync, lstatSync } from "node:fs";
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";

import {
  createIsolationWorkspace,
  createIsolationWorkspaceWithTimeout,
  destroyIsolationWorkspace,
} from "./isolation";
import { loadValidationScenarios , validateScenarioManifest } from "./manifest";
import { classifyProviderFailure, selectPreferredOllamaModel } from "./provider-smoke";
import {
  evaluateAssertions,
  renderSummaryMarkdown,
  renderScenarioReviewMarkdown,
  summarizeScenarioReports,
} from "./reporting";
import { runValidationScenario } from "./runner";
import type {
  ValidationAssertion,
  ValidationScenario,
  ValidationScenarioReport,
} from "./types";

const tempPaths: string[] = [];

async function makeTempDir(prefix: string): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), prefix));
  tempPaths.push(dir);
  return dir;
}

async function initGitRepo(repoRoot: string): Promise<void> {
  await Bun.$`git init`.cwd(repoRoot).quiet();
  await Bun.$`git config user.email devagent@example.com`.cwd(repoRoot).quiet();
  await Bun.$`git config user.name DevAgent Validation`.cwd(repoRoot).quiet();
}

async function commitAll(repoRoot: string, message: string): Promise<void> {
  await Bun.$`git add .`.cwd(repoRoot).quiet();
  await Bun.$`git commit -m ${message}`.cwd(repoRoot).quiet();
}

async function addLocalSubmodule(
  repoRoot: string,
  submoduleSourceRoot: string,
  relativePath: string,
): Promise<void> {
  await Bun.$`git -c protocol.file.allow=always submodule add ${submoduleSourceRoot} ${relativePath}`.cwd(repoRoot).quiet();
  await commitAll(repoRoot, "add submodule");
}

beforeEach(() => {
  tempPaths.length = 0;
});

afterEach(async () => {
  await Promise.all(tempPaths.splice(0).map((path) => rm(path, { recursive: true, force: true })));
});

describe("validateScenarioManifest", () => {
  it("accepts a valid manifest", () => {
    const scenario = validateScenarioManifest({
      id: "runtime-core-execute-triage",
      description: "Triage scenario",
      suites: ["smoke"],
      targetRepo: "arkcompiler_runtime_core_docs",
      surface: "execute",
      taskShape: "readonly",
      isolationMode: "temp-copy",
      invocation: {
        type: "execute",
        taskType: "triage",
        workItemTitle: "Triage malformed bytecode issue",
        summary: "Inspect verifier impact area.",
      },
      expectedArtifacts: ["triage-report.md"],
      assertions: [
        {
          type: "contains",
          source: "artifact",
          path: "triage-report.md",
          value: "verifier",
        },
      ],
      verificationCommands: [],
      cleanupPolicy: "destroy",
      requiredToolCalls: [{ tool: "delegate", minCalls: 1 }],
      requiredToolBatches: [{ tool: "delegate", minBatches: 1, minBatchSize: 2 }],
    }, "inline");

    expect(scenario.id).toBe("runtime-core-execute-triage");
    expect(scenario.surface).toBe("execute");
    expect(scenario.requiredToolCalls).toEqual([{ tool: "delegate", minCalls: 1 }]);
    expect(scenario.requiredToolBatches).toEqual([{ tool: "delegate", minBatches: 1, minBatchSize: 2 }]);
  });

  it("rejects invalid manifests", () => {
    expect(() => validateScenarioManifest({
      id: "broken",
      suites: ["smoke"],
      targetRepo: "arkcompiler_runtime_core",
      surface: "desktop",
    }, "broken.json")).toThrow(/surface/i);
  });

  it("loads the checked-in scenario manifests", () => {
    const scenarios = loadValidationScenarios(
      join(process.cwd(), "scripts", "live-validation", "scenarios"),
    );

    expect(scenarios.some((scenario) => scenario.id === "ets-frontend-execute-repair")).toBe(true);
    expect(scenarios.some((scenario) => scenario.id === "runtime-core-docs-execute-design")).toBe(true);
    expect(scenarios.some((scenario) => scenario.id === "runtime-core-docs-execute-breakdown")).toBe(true);
    expect(scenarios.some((scenario) => scenario.id === "runtime-core-docs-execute-issue-generation")).toBe(true);
    expect(scenarios.filter((scenario) => scenario.suites.includes("smoke"))).toHaveLength(3);
    expect(scenarios.filter((scenario) => scenario.suites.includes("full"))).toHaveLength(10);
    expect(scenarios.find((scenario) => scenario.id === "ets-frontend-execute-repair")?.suites).toEqual(["full"]);
    expect(scenarios.find((scenario) => scenario.id === "doctor-provider-model-mismatch")?.expectedExitCode).toBe(1);
    expect(scenarios.find((scenario) => scenario.id === "runtime-core-docs-execute-triage")?.requiredToolCalls).toEqual([
      { tool: "delegate", minCalls: 2 },
    ]);
    expect(scenarios.find((scenario) => scenario.id === "runtime-core-docs-execute-triage")?.requiredToolBatches).toEqual([
      { tool: "delegate", minBatches: 1, minBatchSize: 2 },
    ]);
    expect(scenarios.find((scenario) => scenario.id === "runtime-core-docs-execute-breakdown")?.expectedArtifacts).toEqual([
      "breakdown-doc.md",
      "breakdown-doc.json",
    ]);
    expect(scenarios.find((scenario) => scenario.id === "runtime-core-docs-execute-issue-generation")?.expectedArtifacts).toEqual([
      "issue-spec.md",
      "issue-spec.json",
    ]);
    expect(scenarios.find((scenario) => scenario.id === "runtime-core-docs-execute-triage")?.isolationMode).toBe("worktree");
    expect(scenarios.find((scenario) => scenario.id === "runtime-core-execute-plan")?.isolationMode).toBe("worktree");
    expect(scenarios.find((scenario) => scenario.id === "runtime-core-execute-triage")?.isolationMode).toBe("worktree");
    expect(scenarios.find((scenario) => scenario.id === "runtime-core-execute-triage")?.suites).toEqual(["full"]);
  });
});

describe("createIsolationWorkspace", () => {
  it("creates a tracked temp copy without copying source git metadata", async () => {
    const sourceRoot = await makeTempDir("devagent-live-src-");
    await writeFile(join(sourceRoot, "README.md"), "# Source\n");
    await mkdir(join(sourceRoot, "src"), { recursive: true });
    await writeFile(join(sourceRoot, "src", "main.ts"), "export const ok = true;\n");
    await initGitRepo(sourceRoot);
    await commitAll(sourceRoot, "initial");

    const isolationRoot = await makeTempDir("devagent-live-copy-");
    const workspace = await createIsolationWorkspace({
      mode: "temp-copy",
      sourceRoot,
      targetRoot: join(isolationRoot, "workspace"),
    });

    expect(await readFile(join(workspace.path, "src", "main.ts"), "utf-8")).toContain("ok = true");
    expect(existsSync(join(workspace.path, ".git", "HEAD"))).toBe(false);

    await destroyIsolationWorkspace(workspace);
  });

  it("creates and removes a disposable worktree", async () => {
    const sourceRoot = await makeTempDir("devagent-live-worktree-src-");
    await writeFile(join(sourceRoot, "README.md"), "# Source\n");
    await initGitRepo(sourceRoot);
    await commitAll(sourceRoot, "initial");

    const isolationRoot = await makeTempDir("devagent-live-worktree-target-");
    const workspace = await createIsolationWorkspace({
      mode: "worktree",
      sourceRoot,
      targetRoot: join(isolationRoot, "workspace"),
    });

    expect(existsSync(join(workspace.path, ".git"))).toBe(true);
    await destroyIsolationWorkspace(workspace);
    expect(existsSync(workspace.path)).toBe(false);
  });

  it("times out isolation steps with an explicit setup error", async () => {
    await expect(createIsolationWorkspaceWithTimeout({
      mode: "temp-copy",
      sourceRoot: "/tmp/source",
      targetRoot: "/tmp/target",
    }, 10, async () => {
      await Bun.sleep(50);
      return {
        mode: "temp-copy",
        path: "/tmp/target",
        sourceRoot: "/tmp/source",
      };
    })).rejects.toThrow(/setup failed during isolation/i);
  });

  it("materializes tracked submodules as isolated directories instead of source-linked symlinks", async () => {
    const submoduleSource = await makeTempDir("devagent-live-submodule-src-");
    await writeFile(join(submoduleSource, "lib.txt"), "submodule source\n");
    await initGitRepo(submoduleSource);
    await commitAll(submoduleSource, "initial");

    const sourceRoot = await makeTempDir("devagent-live-parent-src-");
    await writeFile(join(sourceRoot, "README.md"), "# Parent\n");
    await initGitRepo(sourceRoot);
    await commitAll(sourceRoot, "initial");
    await addLocalSubmodule(sourceRoot, submoduleSource, "vendor/submodule");

    const isolationRoot = await makeTempDir("devagent-live-submodule-copy-");
    const workspace = await createIsolationWorkspace({
      mode: "temp-copy",
      sourceRoot,
      targetRoot: join(isolationRoot, "workspace"),
    });

    const copiedSubmodulePath = join(workspace.path, "vendor", "submodule");
    expect(lstatSync(copiedSubmodulePath).isSymbolicLink()).toBe(false);
    expect(await readFile(join(copiedSubmodulePath, "lib.txt"), "utf-8")).toContain("submodule source");

    await writeFile(join(copiedSubmodulePath, "lib.txt"), "modified copy\n");
    expect(await readFile(join(sourceRoot, "vendor", "submodule", "lib.txt"), "utf-8")).toContain("submodule source");

    await destroyIsolationWorkspace(workspace);
  });
});

describe("evaluateAssertions", () => {
  it("evaluates contains and regex assertions against collected outputs", () => {
    const assertions: ValidationAssertion[] = [
      { type: "contains", source: "stdout", value: "review completed" },
      { type: "matches", source: "repoDiff", pattern: "command injection|eval" },
    ];

    const result = evaluateAssertions(assertions, {
      stdout: "review completed with findings",
      stderr: "",
      repoDiff: "Potential command injection via eval(userInput)",
      repoStatus: "",
      events: "",
      artifacts: new Map(),
    });

    expect(result.passed).toBe(true);
    expect(result.results).toHaveLength(2);
  });
});

describe("summarizeScenarioReports", () => {
  it("aggregates scenario outcomes", () => {
    const reports: ValidationScenarioReport[] = [
      {
        scenarioId: "a",
        description: "A",
        targetRepo: "arkcompiler_ets_frontend",
        surface: "execute",
        taskShape: "repair",
        taskType: "repair",
        provider: "chatgpt",
        model: "gpt-5.4",
        status: "passed",
        startedAt: new Date(0).toISOString(),
        finishedAt: new Date(1).toISOString(),
        durationMs: 1,
        sourceRepoPath: "/src/a",
        isolationPath: "/tmp/a",
        outputDir: "/tmp/out/a",
        command: { executable: "bun", args: ["run"], exitCode: 0 },
        artifactInventory: [],
        artifactValidation: { passed: true, checks: [] },
        repoMutation: {
          expectedWorkspaceEffect: "mutating",
          passed: true,
          observedChanges: true,
          summary: "Mutating scenario produced tracked workspace changes.",
        },
        assertionResults: [],
        toolCallAssertionResults: [],
        toolBatchAssertionResults: [],
        verificationResults: [],
        timing: { durationMs: 1 },
        cost: {},
        rawOutputs: {},
      },
      {
        scenarioId: "b",
        description: "B",
        targetRepo: "arkcompiler_runtime_core",
        surface: "cli",
        taskShape: "review",
        provider: "chatgpt",
        model: "gpt-5.4",
        status: "failed",
        failureClass: "assertion",
        startedAt: new Date(0).toISOString(),
        finishedAt: new Date(2).toISOString(),
        durationMs: 2,
        sourceRepoPath: "/src/b",
        isolationPath: "/tmp/b",
        outputDir: "/tmp/out/b",
        command: { executable: "bun", args: ["run"], exitCode: 1 },
        artifactInventory: [],
        artifactValidation: { passed: false, checks: [] },
        repoMutation: {
          expectedWorkspaceEffect: "non-mutating",
          passed: true,
          observedChanges: false,
          summary: "Readonly scenario kept the workspace clean.",
        },
        assertionResults: [],
        toolCallAssertionResults: [],
        toolBatchAssertionResults: [],
        verificationResults: [],
        timing: { durationMs: 2 },
        cost: {},
        rawOutputs: {},
      },
      {
        scenarioId: "c",
        description: "C",
        targetRepo: "arkcompiler_runtime_core_docs",
        surface: "cli",
        taskShape: "readonly",
        provider: "chatgpt",
        model: "gpt-5.4",
        status: "passed",
        startedAt: new Date(0).toISOString(),
        finishedAt: new Date(3).toISOString(),
        durationMs: 3,
        sourceRepoPath: "/src/c",
        isolationPath: "/tmp/c",
        outputDir: "/tmp/out/c",
        command: { executable: "bun", args: ["run"], exitCode: 0 },
        artifactInventory: [],
        artifactValidation: { passed: true, checks: [] },
        repoMutation: {
          expectedWorkspaceEffect: "non-mutating",
          passed: true,
          observedChanges: false,
          summary: "Readonly scenario kept the workspace clean.",
        },
        assertionResults: [],
        toolCallAssertionResults: [],
        toolBatchAssertionResults: [],
        verificationResults: [],
        timing: { durationMs: 3 },
        cost: {},
        rawOutputs: {},
      },
    ];

    const summary = summarizeScenarioReports(reports, {
      provider: "chatgpt",
      model: "gpt-5.4",
      suite: "smoke",
    });

    expect(summary.total).toBe(3);
    expect(summary.passed).toBe(2);
    expect(summary.failed).toBe(1);
    expect(summary.blocked).toBe(0);
    expect(summary.durationMs).toBe(6);
  });

  it("renders durations in the aggregate markdown", () => {
    const summary = summarizeScenarioReports([
      {
        scenarioId: "a",
        description: "A",
        targetRepo: "arkcompiler_runtime_core_docs",
        surface: "execute",
        taskShape: "readonly",
        taskType: "design",
        provider: "chatgpt",
        model: "gpt-5.4",
        status: "passed",
        startedAt: new Date(0).toISOString(),
        finishedAt: new Date(5).toISOString(),
        durationMs: 5,
        sourceRepoPath: "/src/a",
        isolationPath: "/tmp/a",
        outputDir: "/tmp/out/a",
        command: { executable: "bun", args: ["run"], exitCode: 0 },
        artifactInventory: [],
        artifactValidation: { passed: true, checks: [] },
        repoMutation: {
          expectedWorkspaceEffect: "non-mutating",
          passed: true,
          observedChanges: false,
          summary: "Readonly scenario kept the workspace clean.",
        },
        assertionResults: [],
        toolCallAssertionResults: [],
        toolBatchAssertionResults: [],
        verificationResults: [],
        timing: { durationMs: 5 },
        cost: {},
        rawOutputs: {},
      },
    ], {
      provider: "chatgpt",
      model: "gpt-5.4",
      suite: "scenario",
    });

    const markdown = renderSummaryMarkdown(summary);
    expect(markdown).toContain("- Duration: 5 ms");
    expect(markdown).toContain("## a");
  });
});

describe("renderScenarioReviewMarkdown", () => {
  it("renders stage review and human judgment details", () => {
    const markdown = renderScenarioReviewMarkdown({
      scenarioId: "runtime-core-docs-execute-design",
      description: "Design docs flow",
      targetRepo: "arkcompiler_runtime_core_docs",
      surface: "execute",
      taskShape: "readonly",
      taskType: "design",
      provider: "chatgpt",
      model: "gpt-5.4",
      status: "passed",
      startedAt: new Date(0).toISOString(),
      finishedAt: new Date(1).toISOString(),
      durationMs: 1,
      sourceRepoPath: "/src",
      isolationPath: "/tmp/workspace",
      outputDir: "/tmp/out",
      command: { executable: "bun", args: ["run"], exitCode: 0 },
      artifactInventory: [
        { path: "design-doc.md", category: "expected", exists: true, sizeBytes: 1200 },
        { path: "request.json", category: "request", exists: true, sizeBytes: 400 },
      ],
      artifactValidation: { passed: true, checks: [] },
      repoMutation: {
        expectedWorkspaceEffect: "non-mutating",
        passed: true,
        observedChanges: false,
        summary: "Readonly scenario kept the workspace clean.",
      },
      stageReview: {
        verdict: "pass",
        handoffReady: true,
        summary: "Artifacts look coherent enough for the next stage.",
        humanJudgment: "Review the artifacts directly before treating this as a golden packet.",
        checks: [
          {
            name: "primary-artifacts-non-empty",
            severity: "hard",
            passed: true,
            message: "Expected stage artifacts contain non-empty content.",
          },
        ],
        followUpIssues: [],
      },
      assertionResults: [],
      toolCallAssertionResults: [],
      toolBatchAssertionResults: [],
      verificationResults: [],
      timing: { durationMs: 1 },
      cost: {},
      rawOutputs: {
        requestPath: "/tmp/out/request.json",
        stdoutPath: "/tmp/out/stdout.txt",
      },
    });

    expect(markdown).toContain("## Artifact Review");
    expect(markdown).toContain("### Human Judgment");
    expect(markdown).toContain("design-doc.md");
  });
});

describe("runValidationScenario", () => {
  it("runs an execute scenario through a fake process and validates artifacts", async () => {
    const harnessRoot = await makeTempDir("devagent-live-harness-");
    const sourceRoot = await makeTempDir("devagent-live-execute-src-");
    await writeFile(join(sourceRoot, "README.md"), "# Source\n");
    await initGitRepo(sourceRoot);
    await commitAll(sourceRoot, "initial");

    const fakeCli = join(harnessRoot, "fake-devagent.mjs");
    await writeFile(fakeCli, `
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";

const args = process.argv.slice(2);
if (args[0] === "auth" && args[1] === "status") {
  process.stderr.write("Provider  Source  Key\\nchatgpt  oauth   tok***\\n");
  process.exit(0);
}
if (args[0] === "execute") {
  const requestPath = args[args.indexOf("--request") + 1];
  const request = JSON.parse(readFileSync(requestPath, "utf-8"));
  if (request.executor?.reasoning !== "low") {
    process.stderr.write("missing execute reasoning override\\n");
    process.exit(1);
  }
  const artifactDir = args[args.indexOf("--artifact-dir") + 1];
  mkdirSync(artifactDir, { recursive: true });
  writeFileSync(join(artifactDir, "triage-report.md"), "verifier and assembler impact\\n");
  writeFileSync(join(artifactDir, "result.json"), JSON.stringify({ status: "success", metrics: { durationMs: 10 } }, null, 2));
  writeFileSync(
    join(artifactDir, "engine-events.jsonl"),
    JSON.stringify({ type: "tool_call", tool: "delegate", callId: "call-1", batchId: "delegate-batch-1", batchSize: 2 }) + "\\n" +
    JSON.stringify({ type: "tool_call", tool: "delegate", callId: "call-2", batchId: "delegate-batch-1", batchSize: 2 }) + "\\n",
  );
  process.stdout.write(JSON.stringify({ type: "started" }) + "\\n");
  process.stdout.write(JSON.stringify({ type: "artifact" }) + "\\n");
  process.stdout.write(JSON.stringify({ type: "completed" }) + "\\n");
  process.exit(0);
}
process.stdout.write("cli success\\n");
process.stderr.write("[arkts] ArkTS linter enabled\\n");
process.exit(0);
`);
    chmodSync(fakeCli, 0o755);

    const scenario: ValidationScenario = validateScenarioManifest({
      id: "runtime-core-execute-triage",
      description: "Triage runtime core",
      suites: ["smoke"],
      targetRepo: "arkcompiler_runtime_core_docs",
      surface: "execute",
      taskShape: "readonly",
      isolationMode: "worktree",
      invocation: {
        type: "execute",
        taskType: "triage",
        workItemTitle: "Triage malformed ABC issue",
        summary: "Inspect verifier and assembler impact.",
        reasoning: "low",
      },
      expectedArtifacts: ["triage-report.md"],
      assertions: [
        { type: "contains", source: "artifact", path: "triage-report.md", value: "verifier" },
      ],
      verificationCommands: [],
      cleanupPolicy: "destroy",
      requiredToolCalls: [
        { tool: "delegate", minCalls: 1 },
      ],
      requiredToolBatches: [
        { tool: "delegate", minBatches: 1, minBatchSize: 2 },
      ],
    }, "inline");

    const outputRoot = await makeTempDir("devagent-live-output-");
    const report = await runValidationScenario(scenario, {
      devagentRoot: dirname(dirname(harnessRoot)),
      sourceRepoRoots: {
        arkcompiler_runtime_core_docs: sourceRoot,
      },
      provider: "chatgpt",
      model: "gpt-5.4",
      outputRoot,
      command: {
        executable: "node",
        baseArgs: [fakeCli],
      },
      authStatusOverride: { configuredProviders: ["chatgpt"] },
    });

    expect(report.status).toBe("passed");
    expect(report.artifactValidation.passed).toBe(true);
    expect(report.command.exitCode).toBe(0);
    expect(report.rawOutputs.requestPath).toContain("request.json");
    expect(report.artifactInventory.some((entry) => entry.path === "triage-report.md" && entry.category === "expected")).toBe(true);
    expect(report.repoMutation.passed).toBe(true);
    expect(report.stageReview?.verdict).toBe("pass");
    expect(report.observedToolCalls).toMatchObject({ delegate: 2 });
    expect(report.observedToolBatches).toMatchObject({ delegate: { batchCount: 1, maxBatchSize: 2 } });
    expect(report.toolCallAssertionResults).toEqual([
      expect.objectContaining({ tool: "delegate", minCalls: 1, observedCalls: 2, passed: true }),
    ]);
    expect(report.toolBatchAssertionResults).toEqual([
      expect.objectContaining({ tool: "delegate", minBatches: 1, minBatchSize: 2, observedBatches: 1, observedMaxBatchSize: 2, passed: true }),
    ]);
    const request = JSON.parse(await readFile(join(outputRoot, "runtime-core-execute-triage", "request.json"), "utf-8")) as { execution: { repositories: Array<{ isolation: string }> } };
    expect(request.execution.repositories[0]?.isolation).toBe("git-worktree");
  });

  it("supports strict structured execute stages with JSON and rendered artifacts", async () => {
    const harnessRoot = await makeTempDir("devagent-live-harness-");
    const sourceRoot = await makeTempDir("devagent-live-breakdown-src-");
    await writeFile(join(sourceRoot, "README.md"), "# Source\n");
    await initGitRepo(sourceRoot);
    await commitAll(sourceRoot, "initial");

    const fakeCli = join(harnessRoot, "fake-devagent-breakdown.mjs");
    await writeFile(fakeCli, `
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const args = process.argv.slice(2);
if (args[0] === "auth" && args[1] === "status") {
  process.stderr.write("Provider  Source  Key\\nchatgpt  oauth   tok***\\n");
  process.exit(0);
}
if (args[0] === "execute") {
  const requestPath = args[args.indexOf("--request") + 1];
  const request = JSON.parse(readFileSync(requestPath, "utf-8"));
  if (request.taskType !== "breakdown") {
    process.stderr.write("unexpected task type\\n");
    process.exit(1);
  }
  const artifactDir = args[args.indexOf("--artifact-dir") + 1];
  mkdirSync(artifactDir, { recursive: true });
  writeFileSync(join(artifactDir, "breakdown-doc.md"), "- [ ] B1. Tighten verifier docs\\n");
  writeFileSync(join(artifactDir, "breakdown-doc.json"), JSON.stringify({
    summary: "Ordered breakdown",
    executionOrder: ["B1"],
    tasks: [{
      id: "B1",
      title: "Tighten verifier docs",
      checklistLabel: "B1. Tighten verifier docs",
      objective: "Document verifier work",
      rationale: "Ground the docs pass",
      grounding: {
        designRefs: ["DesignDoc#Verifier"],
        repoPaths: ["docs/bc_verification"],
        codeSymbols: ["Verifier"]
      },
      dependencies: [],
      acceptanceCriteria: ["Docs updated"],
      expectedChanges: ["Refresh staged workflow docs"],
      validation: ["bun run check:oss"],
      riskNotes: ["Doc drift"],
      sizeBudget: {
        maxEstimatedChangedLines: 80,
        estimateReason: "Docs-only slice"
      }
    }]
  }, null, 2));
  writeFileSync(join(artifactDir, "result.json"), JSON.stringify({ status: "success", metrics: { durationMs: 10 } }, null, 2));
  process.stdout.write(JSON.stringify({ type: "started" }) + "\\n");
  process.stdout.write(JSON.stringify({ type: "completed" }) + "\\n");
  process.exit(0);
}
process.exit(1);
`);
    chmodSync(fakeCli, 0o755);

    const scenario: ValidationScenario = validateScenarioManifest({
      id: "runtime-core-docs-execute-breakdown",
      description: "Breakdown runtime core docs",
      suites: ["full"],
      targetRepo: "arkcompiler_runtime_core_docs",
      surface: "execute",
      taskShape: "readonly",
      isolationMode: "worktree",
      invocation: {
        type: "execute",
        taskType: "breakdown",
        workItemTitle: "Break down docs work",
        summary: "Produce breakdown",
      },
      expectedArtifacts: ["breakdown-doc.md", "breakdown-doc.json"],
      assertions: [
        { type: "contains", source: "artifact", path: "breakdown-doc.md", value: "- [ ] B1." },
        { type: "contains", source: "artifact", path: "breakdown-doc.json", value: "\"executionOrder\"" },
      ],
      verificationCommands: [],
      cleanupPolicy: "destroy",
    }, "inline");

    const outputRoot = await makeTempDir("devagent-live-output-");
    const report = await runValidationScenario(scenario, {
      devagentRoot: dirname(dirname(harnessRoot)),
      sourceRepoRoots: {
        arkcompiler_runtime_core_docs: sourceRoot,
      },
      provider: "chatgpt",
      model: "gpt-5.4",
      outputRoot,
      command: {
        executable: "node",
        baseArgs: [fakeCli],
      },
      authStatusOverride: { configuredProviders: ["chatgpt"] },
    });

    expect(report.status).toBe("passed");
    expect(report.artifactValidation.passed).toBe(true);
    expect(report.stageReview?.checks.some((check) => check.name === "structured-json-parse" && check.passed)).toBe(true);
    const request = JSON.parse(await readFile(join(outputRoot, "runtime-core-docs-execute-breakdown", "request.json"), "utf-8")) as { expectedArtifacts: string[] };
    expect(request.expectedArtifacts).toEqual(["breakdown-doc"]);
  });

  it("renders assertion templates before evaluating CLI repo diffs", async () => {
    const harnessRoot = await makeTempDir("devagent-live-harness-");
    const sourceRoot = await makeTempDir("devagent-live-cli-src-");
    await writeFile(join(sourceRoot, "README.md"), "# Source\n");
    await initGitRepo(sourceRoot);
    await commitAll(sourceRoot, "initial");

    const fakeCli = join(harnessRoot, "fake-devagent-cli.mjs");
    await writeFile(fakeCli, `
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";

const args = process.argv.slice(2);
if (args[0] === "auth" && args[1] === "status") {
  process.stderr.write("Provider  Source  Key\\nchatgpt  oauth   tok***\\n");
  process.exit(0);
}
const configText = readFileSync(join(process.env["HOME"], ".config", "devagent", "config.toml"), "utf-8");
const match = configText.match(/log_dir\\s*=\\s*\"([^\"]+)\"/);
if (match) {
  mkdirSync(match[1], { recursive: true });
  writeFileSync(join(match[1], "session.jsonl"), JSON.stringify({ event: "tool:after", data: { name: "delegate" } }) + "\\n");
}
const target = join(process.cwd(), "devagent_live_validation", "CounterFix.ets");
mkdirSync(dirname(target), { recursive: true });
writeFileSync(target, "let counter: number = 1;\\n");
process.stdout.write("cli success\\n");
process.stderr.write("[arkts] ArkTS linter enabled\\n");
process.exit(0);
`);
    chmodSync(fakeCli, 0o755);

    const scenario: ValidationScenario = validateScenarioManifest({
      id: "ets-frontend-cli-arkts",
      description: "CLI scenario",
      suites: ["smoke"],
      targetRepo: "arkcompiler_ets_frontend",
      surface: "cli",
      taskShape: "implement",
      isolationMode: "temp-copy",
      baselineAfterSetup: true,
      variables: {
        targetFile: "devagent_live_validation/CounterFix.ets",
      },
      preSetup: [
        {
          kind: "write-file",
          path: "${targetFile}",
          content: "let counter: any = 1;\\n",
        },
      ],
      invocation: {
        type: "cli",
        query: "Fix ${targetFile}",
        safetyMode: "autopilot",
      },
      expectedArtifacts: [],
      assertions: [
        { type: "contains", source: "stderr", value: "[arkts] ArkTS linter enabled" },
        { type: "contains", source: "repoDiff", value: "${targetFile}" },
      ],
      verificationCommands: [],
      cleanupPolicy: "destroy",
    }, "inline");

    const outputRoot = await makeTempDir("devagent-live-output-");
    const report = await runValidationScenario(scenario, {
      devagentRoot: dirname(dirname(harnessRoot)),
      sourceRepoRoots: {
        arkcompiler_ets_frontend: sourceRoot,
      },
      provider: "chatgpt",
      model: "gpt-5.4",
      outputRoot,
      command: {
        executable: "node",
        baseArgs: [fakeCli],
      },
      authStatusOverride: { configuredProviders: ["chatgpt"] },
    });

    expect(report.status).toBe("passed");
    expect(report.assertionResults.every((result) => result.passed)).toBe(true);
    expect(report.eventsSourcePath).toContain("cli-logs");
    expect(report.observedToolCalls).toMatchObject({ delegate: 1 });
  });

  it("supports CLI subcommand scenarios with expected failing exit codes", async () => {
    const harnessRoot = await makeTempDir("devagent-live-harness-");
    const sourceRoot = await makeTempDir("devagent-live-doctor-src-");
    await writeFile(join(sourceRoot, "README.md"), "# Source\n");
    await initGitRepo(sourceRoot);
    await commitAll(sourceRoot, "initial");

    const fakeCli = join(harnessRoot, "fake-devagent-doctor.mjs");
    await writeFile(fakeCli, `
const args = process.argv.slice(2);
if (args[0] === "auth" && args[1] === "status") {
  process.stderr.write("Provider  Source  Key\\n");
  process.exit(0);
}
if (args[0] === "doctor") {
  process.stdout.write([
    "devagent v0.1.0",
    "",
    "Blocking issues:",
    "",
    "  - Provider/model pairing: Configured model \\"cortex\\" is not registered for provider \\"openai\\". It is registered for \\"devagent-api\\". Switch provider or choose a model registered for \\"openai\\".",
    "",
    "What to do next:",
    "",
    "  Provider/model pairing:",
    "    Run now: devagent --provider devagent-api --model cortex \\"<your prompt>\\"",
    "    Set in ~/.config/devagent/config.toml:",
    "    provider = \\"devagent-api\\"",
    "    model = \\"cortex\\"",
    "    Export credentials: export DEVAGENT_API_KEY=ilg_...",
    "    Or store credentials: devagent auth login",
    "",
    "Effective config:",
    "",
    "  Provider: openai (config)",
    "  Model: cortex (config)",
    "  Credential: missing",
    "  Registered providers: devagent-api",
    "",
    "Checks:",
    "",
    "  ! Provider: openai: no API key (set OPENAI_API_KEY or run devagent auth login). Secondary until provider/model pairing is fixed.",
    "",
    "Blocking issues found.",
  ].join("\\n"));
  process.exit(1);
}
process.exit(1);
`);
    chmodSync(fakeCli, 0o755);

    const scenario: ValidationScenario = validateScenarioManifest({
      id: "doctor-provider-model-mismatch",
      description: "Doctor mismatch",
      suites: ["full"],
      targetRepo: "arkcompiler_runtime_core_docs",
      surface: "cli",
      taskShape: "readonly",
      isolationMode: "temp-copy",
      preSetup: [
        {
          kind: "write-file",
          path: "${homeDir}/.config/devagent/config.toml",
          content: 'provider = "openai"\\nmodel = "cortex"\\n',
        },
      ],
      invocation: {
        type: "cli-command",
        args: ["doctor"],
      },
      expectedExitCode: 1,
      expectedArtifacts: [],
      assertions: [
        { type: "contains", source: "stdout", value: "Blocking issues:" },
        { type: "contains", source: "stdout", value: 'Run now: devagent --provider devagent-api --model cortex "<your prompt>"' },
        { type: "contains", source: "stdout", value: 'provider = "devagent-api"' },
        { type: "contains", source: "stdout", value: 'model = "cortex"' },
        { type: "contains", source: "stdout", value: "export DEVAGENT_API_KEY=ilg_..." },
        { type: "contains", source: "stdout", value: "Credential: missing" },
        { type: "contains", source: "stdout", value: "Registered providers: devagent-api" },
        { type: "contains", source: "stdout", value: "! Provider: openai:" },
      ],
      verificationCommands: [],
      cleanupPolicy: "destroy",
      requiresAuth: false,
    }, "inline");

    const outputRoot = await makeTempDir("devagent-live-output-");
    const report = await runValidationScenario(scenario, {
      devagentRoot: dirname(dirname(harnessRoot)),
      sourceRepoRoots: {
        arkcompiler_runtime_core_docs: sourceRoot,
      },
      provider: "chatgpt",
      model: "gpt-5.4",
      outputRoot,
      command: {
        executable: "node",
        baseArgs: [fakeCli],
      },
      authStatusOverride: { configuredProviders: [] },
    });

    expect(report.status).toBe("passed");
    expect(report.command.exitCode).toBe(1);
    expect(report.assertionResults.every((result) => result.passed)).toBe(true);
  });

  it("treats prepared staged review diffs as baseline rather than unexpected mutations", async () => {
    const harnessRoot = await makeTempDir("devagent-live-harness-");
    const sourceRoot = await makeTempDir("devagent-live-review-src-");
    await writeFile(join(sourceRoot, "README.md"), "# Source\n");
    await mkdir(join(sourceRoot, "scripts"), { recursive: true });
    await initGitRepo(sourceRoot);
    await commitAll(sourceRoot, "initial");

    const fakeCli = join(harnessRoot, "fake-devagent-review.mjs");
    await writeFile(fakeCli, `
const args = process.argv.slice(2);
if (args[0] === "auth" && args[1] === "status") {
  process.stderr.write("Provider  Source  Key\\nchatgpt  oauth   tok***\\n");
  process.exit(0);
}
process.stdout.write("Critical: unsafe eval on untrusted input\\n");
process.exit(0);
`);
    chmodSync(fakeCli, 0o755);

    const scenario: ValidationScenario = validateScenarioManifest({
      id: "runtime-core-cli-review",
      description: "CLI review scenario",
      suites: ["full"],
      targetRepo: "arkcompiler_runtime_core",
      surface: "cli",
      taskShape: "review",
      isolationMode: "worktree",
      preSetup: [
        {
          kind: "write-file",
          path: "scripts/devagent_live_validation_review.sh",
          content: "#!/bin/sh\\n\\nuser_input=\\\"$1\\\"\\neval \\\"$user_input\\\"\\n",
          executable: true,
        },
        {
          kind: "run-command",
          cwd: "repo",
          command: "git add scripts/devagent_live_validation_review.sh",
        },
      ],
      invocation: {
        type: "cli",
        query: "Review the staged diff. Focus on correctness and security. Report concrete defects only.",
        safetyMode: "autopilot",
        maxIterations: 14,
      },
      expectedArtifacts: [],
      assertions: [
        {
          type: "matches",
          source: "stdout",
          pattern: "unsafe eval|eval",
        },
      ],
      verificationCommands: [],
      cleanupPolicy: "destroy",
      requiresAuth: true,
    }, "inline");

    const outputRoot = await makeTempDir("devagent-live-output-");
    const report = await runValidationScenario(scenario, {
      devagentRoot: dirname(dirname(harnessRoot)),
      sourceRepoRoots: {
        arkcompiler_runtime_core: sourceRoot,
      },
      provider: "chatgpt",
      model: "gpt-5.4",
      outputRoot,
      command: {
        executable: "node",
        baseArgs: [fakeCli],
      },
      authStatusOverride: { configuredProviders: ["chatgpt"] },
    });

    expect(report.status).toBe("passed");
    expect(report.repoMutation.passed).toBe(true);
    expect(report.repoMutation.summary).toContain("prepared baseline");
  });

  it("hydrates ArkTS linter assets into execute temp-copy workspaces", async () => {
    const harnessRoot = await makeTempDir("devagent-live-harness-");
    const sourceRoot = await makeTempDir("devagent-live-ets-src-");
    await writeFile(join(sourceRoot, ".gitignore"), "ets2panda/linter/dist/\nets2panda/linter/node_modules/\n");
    await writeFile(join(sourceRoot, "README.md"), "# Source\n");
    await mkdir(join(sourceRoot, "ets2panda", "linter", "dist"), { recursive: true });
    await mkdir(join(sourceRoot, "ets2panda", "linter", "node_modules"), { recursive: true });
    await writeFile(join(sourceRoot, "ets2panda", "linter", "dist", "tslinter.js"), "module.exports = {};\n");
    await writeFile(join(sourceRoot, "ets2panda", "linter", "node_modules", ".keep"), "ok\n");
    await initGitRepo(sourceRoot);
    await commitAll(sourceRoot, "initial");

    const fakeCli = join(harnessRoot, "fake-devagent-execute.mjs");
    await writeFile(fakeCli, `
import { mkdirSync, readFileSync, writeFileSync, lstatSync } from "node:fs";
import { dirname, join } from "node:path";

const args = process.argv.slice(2);
if (args[0] === "auth" && args[1] === "status") {
  process.stderr.write("Provider  Source  Key\\nchatgpt  oauth   tok***\\n");
  process.exit(0);
}
if (args[0] === "execute") {
  const requestPath = args[args.indexOf("--request") + 1];
  const artifactDir = args[args.indexOf("--artifact-dir") + 1];
  const request = JSON.parse(readFileSync(requestPath, "utf-8"));
  const repoRoot = request.repositories[0].repoRoot;
  const distPath = join(repoRoot, "ets2panda", "linter", "dist");
  const nodeModulesPath = join(repoRoot, "ets2panda", "linter", "node_modules");
  const distIsLink = lstatSync(distPath).isSymbolicLink();
  const nodeModulesIsLink = lstatSync(nodeModulesPath).isSymbolicLink();
  mkdirSync(artifactDir, { recursive: true });
  writeFileSync(join(artifactDir, "final-summary.md"), "validated ArkTS repair\\n");
  writeFileSync(join(artifactDir, "result.json"), JSON.stringify({ status: "success" }, null, 2));
  process.stdout.write(JSON.stringify({ type: "started" }) + "\\n");
  process.stdout.write(JSON.stringify({ type: "completed" }) + "\\n");
  if (!distIsLink || !nodeModulesIsLink) {
    process.stderr.write("expected ArkTS linter assets to be symlinked into workspace\\n");
    process.exit(1);
  }
  process.exit(0);
}
process.exit(1);
`);
    chmodSync(fakeCli, 0o755);

    const scenario: ValidationScenario = validateScenarioManifest({
      id: "ets-frontend-execute-repair",
      description: "Execute repair",
      suites: ["smoke"],
      targetRepo: "arkcompiler_ets_frontend",
      surface: "execute",
      taskShape: "repair",
      isolationMode: "temp-copy",
      requiresArktsLinter: true,
      invocation: {
        type: "execute",
        taskType: "repair",
        workItemTitle: "Repair ArkTS fixture",
        summary: "Fix the ArkTS fixture.",
      },
      expectedArtifacts: ["final-summary.md"],
      assertions: [
        { type: "contains", source: "artifact", path: "final-summary.md", value: "validated" },
      ],
      verificationCommands: [],
      cleanupPolicy: "destroy",
    }, "inline");

    const outputRoot = await makeTempDir("devagent-live-output-");
    const report = await runValidationScenario(scenario, {
      devagentRoot: dirname(dirname(harnessRoot)),
      sourceRepoRoots: {
        arkcompiler_ets_frontend: sourceRoot,
      },
      provider: "chatgpt",
      model: "gpt-5.4",
      outputRoot,
      command: {
        executable: "node",
        baseArgs: [fakeCli],
      },
      authStatusOverride: { configuredProviders: ["chatgpt"] },
    });

    expect(report.status).toBe("passed");
    expect(report.artifactValidation.passed).toBe(true);
  });

  it("fails with assertion classification when required delegate calls are missing", async () => {
    const harnessRoot = await makeTempDir("devagent-live-harness-");
    const sourceRoot = await makeTempDir("devagent-live-execute-missing-delegate-src-");
    await writeFile(join(sourceRoot, "README.md"), "# Source\n");
    await initGitRepo(sourceRoot);
    await commitAll(sourceRoot, "initial");

    const fakeCli = join(harnessRoot, "fake-devagent-no-delegate.mjs");
    await writeFile(fakeCli, `
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
const args = process.argv.slice(2);
if (args[0] === "auth" && args[1] === "status") {
  process.stderr.write("Provider  Source  Key\\nchatgpt  oauth   tok***\\n");
  process.exit(0);
}
if (args[0] === "execute") {
  const artifactDir = args[args.indexOf("--artifact-dir") + 1];
  mkdirSync(artifactDir, { recursive: true });
  writeFileSync(join(artifactDir, "triage-report.md"), "verifier only\\n");
  writeFileSync(join(artifactDir, "result.json"), JSON.stringify({ status: "success" }, null, 2));
  writeFileSync(join(artifactDir, "engine-events.jsonl"), JSON.stringify({ type: "tool_call", tool: "read_file" }) + "\\n");
  process.stdout.write(JSON.stringify({ type: "started" }) + "\\n");
  process.stdout.write(JSON.stringify({ type: "completed" }) + "\\n");
  process.exit(0);
}
process.exit(1);
`);
    chmodSync(fakeCli, 0o755);

    const scenario = validateScenarioManifest({
      id: "runtime-core-execute-triage",
      description: "Missing delegate",
      suites: ["smoke"],
      targetRepo: "arkcompiler_runtime_core_docs",
      surface: "execute",
      taskShape: "readonly",
      isolationMode: "temp-copy",
      invocation: {
        type: "execute",
        taskType: "triage",
        workItemTitle: "Triage",
        summary: "Inspect verifier impact.",
      },
      expectedArtifacts: ["triage-report.md"],
      assertions: [
        { type: "contains", source: "artifact", path: "triage-report.md", value: "verifier" },
      ],
      verificationCommands: [],
      cleanupPolicy: "destroy",
      requiredToolCalls: [{ tool: "delegate", minCalls: 1 }],
    }, "inline");

    const outputRoot = await makeTempDir("devagent-live-output-");
    const report = await runValidationScenario(scenario, {
      devagentRoot: dirname(dirname(harnessRoot)),
      sourceRepoRoots: {
        arkcompiler_runtime_core_docs: sourceRoot,
      },
      provider: "chatgpt",
      model: "gpt-5.4",
      outputRoot,
      command: {
        executable: "node",
        baseArgs: [fakeCli],
      },
      authStatusOverride: { configuredProviders: ["chatgpt"] },
    });

    expect(report.status).toBe("failed");
    expect(report.failureClass).toBe("assertion");
    expect(report.toolCallAssertionResults).toEqual([
      expect.objectContaining({ tool: "delegate", passed: false, observedCalls: 0 }),
    ]);
  });

  it("fails with assertion classification when delegate batching is missing", async () => {
    const harnessRoot = await makeTempDir("devagent-live-harness-");
    const sourceRoot = await makeTempDir("devagent-live-execute-missing-batch-src-");
    await writeFile(join(sourceRoot, "README.md"), "# Source\n");
    await initGitRepo(sourceRoot);
    await commitAll(sourceRoot, "initial");

    const fakeCli = join(harnessRoot, "fake-devagent-no-batch.mjs");
    await writeFile(fakeCli, `
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
const args = process.argv.slice(2);
if (args[0] === "auth" && args[1] === "status") {
  process.stderr.write("Provider  Source  Key\\nchatgpt  oauth   tok***\\n");
  process.exit(0);
}
if (args[0] === "execute") {
  const artifactDir = args[args.indexOf("--artifact-dir") + 1];
  mkdirSync(artifactDir, { recursive: true });
  writeFileSync(join(artifactDir, "triage-report.md"), "verifier and assembler\\n");
  writeFileSync(join(artifactDir, "result.json"), JSON.stringify({ status: "success" }, null, 2));
  writeFileSync(
    join(artifactDir, "engine-events.jsonl"),
    JSON.stringify({ type: "tool_call", tool: "delegate", callId: "call-1", batchId: "delegate-batch-1", batchSize: 1 }) + "\\n" +
    JSON.stringify({ type: "tool_call", tool: "delegate", callId: "call-2", batchId: "delegate-batch-2", batchSize: 1 }) + "\\n",
  );
  process.stdout.write(JSON.stringify({ type: "started" }) + "\\n");
  process.stdout.write(JSON.stringify({ type: "completed" }) + "\\n");
  process.exit(0);
}
process.exit(1);
`);
    chmodSync(fakeCli, 0o755);

    const scenario = validateScenarioManifest({
      id: "runtime-core-execute-triage",
      description: "Missing delegate batch",
      suites: ["smoke"],
      targetRepo: "arkcompiler_runtime_core_docs",
      surface: "execute",
      taskShape: "readonly",
      isolationMode: "temp-copy",
      invocation: {
        type: "execute",
        taskType: "triage",
        workItemTitle: "Triage",
        summary: "Inspect verifier impact.",
      },
      expectedArtifacts: ["triage-report.md"],
      assertions: [
        { type: "contains", source: "artifact", path: "triage-report.md", value: "verifier" },
      ],
      verificationCommands: [],
      cleanupPolicy: "destroy",
      requiredToolCalls: [{ tool: "delegate", minCalls: 1 }],
      requiredToolBatches: [{ tool: "delegate", minBatches: 1, minBatchSize: 2 }],
    }, "inline");

    const outputRoot = await makeTempDir("devagent-live-output-");
    const report = await runValidationScenario(scenario, {
      devagentRoot: dirname(dirname(harnessRoot)),
      sourceRepoRoots: {
        arkcompiler_runtime_core_docs: sourceRoot,
      },
      provider: "chatgpt",
      model: "gpt-5.4",
      outputRoot,
      command: {
        executable: "node",
        baseArgs: [fakeCli],
      },
      authStatusOverride: { configuredProviders: ["chatgpt"] },
    });

    expect(report.status).toBe("failed");
    expect(report.failureClass).toBe("assertion");
    expect(report.toolBatchAssertionResults).toEqual([
      expect.objectContaining({ tool: "delegate", passed: false, observedBatches: 2, observedMaxBatchSize: 1 }),
    ]);
  });

  it("prefers batch-aware execute tool_call entries over legacy assistant grouping", async () => {
    const harnessRoot = await makeTempDir("devagent-live-harness-");
    const sourceRoot = await makeTempDir("devagent-live-execute-mixed-batch-src-");
    await writeFile(join(sourceRoot, "README.md"), "# Source\n");
    await initGitRepo(sourceRoot);
    await commitAll(sourceRoot, "initial");

    const fakeCli = join(harnessRoot, "fake-devagent-mixed-batch.mjs");
    await writeFile(fakeCli, `
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
const args = process.argv.slice(2);
if (args[0] === "auth" && args[1] === "status") {
  process.stderr.write("Provider  Source  Key\\nchatgpt  oauth   tok***\\n");
  process.exit(0);
}
if (args[0] === "execute") {
  const artifactDir = args[args.indexOf("--artifact-dir") + 1];
  mkdirSync(artifactDir, { recursive: true });
  writeFileSync(join(artifactDir, "triage-report.md"), "verifier and assembler\\n");
  writeFileSync(join(artifactDir, "result.json"), JSON.stringify({ status: "success" }, null, 2));
  writeFileSync(
    join(artifactDir, "engine-events.jsonl"),
    JSON.stringify({ event: "message:assistant", data: { toolCalls: [{ name: "delegate" }] } }) + "\\n" +
    JSON.stringify({ type: "tool_call", tool: "delegate", callId: "call-1", batchId: "delegate-batch-1", batchSize: 2 }) + "\\n" +
    JSON.stringify({ type: "tool_call", tool: "delegate", callId: "call-2", batchId: "delegate-batch-1", batchSize: 2 }) + "\\n",
  );
  process.stdout.write(JSON.stringify({ type: "started" }) + "\\n");
  process.stdout.write(JSON.stringify({ type: "completed" }) + "\\n");
  process.exit(0);
}
process.exit(1);
`);
    chmodSync(fakeCli, 0o755);

    const scenario = validateScenarioManifest({
      id: "runtime-core-execute-triage",
      description: "Mixed execute batch observation",
      suites: ["smoke"],
      targetRepo: "arkcompiler_runtime_core_docs",
      surface: "execute",
      taskShape: "readonly",
      isolationMode: "temp-copy",
      invocation: {
        type: "execute",
        taskType: "triage",
        workItemTitle: "Triage",
        summary: "Inspect verifier impact.",
      },
      expectedArtifacts: ["triage-report.md"],
      assertions: [
        { type: "contains", source: "artifact", path: "triage-report.md", value: "verifier" },
      ],
      verificationCommands: [],
      cleanupPolicy: "destroy",
      requiredToolCalls: [{ tool: "delegate", minCalls: 2 }],
      requiredToolBatches: [{ tool: "delegate", minBatches: 1, minBatchSize: 2 }],
    }, "inline");

    const outputRoot = await makeTempDir("devagent-live-output-");
    const report = await runValidationScenario(scenario, {
      devagentRoot: dirname(dirname(harnessRoot)),
      sourceRepoRoots: {
        arkcompiler_runtime_core_docs: sourceRoot,
      },
      provider: "chatgpt",
      model: "gpt-5.4",
      outputRoot,
      command: {
        executable: "node",
        baseArgs: [fakeCli],
      },
      authStatusOverride: { configuredProviders: ["chatgpt"] },
    });

    expect(report.status).toBe("passed");
    expect(report.observedToolCalls).toMatchObject({ delegate: 2 });
    expect(report.observedToolBatches).toMatchObject({ delegate: { batchCount: 1, maxBatchSize: 2 } });
  });

  it("writes a setup failure report when pre-setup throws", async () => {
    const harnessRoot = await makeTempDir("devagent-live-harness-");
    const sourceRoot = await makeTempDir("devagent-live-setup-failure-src-");
    await writeFile(join(sourceRoot, "README.md"), "# Source\n");
    await initGitRepo(sourceRoot);
    await commitAll(sourceRoot, "initial");

    const fakeCli = join(harnessRoot, "fake-devagent-auth.mjs");
    await writeFile(fakeCli, `
const args = process.argv.slice(2);
if (args[0] === "auth" && args[1] === "status") {
  process.stderr.write("Provider  Source  Key\\nchatgpt  oauth   tok***\\n");
  process.exit(0);
}
process.exit(1);
`);
    chmodSync(fakeCli, 0o755);

    const scenario = validateScenarioManifest({
      id: "setup-failure",
      description: "Setup failure",
      suites: ["smoke"],
      targetRepo: "arkcompiler_runtime_core",
      surface: "cli",
      taskShape: "review",
      isolationMode: "temp-copy",
      preSetup: [
        { kind: "run-command", command: "exit 7" },
      ],
      invocation: {
        type: "cli",
        query: "Review the repo",
        safetyMode: "autopilot",
      },
      expectedArtifacts: [],
      assertions: [],
      verificationCommands: [],
      cleanupPolicy: "destroy",
    }, "inline");

    const outputRoot = await makeTempDir("devagent-live-output-");
    const report = await runValidationScenario(scenario, {
      devagentRoot: dirname(dirname(harnessRoot)),
      sourceRepoRoots: {
        arkcompiler_runtime_core: sourceRoot,
      },
      provider: "chatgpt",
      model: "gpt-5.4",
      outputRoot,
      command: {
        executable: "node",
        baseArgs: [fakeCli],
      },
      authStatusOverride: { configuredProviders: ["chatgpt"] },
    });

    expect(report.status).toBe("failed");
    expect(report.failureClass).toBe("setup");
    expect(existsSync(join(outputRoot, "setup-failure", "report.json"))).toBe(true);
  });

  it("fails fast with provider classification when a required provider auth is missing", async () => {
    const sourceRoot = await makeTempDir("devagent-live-provider-block-src-");
    await writeFile(join(sourceRoot, "README.md"), "# Source\n");
    await initGitRepo(sourceRoot);
    await commitAll(sourceRoot, "initial");

    const scenario = validateScenarioManifest({
      id: "provider-auth-missing",
      description: "Provider auth missing",
      suites: ["smoke"],
      targetRepo: "arkcompiler_runtime_core",
      surface: "cli",
      taskShape: "readonly",
      isolationMode: "temp-copy",
      invocation: {
        type: "cli",
        query: "Hello",
        safetyMode: "default",
      },
      expectedArtifacts: [],
      assertions: [],
      verificationCommands: [],
      cleanupPolicy: "destroy",
      requiresAuth: true,
      requiredProvider: "openai",
    }, "inline");

    const outputRoot = await makeTempDir("devagent-live-output-");
    const report = await runValidationScenario(scenario, {
      devagentRoot: dirname(sourceRoot),
      sourceRepoRoots: {
        arkcompiler_runtime_core: sourceRoot,
      },
      provider: "chatgpt",
      model: "gpt-5.4",
      outputRoot,
      authStatusOverride: { configuredProviders: ["chatgpt"] },
    });

    expect(report.status).toBe("blocked");
    expect(report.failureClass).toBe("provider");
    expect(report.failureMessage).toContain("openai auth is not configured");
  });

  it("fails fast with provider classification when required provider auth is expired", async () => {
    const sourceRoot = await makeTempDir("devagent-live-provider-expired-src-");
    await writeFile(join(sourceRoot, "README.md"), "# Source\n");
    await initGitRepo(sourceRoot);
    await commitAll(sourceRoot, "initial");

    const scenario = validateScenarioManifest({
      id: "provider-auth-expired",
      description: "Provider auth expired",
      suites: ["smoke"],
      targetRepo: "arkcompiler_runtime_core",
      surface: "cli",
      taskShape: "readonly",
      isolationMode: "temp-copy",
      invocation: {
        type: "cli",
        query: "Hello",
        safetyMode: "default",
      },
      expectedArtifacts: [],
      assertions: [],
      verificationCommands: [],
      cleanupPolicy: "destroy",
      requiresAuth: true,
      requiredProvider: "chatgpt",
    }, "inline");

    const outputRoot = await makeTempDir("devagent-live-output-");
    const report = await runValidationScenario(scenario, {
      devagentRoot: dirname(sourceRoot),
      sourceRepoRoots: {
        arkcompiler_runtime_core: sourceRoot,
      },
      provider: "chatgpt",
      model: "gpt-5.4",
      outputRoot,
      authStatusOverride: {
        configuredProviders: ["chatgpt"],
        expiredProviders: ["chatgpt"],
      },
    });

    expect(report.status).toBe("blocked");
    expect(report.failureClass).toBe("provider");
    expect(report.failureMessage).toContain("chatgpt auth is expired");
  });
});

describe("classifyProviderFailure", () => {
  it("classifies invalid credentials as blocked", () => {
    expect(classifyProviderFailure({
      exitCode: 1,
      stdout: "",
      stderr: "statusCode: 401\nurl: 'https://api.deepseek.com/v1/chat/completions'\nAuthentication Fails, Your api key: ****1234 is invalid\n",
      timedOut: false,
      durationMs: 100,
    })).toEqual({
      status: "blocked",
      blockedReason: "invalid stored credential",
    });
  });

  it("classifies upstream 5xx outages as blocked service failures", () => {
    expect(classifyProviderFailure({
      exitCode: 1,
      stdout: "",
      stderr: "statusCode: 503\nurl: 'https://internal-llm-gateway.example/v1/chat/completions'\nService Temporarily Unavailable\n",
      timedOut: false,
      durationMs: 100,
    })).toEqual({
      status: "blocked",
      blockedReason: "provider service unavailable (503) at https://internal-llm-gateway.example/v1/chat/completions",
    });
  });
});

describe("selectPreferredOllamaModel", () => {
  it("picks qwen3.5:9b when it is installed locally", () => {
    expect(selectPreferredOllamaModel([
      "NAME                    ID              SIZE      MODIFIED",
      "qwen3.5:9b              abc             6.6 GB    43 minutes ago",
      "glm-4.7-flash:latest    def             19 GB     7 weeks ago",
    ].join("\n"))).toEqual({
      model: "qwen3.5:9b",
    });
  });

  it("blocks Ollama smoke when qwen3.5:9b is unavailable", () => {
    expect(selectPreferredOllamaModel([
      "NAME                    ID              SIZE      MODIFIED",
      "glm-4.7-flash:latest    def             19 GB     7 weeks ago",
      "qwen3-coder:30b         ghi             18 GB     7 weeks ago",
    ].join("\n"))).toEqual({
      model: null,
      blockedReason: 'Required Ollama model "qwen3.5:9b" is not installed locally.',
    });
  });
});
