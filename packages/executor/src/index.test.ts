import { chmod, mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { afterEach, describe, expect, it } from "vitest";
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
  resolveVerifyNodeBinary,
  rewriteVerifyCommand,
  resolveRequestedSkills,
  validateExecutionCapabilities,
} from "./index.js";

function createStrictStageResponse(taskType: "breakdown" | "issue-generation"): string {
  if (taskType === "breakdown") {
    return JSON.stringify({
      structured: {
        summary: "Ordered delivery breakdown",
        executionOrder: ["B1", "B2"],
        tasks: [
          {
            id: "B1",
            title: "Normalize input handling",
            checklistLabel: "B1. Normalize input handling in src/cli.ts",
            objective: "Reject blank names before rendering output.",
            rationale: "Acceptance criteria require blank names to fail fast.",
            grounding: {
              designRefs: ["DesignDoc#InputValidation"],
              repoPaths: ["src/cli.ts", "src/cli.test.ts"],
              codeSymbols: ["parseName"],
            },
            dependencies: [],
            acceptanceCriteria: ["Blank names are rejected."],
            expectedChanges: ["Update parser.", "Add regression test."],
            validation: ["bun test"],
            riskNotes: [],
            sizeBudget: {
              maxEstimatedChangedLines: 120,
              estimateReason: "Parser and test-only slice.",
            },
          },
          {
            id: "B2",
            title: "Add JSON output",
            checklistLabel: "B2. Add JSON output in src/render.ts",
            objective: "Expose a machine-readable greeting format.",
            rationale: "Acceptance criteria require JSON mode.",
            grounding: {
              designRefs: ["DesignDoc#OutputContract"],
              repoPaths: ["src/render.ts", "README.md"],
              codeSymbols: ["renderGreeting"],
            },
            dependencies: ["B1"],
            acceptanceCriteria: ["JSON output includes message and template."],
            expectedChanges: ["Add renderer branch.", "Document JSON mode."],
            validation: ["bun test", "bun run build"],
            riskNotes: ["Keep text mode stable."],
            sizeBudget: {
              maxEstimatedChangedLines: 180,
              estimateReason: "Renderer, tests, and docs remain small.",
            },
          },
        ],
      },
      rendered: [
        "- [ ] B1. Normalize input handling in src/cli.ts",
        "- [ ] B2. Add JSON output in src/render.ts",
      ].join("\n"),
    });
  }

  return JSON.stringify({
    structured: {
      summary: "Executable issue queue",
      issues: [
        {
          id: "I1",
          title: "Normalize names before rendering",
          problemStatement: "Blank names currently pass through the CLI.",
          rationale: "This completes breakdown task B1 first.",
          scope: ["Trim whitespace", "Reject blank names"],
          acceptanceCriteria: ["Blank names throw a validation error."],
          dependencies: [],
          linkedDesignSections: ["DesignDoc#InputValidation"],
          linkedBreakdownTaskIds: ["B1"],
          grounding: {
            repoPaths: ["src/cli.ts", "src/cli.test.ts"],
            codeSymbols: ["parseName"],
          },
          requiredTests: ["Add blank-input coverage."],
          outOfScope: ["JSON rendering"],
          implementationNotes: ["Do not touch renderer in this issue."],
        },
        {
          id: "I2",
          title: "Add JSON greeting output",
          problemStatement: "The CLI only emits text output today.",
          rationale: "This follows breakdown task B2 after validation is in place.",
          scope: ["Add JSON render path", "Document JSON usage"],
          acceptanceCriteria: ["JSON output includes message and template."],
          dependencies: ["I1"],
          linkedDesignSections: ["DesignDoc#OutputContract"],
          linkedBreakdownTaskIds: ["B2"],
          grounding: {
            repoPaths: ["src/render.ts", "README.md"],
            codeSymbols: ["renderGreeting"],
          },
          requiredTests: ["Add JSON mode coverage."],
          outOfScope: ["New templates"],
          implementationNotes: ["Preserve current text output shape."],
        },
      ],
    },
    rendered: [
      "# Issue Queue",
      "",
      "- I1: Normalize names before rendering",
      "- I2: Add JSON greeting output",
    ].join("\n"),
  });
}

const envKeysToReset = [
  "DEVAGENT_EXECUTOR_FAKE_RESPONSE",
  "DEVAGENT_EXECUTOR_FAKE_RESPONSE_PLAN",
  "DEVAGENT_EXECUTOR_FAKE_RESPONSE_TASK_INTAKE",
  "DEVAGENT_EXECUTOR_FAKE_RESPONSE_ISSUE_GENERATION",
] as const;

const originalEnv = new Map<string, string | undefined>(
  envKeysToReset.map((key) => [key, process.env[key]]),
);

afterEach(() => {
  for (const key of envKeysToReset) {
    const value = originalEnv.get(key);
    if (value === undefined) {
      delete process.env[key];
    } else {
      process.env[key] = value;
    }
  }
});

function createRequest(taskType: TaskExecutionRequest["taskType"]): TaskExecutionRequest {
  const workspaceId = "workspace-1";
  const repositoryId = "repo-1";
  return {
    protocolVersion: PROTOCOL_VERSION,
    taskId: `task-${taskType}`,
    taskType,
    workspaceRef: {
      id: workspaceId,
      name: "Executor Workspace",
      provider: "github",
      primaryRepositoryId: repositoryId,
    },
    repositories: [{
      id: repositoryId,
      workspaceId,
      alias: "primary",
      name: "repo",
      repoRoot: "/tmp/repo",
      repoFullName: "example/repo",
      defaultBranch: "main",
      provider: "github",
    }],
    workItem: {
      id: "item-42",
      kind: "github-issue",
      externalId: "42",
      title: "Refactor runner",
      repositoryId,
    },
    execution: {
      primaryRepositoryId: repositoryId,
      repositories: [{
        repositoryId,
        alias: "primary",
        sourceRepoPath: "/tmp/repo",
        workBranch: `devagent/${taskType}/task-${taskType}`,
        isolation: "temp-copy",
      }],
    },
    targetRepositoryIds: [repositoryId],
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
    capabilities: {
      canSyncTasks: true,
      canCreateTask: true,
      canComment: true,
      canReview: true,
      canMerge: true,
      canOpenReviewable: true,
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
    const skillDir = join(repoRoot, ".agents", "skills", "testing");
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

  it("skips missing requested skills instead of failing resolution", async () => {
    const repoRoot = await createRepoRoot();

    const warnings: string[] = [];
    const resolved = await resolveRequestedSkills(repoRoot, ["testing"], "session-1", (message) => {
      warnings.push(message);
    });

    expect(resolved).toEqual([]);
    expect(warnings).toHaveLength(1);
    expect(warnings[0]).toMatch(/Requested skill "testing" could not be loaded and will be skipped/i);

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
    expect(query).toContain("Workspace: Executor Workspace");
    expect(query).toContain("Target repositories: primary");
  });

  it("includes imported reviewable context in the task query", () => {
    const request = createRequest("review");
    request.reviewable = {
      id: "review-1",
      provider: "github",
      type: "github-pr",
      externalId: "123",
      title: "Stabilize multi-repo runner",
      url: "https://github.com/example/repo/pull/123",
      repositoryId: "repo-1",
    };

    const query = buildTaskQuery(request);

    expect(query).toContain("Review target: github-pr 123");
    expect(query).toContain("Review title: Stabilize multi-repo runner");
  });

  it("marks triage as analysis-only and forbids verification commands by default", () => {
    const request = createRequest("triage");
    const query = buildTaskQuery(request);

    expect(query).toContain("Workspace is analysis-only for triage. No file changes are allowed.");
    expect(query).toContain("Do not run project verification commands unless the request explicitly requires them.");
  });

  it("marks plan as planning-only and forbids verification commands by default", () => {
    const request = createRequest("plan");
    const query = buildTaskQuery(request);

    expect(query).toContain("Workspace is planning-only for plan. No file changes are allowed.");
    expect(query).toContain("Do not run project verification commands unless the request explicitly requires them.");
  });

  it("marks task-intake as analysis-only and forbids verification commands by default", () => {
    const request = createRequest("task-intake");
    const query = buildTaskQuery(request);

    expect(query).toContain("Workspace is analysis-only for task intake. No file changes are allowed.");
    expect(query).toContain("Do not run project verification commands unless the request explicitly requires them.");
  });

  it("marks design as non-mutating and forbids verification commands by default", () => {
    const request = createRequest("design");
    const query = buildTaskQuery(request);

    expect(query).toContain("Workspace is design-only for this stage. No file changes are allowed.");
    expect(query).toContain("Do not run project verification commands unless the request explicitly requires them.");
    expect(query).toContain("Do not use update_plan for this stage.");
  });

  it("forces breakdown into a strict grounded checklist contract", () => {
    const query = buildTaskQuery(createRequest("breakdown"));

    expect(query).toContain("Do not use update_plan for this stage.");
    expect(query).toContain("ordered checklist of small executable tasks");
    expect(query).toContain("fewer than 500 changed lines");
    expect(query).toContain("concrete repo paths or symbols");
    expect(query).toContain("\"summary\": \"short summary\"");
    expect(query).toContain("\"executionOrder\": [\"B1\", \"B2\"]");
    expect(query).toContain("\"checklistLabel\": \"B1. concrete checklist item\"");
    expect(query).toContain("Do not rename keys, omit required fields, or add extra fields.");
    expect(query).toContain("\"structured\": <BreakdownDoc>");
    expect(query).toContain("Return only the JSON object");
  });

  it("forces issue generation to derive from approved breakdown tasks", () => {
    const query = buildTaskQuery(createRequest("issue-generation"));

    expect(query).toContain("Do not use update_plan for this stage.");
    expect(query).toContain("Generate executable issue specs directly from the approved breakdown tasks.");
    expect(query).toContain("Do not infer issues from document headings.");
    expect(query).toContain("link to one or more approved breakdown task ids");
    expect(query).toContain("\"issues\": [");
    expect(query).toContain("\"linkedBreakdownTaskIds\": [\"B1\"]");
    expect(query).toContain("\"implementationNotes\": [\"implementation note\"]");
    expect(query).toContain("Do not rename keys, omit required fields, or add extra fields.");
    expect(query).toContain("\"structured\": <IssueSpecDoc>");
  });

  it("includes issue unit and context bundle metadata when provided", () => {
    const request = createRequest("plan");
    request.issueUnit = {
      id: "issue-unit-1",
      title: "Add JSON mode",
      sequence: 1,
      dependencyIds: [],
      acceptanceCriteria: ["Supports --json output"],
      linkedArtifactVersionIds: ["artifact-version-1"],
    };
    request.contextBundle = {
      id: "bundle-1",
      artifactVersionIds: ["artifact-version-1", "artifact-version-2"],
      summary: "Approved design and breakdown context",
    };

    const query = buildTaskQuery(request);
    expect(query).toContain("Issue unit: [1] Add JSON mode");
    expect(query).toContain("Context bundle: bundle-1");
    expect(query).toContain("artifact-version-1");
  });

  it("promotes known workflow comments into dedicated sections and preserves unknown comments", () => {
    const request = createRequest("implement");
    request.context.comments = [
      {
        author: "issue-spec-artifact",
        body: "Approved issue spec artifact:\n\n# Issue Spec\n\nKeep the edit small.",
      },
      {
        author: "review-report",
        body: "Review report artifact:\n\nNo defects found.",
      },
      {
        author: "teammate",
        body: "Please keep the README wording consistent with the issue text.",
      },
    ];

    const query = buildTaskQuery(request);

    expect(query).toContain("Approved issue spec artifact:\n# Issue Spec");
    expect(query).toContain("Review report artifact:\nNo defects found.");
    expect(query).toContain("Comments:\n- teammate: Please keep the README wording consistent with the issue text.");
    expect(query).not.toContain("- issue-spec-artifact:");
    expect(query).not.toContain("- review-report:");
  });

  it("prioritizes design context ahead of generic context for breakdown", () => {
    const request = createRequest("breakdown");
    request.context.summary = "Generic summary comes later.";
    request.context.comments = [{
      author: "design-artifact",
      body: "Approved design artifact:\n\n# Design\n\nUse README.md only.",
    }];

    const query = buildTaskQuery(request);

    expect(query.indexOf("Approved design artifact:")).toBeLessThan(query.indexOf("Summary:\nGeneric summary comes later."));
  });

  it("orders workflow-chain context for implement", () => {
    const request = createRequest("implement");
    request.context.summary = "Implement the approved issue.";
    request.issueUnit = {
      id: "issue-unit-1",
      title: "Document validation flow",
      sequence: 1,
      dependencyIds: [],
      acceptanceCriteria: ["README explains the validation workflow."],
      linkedArtifactVersionIds: ["artifact-1"],
    };
    request.context.changedFilesHint = ["README.md", "README.md", "docs/flow.md"];
    request.context.comments = [
      {
        author: "issue-spec-artifact",
        body: "Approved issue spec artifact:\n\n# Issue Spec\n\nUse README.md only.",
      },
      {
        author: "breakdown-artifact",
        body: "Approved breakdown artifact:\n\n# Breakdown\n\nOne docs task.",
      },
    ];

    const query = buildTaskQuery(request);

    expect(query.indexOf("Approved issue spec artifact:")).toBeLessThan(query.indexOf("Issue unit details:"));
    expect(query.indexOf("Issue unit details:")).toBeLessThan(query.indexOf("Focus files:"));
    expect(query.indexOf("Focus files:")).toBeLessThan(query.indexOf("Summary:\nImplement the approved issue."));
    expect(query.match(/- README\.md/g)).toHaveLength(1);
    expect(query).toContain("- docs/flow.md");
  });

  it("orders workflow-chain context for review and repair", () => {
    const reviewRequest = createRequest("review");
    reviewRequest.context.summary = "Review the approved change.";
    reviewRequest.issueUnit = {
      id: "issue-unit-1",
      title: "Document validation flow",
      sequence: 1,
      dependencyIds: [],
      acceptanceCriteria: ["README explains the validation workflow."],
      linkedArtifactVersionIds: ["artifact-1"],
    };
    reviewRequest.context.changedFilesHint = ["README.md"];
    reviewRequest.context.comments = [
      {
        author: "issue-spec-artifact",
        body: "Approved issue spec artifact:\n\n# Issue Spec\n\nUse README.md only.",
      },
      {
        author: "implementation-summary",
        body: "Implementation summary artifact:\n\nUpdated README.md.",
      },
    ];

    const reviewQuery = buildTaskQuery(reviewRequest);
    expect(reviewQuery.indexOf("Approved issue spec artifact:")).toBeLessThan(reviewQuery.indexOf("Implementation summary artifact:"));
    expect(reviewQuery.indexOf("Implementation summary artifact:")).toBeLessThan(reviewQuery.indexOf("Issue unit details:"));
    expect(reviewQuery.indexOf("Issue unit details:")).toBeLessThan(reviewQuery.indexOf("Focus files:"));

    const repairRequest = createRequest("repair");
    repairRequest.context.summary = "Repair concrete review findings.";
    repairRequest.issueUnit = reviewRequest.issueUnit;
    repairRequest.context.changedFilesHint = ["README.md"];
    repairRequest.context.comments = [
      {
        author: "issue-spec-artifact",
        body: "Approved issue spec artifact:\n\n# Issue Spec\n\nUse README.md only.",
      },
      {
        author: "implementation-summary",
        body: "Implementation summary artifact:\n\nUpdated README.md.",
      },
      {
        author: "review-report",
        body: "Review report artifact:\n\nSeverity: medium\nClarify the workflow wording.",
      },
    ];

    const repairQuery = buildTaskQuery(repairRequest);
    expect(repairQuery.indexOf("Review report artifact:")).toBeLessThan(repairQuery.indexOf("Implementation summary artifact:"));
    expect(repairQuery.indexOf("Implementation summary artifact:")).toBeLessThan(repairQuery.indexOf("Approved issue spec artifact:"));
    expect(repairQuery.indexOf("Approved issue spec artifact:")).toBeLessThan(repairQuery.indexOf("Issue unit details:"));
    expect(repairQuery.indexOf("Issue unit details:")).toBeLessThan(repairQuery.indexOf("Focus files:"));
  });

  it("truncates promoted workflow artifacts deterministically", () => {
    const request = createRequest("repair");
    const longBody = `${"a".repeat(4_200)}\nsecond line`;
    request.context.comments = [{
      author: "review-report",
      body: `Review report artifact:\n\n${longBody}`,
    }];

    const query = buildTaskQuery(request);

    expect(query).toContain("[workflow context truncated at 4000 chars]");
    expect(query).toContain("Review report artifact:");
  });

  it("emits a warning log and continues when a requested skill is missing", async () => {
    const repoRoot = await createRepoRoot();
    const artifactDir = join(repoRoot, "artifacts");
    await mkdir(artifactDir, { recursive: true });

    const events: TaskExecutionEvent[] = [];
    const request = createRequest("plan");
    request.context.skills = ["testing"];

    const result = await executeTask({
      request,
      artifactDir,
      repoRoot,
      emit: (event) => {
        events.push(event);
      },
      runQuery: async () => ({
        success: true,
        responseText: "<final_artifact>Plan complete.</final_artifact>",
        iterations: 1,
      }),
    });

    expect(result.status).toBe("success");
    expect(events.some((event) =>
      event.type === "log" &&
      event.stream === "stderr" &&
      event.message.includes('Requested skill "testing" could not be loaded and will be skipped'),
    )).toBe(true);

    await rm(repoRoot, { recursive: true, force: true });
  });
});

describe("verify commands", () => {
  it("resolves the first real Node binary instead of a Bun shim", async () => {
    const repoRoot = await createRepoRoot();
    const bunBin = join(repoRoot, "bun-bin");
    const nodeBin = join(repoRoot, "node-bin");
    await mkdir(bunBin, { recursive: true });
    await mkdir(nodeBin, { recursive: true });
    const bunNodePath = join(bunBin, "node");
    const realNodePath = join(nodeBin, "node");
    await writeFile(
      bunNodePath,
      "#!/bin/sh\nif [ \"$1\" = \"-p\" ]; then\n  echo bun\n  exit 0\nfi\necho bun\n",
    );
    await writeFile(
      realNodePath,
      "#!/bin/sh\nif [ \"$1\" = \"-p\" ]; then\n  echo node\n  exit 0\nfi\necho node\n",
    );
    await chmod(bunNodePath, 0o755);
    await chmod(realNodePath, 0o755);

    await expect(resolveVerifyNodeBinary(`${bunBin}:${nodeBin}`)).resolves.toBe(realNodePath);

    await rm(repoRoot, { recursive: true, force: true });
  });

  it("rewrites only leading node invocations to an explicit binary", () => {
    expect(rewriteVerifyCommand("node ./node_modules/vitest/vitest.mjs run", "/tmp/node"))
      .toBe("'/tmp/node' ./node_modules/vitest/vitest.mjs run");
    expect(rewriteVerifyCommand("NODE_OPTIONS=--trace-warnings node script.js", "/tmp/node"))
      .toBe("NODE_OPTIONS=--trace-warnings '/tmp/node' script.js");
    expect(rewriteVerifyCommand("bunx vitest run", "/tmp/node"))
      .toBe("bunx vitest run");
  });

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

  it("reads stage-specific fake responses for hyphenated task types", () => {
    process.env["DEVAGENT_EXECUTOR_FAKE_RESPONSE_TASK_INTAKE"] = "Task intake response";
    process.env["DEVAGENT_EXECUTOR_FAKE_RESPONSE_ISSUE_GENERATION"] = "Issue generation response";

    expect(readFakeTaskResponse("task-intake")).toBe("Task intake response");
    expect(readFakeTaskResponse("issue-generation")).toBe("Issue generation response");
  });

  for (const taskType of [
    "task-intake",
    "design",
    "breakdown",
    "issue-generation",
    "triage",
    "plan",
    "test-plan",
    "implement",
    "verify",
    "review",
    "repair",
    "completion",
  ] as const) {
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
        runQuery: async ({ query }) => {
          const responseText = taskType === "breakdown" || taskType === "issue-generation"
            ? createStrictStageResponse(taskType)
            : `Handled ${taskType}\n\n${query}`;
          return {
            success: true,
            responseText,
            iterations: 1,
          };
        },
        emit: (event) => {
          events.push(event);
        },
      });

      expect(result.status).toBe("success");
      expect(result.outcome).toBe("completed");
      const expectedArtifactCount = taskType === "breakdown" || taskType === "issue-generation" ? 2 : 1;
      expect(result.artifacts).toHaveLength(expectedArtifactCount);
      expect(result.artifacts[0]?.kind).toBe(artifactInfoForTask(taskType).kind);
      expect(events.map((event) => event.type)).toContain("started");
      expect(events.map((event) => event.type)).toContain("artifact");
      expect(events.at(-1)?.type).toBe("completed");

      const artifactText = await readFile(result.artifacts.at(-1)!.path, "utf-8");
      if (taskType === "verify") {
        expect(artifactText).toContain("Overall result: pass");
        expect(events.some((event) => event.type === "log")).toBe(true);
      } else if (taskType === "breakdown" || taskType === "issue-generation") {
        expect(result.artifacts[0]?.variant).toBe("structured");
        expect(result.artifacts[1]?.variant).toBe("rendered");
        expect(result.artifacts[0]?.mimeType).toBe("application/json");
        expect(result.artifacts[1]?.mimeType).toBe("text/markdown");
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

  it("passes continuation state through runQuery and persists returned session metadata", async () => {
    const repoRoot = await createRepoRoot();
    const artifactDir = join(repoRoot, "artifacts");
    const request = createRequest("implement");
    request.continuation = {
      mode: "resume",
      reason: "retry_no_progress",
      instructions: "Continue the previous session and make the code change.",
      session: {
        kind: "devagent-headless-v1",
        payload: {
          version: 1,
          messages: [],
        },
      },
    };

    let seenContinuation: TaskExecutionRequest["continuation"] | undefined;
    const result = await executeTask({
      request,
      artifactDir,
      repoRoot,
      runQuery: async (options) => {
        seenContinuation = options.continuation;
        return {
          success: true,
          responseText: "Implemented the change.",
          iterations: 2,
          session: {
            kind: "devagent-headless-v1",
            payload: {
              version: 1,
              messages: [{ role: "assistant", content: "Implemented the change." }],
            },
          },
        };
      },
      emit: () => {},
    });

    expect(seenContinuation?.mode).toBe("resume");
    expect(result.status).toBe("success");
    expect(result.session?.kind).toBe("devagent-headless-v1");
    expect(result.outcome).toBe("completed");

    await rm(repoRoot, { recursive: true, force: true });
  });

  it("classifies exhausted iterations as no-progress", async () => {
    const repoRoot = await createRepoRoot();
    const artifactDir = join(repoRoot, "artifacts");
    const request = createRequest("plan");

    const result = await executeTask({
      request,
      artifactDir,
      repoRoot,
      runQuery: async () => ({
        success: false,
        responseText: "Stopped after reaching the iteration limit.",
        iterations: 6,
        outcome: "no_progress",
        outcomeReason: "iteration_limit",
      }),
      emit: () => {},
    });

    expect(result.status).toBe("failed");
    expect(result.outcome).toBe("no_progress");
    expect(result.outcomeReason).toBe("iteration_limit");

    await rm(repoRoot, { recursive: true, force: true });
  });

  it("does not attempt strict artifact parsing when the workflow run already failed", async () => {
    const repoRoot = await createRepoRoot();
    const artifactDir = join(repoRoot, "artifacts");
    const request = createRequest("issue-generation");
    const events: TaskExecutionEvent[] = [];

    const result = await executeTask({
      request,
      artifactDir,
      repoRoot,
      runQuery: async () => ({
        success: false,
        responseText: "Progress: inspected files, no final artifact yet.",
        iterations: 3,
        outcome: "no_progress",
        outcomeReason: "no_code",
      }),
      emit: (event) => {
        events.push(event);
      },
    });

    expect(result.status).toBe("failed");
    expect(result.outcome).toBe("no_progress");
    expect(result.outcomeReason).toBe("no_code");
    expect(result.artifacts).toEqual([]);
    expect(result.error?.message).toMatch(/no final answer/i);
    expect(result.error?.message).not.toMatch(/strict JSON/i);
    expect(events.some((event) => event.type === "artifact")).toBe(false);
    expect(events.at(-1)).toMatchObject({ type: "completed", status: "failed" });

    await rm(repoRoot, { recursive: true, force: true });
  });

  it("returns a failed result instead of throwing on invalid strict-stage JSON", async () => {
    const repoRoot = await createRepoRoot();
    const artifactDir = join(repoRoot, "artifacts");
    const request = createRequest("breakdown");
    const events: TaskExecutionEvent[] = [];

    const result = await executeTask({
      request,
      artifactDir,
      repoRoot,
      runQuery: async () => ({
        success: true,
        responseText: "{not-json",
        iterations: 1,
      }),
      emit: (event) => {
        events.push(event);
      },
    });

    expect(result.status).toBe("failed");
    expect(result.outcome).toBe("no_progress");
    expect(result.artifacts).toEqual([]);
    expect(result.error?.message).toMatch(/strict JSON with structured and rendered fields/i);
    expect(events.at(-1)).toMatchObject({ type: "completed", status: "failed" });

    const persisted = JSON.parse(await readFile(join(artifactDir, "result.json"), "utf-8")) as {
      status: string;
      outcome?: string;
      artifacts: unknown[];
    };
    expect(persisted.status).toBe("failed");
    expect(persisted.outcome).toBe("no_progress");
    expect(persisted.artifacts).toEqual([]);

    await rm(repoRoot, { recursive: true, force: true });
  });

  it("returns a failed result when strict-stage structured payload fails schema validation", async () => {
    const repoRoot = await createRepoRoot();
    const artifactDir = join(repoRoot, "artifacts");
    const request = createRequest("issue-generation");
    const events: TaskExecutionEvent[] = [];

    const result = await executeTask({
      request,
      artifactDir,
      repoRoot,
      runQuery: async () => ({
        success: true,
        responseText: JSON.stringify({
          structured: {
            summary: "Invalid issue spec",
            issues: [
              {
                id: "I1",
                title: "Missing required fields",
              },
            ],
          },
          rendered: "- I1: Missing required fields",
        }),
        iterations: 1,
      }),
      emit: (event) => {
        events.push(event);
      },
    });

    expect(result.status).toBe("failed");
    expect(result.outcome).toBe("no_progress");
    expect(result.artifacts).toEqual([]);
    expect(result.error?.message).toBeTruthy();
    expect(events.at(-1)).toMatchObject({ type: "completed", status: "failed" });

    const persisted = JSON.parse(await readFile(join(artifactDir, "result.json"), "utf-8")) as {
      status: string;
      artifacts: unknown[];
    };
    expect(persisted.status).toBe("failed");
    expect(persisted.artifacts).toEqual([]);

    await rm(repoRoot, { recursive: true, force: true });
  });
});
