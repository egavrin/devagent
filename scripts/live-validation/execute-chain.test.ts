import { describe, expect, it } from "bun:test";
import type { IssueSpecDoc } from "@devagent-sdk/types";
import { mkdtemp, mkdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  EXECUTE_CHAIN_STAGES,
  buildExecuteChainRequest,
  extractIssueUnitFromIssueSpec,
} from "./execute-chain-lib";
import { buildStageFailureMessage as buildStageFailureMessageFromRun } from "./execute-chain";

describe("execute chain helpers", () => {
  it("defines the full staged chain in the expected order", () => {
    expect(EXECUTE_CHAIN_STAGES).toEqual([
      "design",
      "breakdown",
      "issue-generation",
      "implement",
      "review",
      "repair",
    ]);
  });

  it("threads design artifacts into breakdown comments", () => {
    const request = buildExecuteChainRequest({
      stage: "breakdown",
      workspaceRoot: "/tmp/workspace",
      sourceRepoRoot: "/tmp/source",
      provider: "chatgpt",
      model: "gpt-5.4",
      taskIdPrefix: "chain",
      priorArtifacts: {
        designDoc: "# Design\n\nUse README.md only.\n",
      },
    });

    expect(request.taskType).toBe("breakdown");
    expect(request.context.comments?.[0]?.author).toBe("design-artifact");
    expect(request.context.comments?.[0]?.body).toContain("Approved design artifact");
    expect(request.context.comments?.[0]?.body).toContain("README.md only");
  });

  it("builds implement requests from the first generated issue", () => {
    const issueSpec: IssueSpecDoc = {
      summary: "One issue",
      issues: [{
        id: "I1",
        title: "Document validation flow",
        problemStatement: "README is missing the validation workflow.",
        rationale: "Close the docs gap.",
        scope: ["README.md update"],
        acceptanceCriteria: ["README mentions assembler and verifier workflow."],
        dependencies: [],
        linkedDesignSections: ["DesignDoc#Workflow"],
        linkedBreakdownTaskIds: ["B1"],
        grounding: {
          repoPaths: ["README.md"],
          codeSymbols: [],
        },
        requiredTests: [],
        outOfScope: [],
        implementationNotes: ["Keep the edit small."],
      }],
    };

    const request = buildExecuteChainRequest({
      stage: "implement",
      workspaceRoot: "/tmp/workspace",
      sourceRepoRoot: "/tmp/source",
      provider: "chatgpt",
      model: "gpt-5.4",
      taskIdPrefix: "chain",
      priorArtifacts: {
        designDoc: "design",
        breakdownDoc: "breakdown",
        issueSpec: "issue-spec",
        issueStructured: issueSpec,
      },
    });

    expect(request.issueUnit).toEqual({
      id: "I1",
      title: "Document validation flow",
      sequence: 1,
      dependencyIds: [],
      acceptanceCriteria: ["README mentions assembler and verifier workflow."],
      linkedArtifactVersionIds: ["B1"],
    });
    expect(request.context.changedFilesHint).toEqual(["README.md"]);
    expect(request.context.comments?.some((comment) => comment.author === "issue-spec-artifact")).toBe(true);
  });

  it("extracts the first issue into an issue unit", () => {
    const issueUnit = extractIssueUnitFromIssueSpec({
      summary: "Issues",
      issues: [{
        id: "I7",
        title: "Doc change",
        problemStatement: "Problem",
        rationale: "Why",
        scope: ["Scope"],
        acceptanceCriteria: ["Done"],
        dependencies: ["I1"],
        linkedDesignSections: ["DesignDoc#Doc"],
        linkedBreakdownTaskIds: ["B3"],
        grounding: {
          repoPaths: ["README.md"],
          codeSymbols: [],
        },
        requiredTests: [],
        outOfScope: [],
        implementationNotes: ["Note"],
      }],
    });

    expect(issueUnit).toEqual({
      id: "I7",
      title: "Doc change",
      sequence: 1,
      dependencyIds: ["I1"],
      acceptanceCriteria: ["Done"],
      linkedArtifactVersionIds: ["B3"],
    });
  });

  it("summarizes result.json errors and final session summary for failed stages", async () => {
    const dir = await mkdtemp(join(tmpdir(), "devagent-chain-failure-"));
    const artifactDir = join(dir, "artifacts");
    await mkdir(artifactDir, { recursive: true });
    await writeFile(join(artifactDir, "result.json"), JSON.stringify({
      protocolVersion: "0.1",
      taskId: "chain-review",
      status: "failed",
      artifacts: [],
      error: {
        code: "EXECUTION_FAILED",
        message: "Task loop exhausted the iteration limit",
      },
      outcome: "no_progress",
      outcomeReason: "iteration_limit",
      metrics: {
        startedAt: "2026-04-21T00:00:00.000Z",
        finishedAt: "2026-04-21T00:00:01.000Z",
        durationMs: 1000,
      },
      session: {
        kind: "devagent-headless-v1",
        payload: {
          version: 1,
          messages: [
            { role: "assistant", content: "No defects found." },
          ],
        },
      },
    }));

    const message = await buildStageFailureMessageFromRun({
      exitCode: 1,
      stdout: "",
      stderr: "",
      timedOut: false,
      durationMs: 1000,
    }, artifactDir);

    expect(message).toContain("Result: failed");
    expect(message).toContain("Error: EXECUTION_FAILED: Task loop exhausted the iteration limit");
    expect(message).toContain("Outcome reason: iteration_limit");
    expect(message).toContain("Final assistant summary: No defects found.");
  });
});
