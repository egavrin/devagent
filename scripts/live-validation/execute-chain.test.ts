import { describe, expect, it } from "bun:test";
import type { IssueSpecDoc } from "@devagent-sdk/types";
import {
  EXECUTE_CHAIN_STAGES,
  buildExecuteChainRequest,
  extractIssueUnitFromIssueSpec,
} from "./execute-chain-lib";

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
});
