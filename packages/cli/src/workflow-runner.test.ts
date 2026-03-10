import { describe, it, expect, vi } from "vitest";
import { parseWorkflowArgs, validateWorkflowArgs, printRunnerDescription, validateInput } from "./workflow-runner.js";

describe("parseWorkflowArgs", () => {
  function makeArgv(...args: string[]): string[] {
    return ["bun", "devagent", ...args];
  }

  it("parses full workflow run command", () => {
    const result = parseWorkflowArgs(makeArgv(
      "workflow", "run",
      "--phase", "triage",
      "--input", "/tmp/input.json",
      "--output", "/tmp/output.json",
      "--events", "/tmp/events.jsonl",
      "--repo", "/my/repo",
      "--provider", "anthropic",
      "--model", "claude-sonnet-4-6",
      "--max-iterations", "20",
      "--approval-mode", "full-auto",
      "--reasoning", "high",
    ));

    expect(result.subcommand).toBe("run");
    expect(result.phase).toBe("triage");
    expect(result.input).toBe("/tmp/input.json");
    expect(result.output).toBe("/tmp/output.json");
    expect(result.events).toBe("/tmp/events.jsonl");
    expect(result.repo).toBe("/my/repo");
    expect(result.provider).toBe("anthropic");
    expect(result.model).toBe("claude-sonnet-4-6");
    expect(result.maxIterations).toBe(20);
    expect(result.approvalMode).toBe("full-auto");
    expect(result.reasoning).toBe("high");
  });

  it("parses --suggest flag as approval mode", () => {
    const result = parseWorkflowArgs(makeArgv(
      "workflow", "run",
      "--phase", "plan",
      "--input", "/tmp/in.json",
      "--output", "/tmp/out.json",
      "--events", "/tmp/ev.jsonl",
      "--repo", "/repo",
      "--suggest",
    ));
    expect(result.approvalMode).toBe("suggest");
  });

  it("parses --auto-edit flag as approval mode", () => {
    const result = parseWorkflowArgs(makeArgv(
      "workflow", "run",
      "--phase", "implement",
      "--input", "/tmp/in.json",
      "--output", "/tmp/out.json",
      "--events", "/tmp/ev.jsonl",
      "--repo", "/repo",
      "--auto-edit",
    ));
    expect(result.approvalMode).toBe("auto-edit");
  });

  it("returns null subcommand for non-workflow args", () => {
    const result = parseWorkflowArgs(makeArgv("chat"));
    expect(result.subcommand).toBeNull();
  });
});

describe("validateWorkflowArgs", () => {
  const validRaw = {
    subcommand: "run",
    phase: "triage",
    input: "/tmp/input.json",
    output: "/tmp/output.json",
    events: "/tmp/events.jsonl",
    repo: "/repo",
    provider: null,
    model: null,
    maxIterations: null,
    approvalMode: null,
    reasoning: null,
  };

  it("validates a correct set of args", () => {
    const args = validateWorkflowArgs(validRaw);
    expect(args.phase).toBe("triage");
    expect(args.approvalMode).toBeFalsy();
  });

  it("validates approval mode and reasoning when provided", () => {
    const args = validateWorkflowArgs({
      ...validRaw,
      approvalMode: "full-auto",
      reasoning: "high",
    });
    expect(args.approvalMode).toBe("full-auto");
    expect(args.reasoning).toBe("high");
  });
});

describe("validateInput", () => {
  it("accepts valid triage input", () => {
    expect(() => validateInput("triage", { issueNumber: 1, title: "Bug", body: "", labels: [], author: "x" })).not.toThrow();
  });

  it("rejects triage input missing issueNumber", () => {
    expect(() => validateInput("triage", { title: "Bug" })).toThrow("issueNumber");
  });

  it("rejects triage input missing title", () => {
    expect(() => validateInput("triage", { issueNumber: 1 })).toThrow("title");
  });

  it("accepts valid verify input", () => {
    expect(() => validateInput("verify", { commands: ["bun test"] })).not.toThrow();
  });

  it("rejects verify input with non-array commands", () => {
    expect(() => validateInput("verify", { commands: "bun test" })).toThrow("commands");
  });

  it("rejects verify input missing commands", () => {
    expect(() => validateInput("verify", {})).toThrow("commands");
  });

  it("rejects implement input missing acceptedPlan", () => {
    expect(() => validateInput("implement", { issueNumber: 1, title: "X", body: "" })).toThrow("acceptedPlan");
  });

  it("rejects repair input missing round", () => {
    expect(() => validateInput("repair", { issueNumber: 1 })).toThrow("round");
  });

  it("accepts valid gate input", () => {
    expect(() => validateInput("gate", {
      sourcePhase: "triage",
      issueNumber: 1,
      stageOutput: { summary: "OK" },
    })).not.toThrow();
  });

  it("rejects gate input missing sourcePhase", () => {
    expect(() => validateInput("gate", { issueNumber: 1, stageOutput: {} })).toThrow("sourcePhase");
  });

  it("rejects gate input missing stageOutput", () => {
    expect(() => validateInput("gate", { sourcePhase: "plan", issueNumber: 1 })).toThrow("stageOutput");
  });
});

describe("printRunnerDescription", () => {
  it("prints valid JSON runner description", () => {
    const writeSpy = vi.spyOn(process.stdout, "write").mockImplementation(() => true);
    printRunnerDescription();

    const output = writeSpy.mock.calls.map((c) => c[0]).join("");
    writeSpy.mockRestore();

    const desc = JSON.parse(output);
    expect(desc.version).toBeDefined();
    expect(desc.supportedPhases).toContain("triage");
    expect(desc.supportedPhases).toContain("repair");
    expect(desc.supportedPhases).toHaveLength(7);
    expect(desc.supportedPhases).toContain("gate");
    expect(desc.availableProviders).toContain("anthropic");
    expect(desc.supportedApprovalModes).toContain("full-auto");
    expect(desc.supportedReasoningLevels).toContain("high");
  });
});
