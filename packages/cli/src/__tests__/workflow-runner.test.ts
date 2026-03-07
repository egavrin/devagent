/**
 * Tests for the headless workflow runner module.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { parseWorkflowArgs, runWorkflowPhase } from "../workflow-runner.js";
import type { WorkflowRunArgs } from "../workflow-runner.js";

let lastTaskLoopArgs: unknown = null;

vi.mock("node:fs", async () => {
  const actual = await vi.importActual<typeof import("node:fs")>("node:fs");
  return {
    ...actual,
    writeFileSync: vi.fn(actual.writeFileSync),
  };
});

vi.mock("@devagent/core", () => {
  return {
    EventBus: class {
      on() {}
    },
    ApprovalGate: class {
      constructor() {}
    },
    ContextManager: class {
      constructor() {}
    },
    MemoryStore: class {
      constructor() {}
    },
    loadConfig: vi.fn(() => ({
      provider: "mock",
      model: "mock-model",
      budget: { maxIterations: 5, maxContextTokens: 1000, responseHeadroom: 100 },
      context: { keepRecentMessages: 2 },
      memory: { dailyDecay: 0, minRelevance: 0, accessBoost: 0 },
      sessionState: {},
      approval: { mode: "full-auto", auditLog: false, toolOverrides: {}, pathRules: [] },
      providers: { mock: { model: "mock-model" } },
    })),
    resolveProviderCredentials: vi.fn(async (config: unknown) => config),
    loadModelRegistry: vi.fn(),
    lookupModelCapabilities: vi.fn(() => undefined),
    lookupModelEntry: vi.fn(() => null),
    DEFAULT_BUDGET: { maxContextTokens: 1000, responseHeadroom: 100, maxIterations: 5 },
    DEFAULT_CONTEXT: { keepRecentMessages: 2 },
    SkillRegistry: class {
      list() {
        return [];
      }
    },
    RepositoryInstructionLoader: class {
      constructor() {}
      load() {
        return [];
      }
    },
    ArtifactStore: class {
      save() {}
    },
    ApprovalMode: {
      SUGGEST: "suggest",
      AUTO_EDIT: "auto-edit",
      FULL_AUTO: "full-auto",
    },
    WORKFLOW_SCHEMA_VERSION: "1",
  };
});

vi.mock("@devagent/providers", () => ({
  createDefaultRegistry: () => ({
    get: () => ({}),
  }),
}));

vi.mock("@devagent/tools", () => ({
  createDefaultToolRegistry: () => ({}),
}));

vi.mock("@devagent/engine", () => ({
  TaskLoop: class {
    constructor(args: unknown) {
      lastTaskLoopArgs = args;
    }
    async run() {
      return {
        lastText: "```json\n{}\n```",
        status: "completed",
        iterations: 0,
        aborted: false,
      };
    }
  },
  CheckpointManager: class {
    constructor() {}
  },
  DoubleCheck: class {
    constructor() {}
  },
  DEFAULT_DOUBLE_CHECK_OPTIONS: {},
  SessionState: class {
    constructor() {}
  },
}));

describe("workflow-runner", () => {
  describe("module exports", () => {
    it("exports runWorkflowPhase", async () => {
      const mod = await import("../workflow-runner.js");
      expect(typeof mod.runWorkflowPhase).toBe("function");
    });

    it("exports handleWorkflowCommand", async () => {
      const mod = await import("../workflow-runner.js");
      expect(typeof mod.handleWorkflowCommand).toBe("function");
    });

    it("exports parseWorkflowArgs", async () => {
      const mod = await import("../workflow-runner.js");
      expect(typeof mod.parseWorkflowArgs).toBe("function");
    });
  });

  describe("parseWorkflowArgs", () => {
    it("parses complete args", () => {
      const result = parseWorkflowArgs([
        "--phase", "triage",
        "--repo", "/tmp/repo",
        "--input", "input.json",
        "--output", "output.json",
      ]);
      expect(result).not.toBeNull();
      expect(result!.phase).toBe("triage");
      expect(result!.repo).toBe("/tmp/repo");
      expect(result!.inputFile).toBe("input.json");
      expect(result!.outputFile).toBe("output.json");
    });

    it("parses optional flags", () => {
      const result = parseWorkflowArgs([
        "--phase", "implement",
        "--repo", "/tmp/repo",
        "--input", "in.json",
        "--output", "out.json",
        "--events", "events.jsonl",
        "--provider", "openai",
        "--model", "gpt-4",
        "--max-iterations", "10",
        "--approval", "full-auto",
      ]);
      expect(result).not.toBeNull();
      expect(result!.phase).toBe("implement");
      expect(result!.eventsFile).toBe("events.jsonl");
      expect(result!.provider).toBe("openai");
      expect(result!.model).toBe("gpt-4");
      expect(result!.maxIterations).toBe(10);
      expect(result!.approval).toBe("full-auto");
    });

    it("returns null for missing required args", () => {
      const result = parseWorkflowArgs(["--phase", "triage"]);
      expect(result).toBeNull();
    });

    it("returns null for invalid phase", () => {
      const result = parseWorkflowArgs([
        "--phase", "invalid",
        "--repo", "/tmp",
        "--input", "in.json",
        "--output", "out.json",
      ]);
      expect(result).toBeNull();
    });

    it("returns null for empty args", () => {
      const result = parseWorkflowArgs([]);
      expect(result).toBeNull();
    });

    it("accepts all valid phases", () => {
      const phases = ["triage", "plan", "implement", "verify", "review", "repair"];
      for (const phase of phases) {
        const result = parseWorkflowArgs([
          "--phase", phase,
          "--repo", "/tmp",
          "--input", "in.json",
          "--output", "out.json",
        ]);
        expect(result).not.toBeNull();
        expect(result!.phase).toBe(phase);
      }
    });
  });

  describe("runWorkflowPhase", () => {
    let tmpDir: string;
    let inputFile: string;
    let outputFile: string;
    let actualFs: typeof import("node:fs");

    beforeEach(async () => {
      actualFs = await vi.importActual<typeof import("node:fs")>("node:fs");
      tmpDir = mkdtempSync(join(tmpdir(), "workflow-runner-test-"));
      inputFile = join(tmpDir, "input.json");
      outputFile = join(tmpDir, "output.json");
      actualFs.writeFileSync(inputFile, JSON.stringify({ issueId: "1" }));
      lastTaskLoopArgs = null;
      vi.clearAllMocks();
    });

    afterEach(() => {
      rmSync(tmpDir, { recursive: true, force: true });
      lastTaskLoopArgs = null;
    });

    it("honors max-iterations 0 override", async () => {
      const args: WorkflowRunArgs = {
        phase: "implement",
        repo: tmpDir,
        inputFile,
        outputFile,
        maxIterations: 0,
      };

      await runWorkflowPhase(args);

      const loopArgs = lastTaskLoopArgs as { config: { budget: { maxIterations: number } } };
      expect(loopArgs.config.budget.maxIterations).toBe(0);
    });

    it("exits when output write fails", async () => {
      const args: WorkflowRunArgs = {
        phase: "triage",
        repo: tmpDir,
        inputFile,
        outputFile,
      };

      const exitSpy = vi.spyOn(process, "exit").mockImplementation((code?: number) => {
        throw new Error(`exit:${code}`);
      });
      const stderrSpy = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
      const writeMock = vi.mocked(writeFileSync);
      writeMock.mockImplementation((path, data, options) => {
        if (path === outputFile) {
          throw new Error("no-write");
        }
        return actualFs.writeFileSync(path as string, data, options as any);
      });

      await expect(runWorkflowPhase(args)).rejects.toThrow("exit:1");
      expect(stderrSpy).toHaveBeenCalled();

      exitSpy.mockRestore();
      stderrSpy.mockRestore();
    });
  });
});
