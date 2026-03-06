/**
 * Tests for the headless workflow runner module.
 */

import { describe, it, expect } from "vitest";
import { parseWorkflowArgs } from "../workflow-runner.js";
import type { WorkflowRunArgs } from "../workflow-runner.js";

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
});
