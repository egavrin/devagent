import { describe, it, expect, beforeEach } from "vitest";
import { SessionState, extractEnvFact } from "./session-state.js";
import { createPlanTool } from "./plan-tool.js";
import { EventBus } from "@devagent/core";
import type { PlanStep } from "./plan-tool.js";
import type {
  SessionStateJSON,
  SessionStatePersistence,
  ToolResultSummary,
} from "./session-state.js";

describe("SessionState", () => {
  let state: SessionState;

  beforeEach(() => {
    state = new SessionState();
  });

  // ─── Plan ────────────────────────────────────────────────────

  describe("plan tracking", () => {
    it("returns null when no plan is set", () => {
      expect(state.getPlan()).toBeNull();
    });

    it("stores and retrieves a plan", () => {
      const steps: PlanStep[] = [
        { description: "Read codebase", status: "completed" },
        { description: "Write tests", status: "in_progress" },
        { description: "Implement feature", status: "pending" },
      ];
      state.setPlan(steps);
      expect(state.getPlan()).toEqual(steps);
    });

    it("overwrites previous plan on each call", () => {
      state.setPlan([{ description: "Old step", status: "pending" }]);
      const newSteps: PlanStep[] = [
        { description: "New step A", status: "in_progress" },
        { description: "New step B", status: "pending" },
      ];
      state.setPlan(newSteps);
      expect(state.getPlan()).toEqual(newSteps);
      expect(state.getPlan()!.length).toBe(2);
    });
  });

  // ─── Plan Tool Getter Indirection ─────────────────────────────

  describe("plan tool getter indirection", () => {
    it("calls setPlan on the current instance after reassignment", async () => {
      const bus = new EventBus();
      let sessionState: SessionState | undefined = new SessionState();
      const oldState = sessionState;

      const tool = createPlanTool(bus, () => sessionState);

      // Simulate resume: replace sessionState with a new instance
      sessionState = new SessionState();

      const result = await tool.handler(
        { steps: JSON.stringify([{ description: "Step 1", status: "pending" }]) },
        { repoRoot: "/tmp" },
      );
      expect(result.success).toBe(true);

      // The NEW instance should have the plan
      const plan = sessionState.getPlan()!;
      expect(plan).toHaveLength(1);
      expect(plan[0]!.description).toBe("Step 1");
      expect(plan[0]!.status).toBe("pending");

      // The OLD instance should NOT have the plan
      expect(oldState.getPlan()).toBeNull();
    });
  });

  // ─── Modified Files ──────────────────────────────────────────

  describe("modified files tracking", () => {
    it("returns empty array when no files recorded", () => {
      expect(state.getModifiedFiles()).toEqual([]);
    });

    it("records and retrieves file paths", () => {
      state.recordModifiedFile("/src/foo.ts");
      state.recordModifiedFile("/src/bar.ts");
      expect(state.getModifiedFiles()).toEqual(["/src/foo.ts", "/src/bar.ts"]);
    });

    it("moves re-recorded file paths to the end", () => {
      state.recordModifiedFile("/src/foo.ts");
      state.recordModifiedFile("/src/bar.ts");
      state.recordModifiedFile("/src/foo.ts");
      expect(state.getModifiedFiles()).toEqual(["/src/bar.ts", "/src/foo.ts"]);
    });

    it("re-recorded file survives cap eviction by moving to most recent", () => {
      const ss = new SessionState({ maxModifiedFiles: 3 });
      ss.recordModifiedFile("/src/a.ts");
      ss.recordModifiedFile("/src/b.ts");
      ss.recordModifiedFile("/src/c.ts");
      ss.recordModifiedFile("/src/a.ts"); // refresh recency
      ss.recordModifiedFile("/src/d.ts"); // evicts oldest (b)
      expect(ss.getModifiedFiles()).toEqual([
        "/src/c.ts",
        "/src/a.ts",
        "/src/d.ts",
      ]);
    });

    it("respects MAX_MODIFIED_FILES cap of 50", () => {
      for (let i = 0; i < 60; i++) {
        state.recordModifiedFile(`/src/file-${i}.ts`);
      }
      const files = state.getModifiedFiles();
      expect(files.length).toBe(50);
      // Should keep the most recent 50 (drop earliest)
      expect(files[0]).toBe("/src/file-10.ts");
      expect(files[49]).toBe("/src/file-59.ts");
    });
  });

  // ─── Environment Facts ───────────────────────────────────────

  describe("environment facts tracking", () => {
    it("returns empty array when no facts stored", () => {
      expect(state.getEnvFacts()).toEqual([]);
    });

    it("stores and retrieves a fact", () => {
      state.addEnvFact("runtime", "Node.js 20.11");
      expect(state.getEnvFacts()).toEqual(["Node.js 20.11"]);
    });

    it("deduplicates by key (overwrites existing)", () => {
      state.addEnvFact("runtime", "Node.js 18");
      state.addEnvFact("runtime", "Node.js 20.11");
      expect(state.getEnvFacts()).toEqual(["Node.js 20.11"]);
    });

    it("stores multiple different facts", () => {
      state.addEnvFact("runtime", "Node.js 20.11");
      state.addEnvFact("os", "Linux x86_64");
      state.addEnvFact("package-manager", "bun 1.1.0");
      const facts = state.getEnvFacts();
      expect(facts).toHaveLength(3);
      expect(facts).toContain("Node.js 20.11");
      expect(facts).toContain("Linux x86_64");
      expect(facts).toContain("bun 1.1.0");
    });

    it("respects MAX_ENV_FACTS cap of 20", () => {
      for (let i = 0; i < 25; i++) {
        state.addEnvFact(`key-${i}`, `fact-${i}`);
      }
      const facts = state.getEnvFacts();
      expect(facts.length).toBe(20);
    });
  });

  // ─── Tool Result Summaries ───────────────────────────────────

  describe("tool result summaries", () => {
    it("returns empty array when no summaries stored", () => {
      expect(state.getToolSummaries()).toEqual([]);
    });

    it("stores and retrieves tool summaries", () => {
      const summary: ToolResultSummary = {
        tool: "edit_file",
        target: "/src/foo.ts",
        summary: "Applied edit: added function bar()",
        iteration: 3,
      };
      state.addToolSummary(summary);
      expect(state.getToolSummaries()).toEqual([summary]);
    });

    it("truncates summaries exceeding 2000 chars", () => {
      const longText = "x".repeat(2500);
      state.addToolSummary({
        tool: "edit_file",
        target: "/src/foo.ts",
        summary: longText,
        iteration: 1,
      });
      const summaries = state.getToolSummaries();
      expect(summaries[0]!.summary.length).toBe(2000);
    });

    it("respects max cap of 30 (default)", () => {
      for (let i = 0; i < 40; i++) {
        state.addToolSummary({
          tool: "edit_file",
          target: `/src/file-${i}.ts`,
          summary: `edit ${i}`,
          iteration: i,
        });
      }
      const summaries = state.getToolSummaries();
      expect(summaries.length).toBe(30);
      expect(summaries[0]!.target).toBe("/src/file-10.ts");
    });

    it("deduplicates by tool+target, replacing older entry", () => {
      state.addToolSummary({
        tool: "read_file",
        target: "/src/main.ts",
        summary: "Read 100 lines",
        iteration: 1,
      });
      state.addToolSummary({
        tool: "run_command",
        target: "run_command",
        summary: "typecheck passed",
        iteration: 2,
      });
      state.addToolSummary({
        tool: "read_file",
        target: "/src/main.ts",
        summary: "Read 150 lines",
        iteration: 3,
      });

      const summaries = state.getToolSummaries();
      expect(summaries.length).toBe(2);
      // The read_file entry should be updated (moved to end)
      expect(summaries[0]!.tool).toBe("run_command");
      expect(summaries[1]!.tool).toBe("read_file");
      expect(summaries[1]!.summary).toBe("Read 150 lines");
      expect(summaries[1]!.iteration).toBe(3);
    });

    it("does not track when trackToolResults is disabled", () => {
      const ss = new SessionState({ trackToolResults: false });
      ss.addToolSummary({
        tool: "edit_file",
        target: "/src/foo.ts",
        summary: "test",
        iteration: 1,
      });
      expect(ss.getToolSummaries()).toEqual([]);
    });
  });

  describe("readonly coverage tracking", () => {
    it("tracks readonly coverage independently of tool summary cap", () => {
      const ss = new SessionState({ maxToolSummaries: 2 });
      for (let i = 0; i < 5; i++) {
        const target = `/src/file-${i}.ts`;
        ss.addToolSummary({
          tool: "read_file",
          target,
          summary: `Read ${i}`,
          iteration: i,
        });
        ss.recordReadonlyCoverage("read_file", target);
      }

      expect(ss.getToolSummaries().length).toBe(2);
      const coverage = ss.getReadonlyCoverage();
      expect(coverage.get("read_file")).toEqual([
        "/src/file-0.ts",
        "/src/file-1.ts",
        "/src/file-2.ts",
        "/src/file-3.ts",
        "/src/file-4.ts",
      ]);
    });

    it("deduplicates coverage targets by recency", () => {
      state.recordReadonlyCoverage("git_diff", "/src/a.ts");
      state.recordReadonlyCoverage("git_diff", "/src/b.ts");
      state.recordReadonlyCoverage("git_diff", "/src/a.ts");

      const coverage = state.getReadonlyCoverage();
      expect(coverage.get("git_diff")).toEqual(["/src/b.ts", "/src/a.ts"]);
    });
  });

  // ─── toSystemMessage ─────────────────────────────────────────

  describe("toSystemMessage", () => {
    it("returns null when state is empty", () => {
      expect(state.toSystemMessage()).toBeNull();
    });

    it("includes plan section", () => {
      state.setPlan([
        { description: "Read codebase", status: "completed" },
        { description: "Write tests", status: "in_progress" },
        { description: "Implement feature", status: "pending" },
      ]);
      const msg = state.toSystemMessage();
      expect(msg).not.toBeNull();
      expect(msg).toContain("[SESSION STATE — preserved across compaction]");
      expect(msg).toContain("## Plan");
      expect(msg).toContain("- [completed] Read codebase");
      expect(msg).toContain("- [in_progress] Write tests");
      expect(msg).toContain("- [pending] Implement feature");
    });

    it("includes preservation instruction even when no steps are completed yet", () => {
      state.setPlan([
        { description: "Locate spec rules", status: "in_progress" },
        { description: "Reproduce behavior", status: "pending" },
        { description: "Compare vs spec", status: "pending" },
        { description: "Write report", status: "pending" },
      ]);
      const msg = state.toSystemMessage()!;
      expect(msg).toContain("## Plan");
      // Must carry an explicit instruction even before any step completes,
      // so the LLM does not rewrite step names after context compaction.
      expect(msg).toMatch(/IMPORTANT.*Do NOT.*steps/i);
    });

    it("includes modified files section", () => {
      state.recordModifiedFile("/src/foo.ts");
      state.recordModifiedFile("/src/bar.ts");
      const msg = state.toSystemMessage();
      expect(msg).not.toBeNull();
      expect(msg).toContain("[SESSION STATE — preserved across compaction]");
      expect(msg).toContain("## Modified files");
      expect(msg).toContain("- /src/foo.ts");
      expect(msg).toContain("- /src/bar.ts");
    });

    it("includes env facts section", () => {
      state.addEnvFact("runtime", "Node.js 20.11");
      state.addEnvFact("os", "Linux x86_64");
      const msg = state.toSystemMessage();
      expect(msg).not.toBeNull();
      expect(msg).toContain("[SESSION STATE — preserved across compaction]");
      expect(msg).toContain("## Environment");
      expect(msg).toContain("- Node.js 20.11");
      expect(msg).toContain("- Linux x86_64");
    });

    it("combines all sections", () => {
      state.setPlan([
        { description: "Step one", status: "completed" },
        { description: "Step two", status: "in_progress" },
      ]);
      state.recordModifiedFile("/src/index.ts");
      state.addEnvFact("runtime", "Node.js 20.11");

      const msg = state.toSystemMessage();
      expect(msg).not.toBeNull();
      expect(msg).toContain("[SESSION STATE — preserved across compaction]");
      expect(msg).toContain("## Plan");
      expect(msg).toContain("- [completed] Step one");
      expect(msg).toContain("- [in_progress] Step two");
      expect(msg).toContain("## Modified files");
      expect(msg).toContain("- /src/index.ts");
      expect(msg).toContain("## Environment");
      expect(msg).toContain("- Node.js 20.11");
    });

    it("only includes sections that have data", () => {
      state.recordModifiedFile("/src/foo.ts");
      const msg = state.toSystemMessage()!;
      expect(msg).toContain("## Modified files");
      expect(msg).not.toContain("## Plan");
      expect(msg).not.toContain("## Environment");
    });

    it("includes tool summaries in full tier", () => {
      state.addToolSummary({
        tool: "edit_file",
        target: "/src/foo.ts",
        summary: "Added function bar()",
        iteration: 5,
      });
      const msg = state.toSystemMessage("full")!;
      expect(msg).toContain("## Recent activity");
      expect(msg).toContain("[iter 5] edit_file(/src/foo.ts): Added function bar()");
    });

    it("includes tool summaries in compact tier (limited to 20)", () => {
      for (let i = 0; i < 25; i++) {
        state.addToolSummary({
          tool: "edit_file",
          target: `/src/f-${i}.ts`,
          summary: `summary ${i}`,
          iteration: i,
        });
      }
      state.recordModifiedFile("/src/foo.ts");
      const msg = state.toSystemMessage("compact")!;
      expect(msg).toContain("## Modified files");
      expect(msg).toContain("## Recent activity");
      const count = (msg.match(/\[iter /g) ?? []).length;
      expect(count).toBe(20);
      expect(msg).toContain("[iter 24]");
      expect(msg).not.toContain("[iter 4]");
    });

    it("includes recent activity in minimal tier (limited to 5)", () => {
      state.setPlan([{ description: "Step one", status: "in_progress" }]);
      state.recordModifiedFile("/src/foo.ts");
      state.addEnvFact("runtime", "Node.js 20");
      for (let i = 0; i < 8; i++) {
        state.addToolSummary({
          tool: "edit_file",
          target: `/src/f-${i}.ts`,
          summary: `summary ${i}`,
          iteration: i,
        });
      }

      const msg = state.toSystemMessage("minimal")!;
      expect(msg).toContain("## Plan");
      expect(msg).not.toContain("## Modified files");
      expect(msg).not.toContain("## Environment");
      expect(msg).toContain("## Recent activity");
      const count = (msg.match(/\[iter /g) ?? []).length;
      expect(count).toBe(5);
      expect(msg).toContain("[iter 7]");
      expect(msg).not.toContain("[iter 2]");
    });

    it("includes all stored summaries in full tier", () => {
      for (let i = 0; i < 25; i++) {
        state.addToolSummary({
          tool: "edit_file",
          target: `/src/f-${i}.ts`,
          summary: `summary ${i}`,
          iteration: i,
        });
      }
      const msg = state.toSystemMessage("full")!;
      const count = (msg.match(/\[iter /g) ?? []).length;
      expect(count).toBe(25);
      expect(msg).toContain("[iter 0]");
      expect(msg).toContain("[iter 24]");
    });

    it("includes readonly coverage section in minimal tier", () => {
      for (let i = 0; i < 6; i++) {
        state.recordReadonlyCoverage("read_file", `/src/f-${i}.ts`);
      }

      const msg = state.toSystemMessage("minimal")!;
      expect(msg).toContain("## Readonly coverage");
      expect(msg).toContain("read_file (6)");
      expect(msg).toContain("/src/f-5.ts");
      expect(msg).toContain("... (+1 more)");
    });

    it("includes IMPORTANT instruction when plan has completed steps", () => {
      state.setPlan([
        { description: "Read codebase", status: "completed" },
        { description: "Write tests", status: "completed" },
        { description: "Implement feature", status: "in_progress" },
        { description: "Deploy", status: "pending" },
      ]);
      const msg = state.toSystemMessage()!;
      expect(msg).toContain("IMPORTANT: The following plan steps reflect verified progress");
      expect(msg).toContain("Do NOT reset completed steps");
      expect(msg).toContain("Continue from where the plan left off");
    });

    it("includes preservation instruction even when no steps are completed (all-pending plan)", () => {
      state.setPlan([
        { description: "Read codebase", status: "in_progress" },
        { description: "Write tests", status: "pending" },
      ]);
      const msg = state.toSystemMessage()!;
      expect(msg).toContain("## Plan");
      // Instruction must always be present so the LLM does not rewrite step
      // names after context compaction, even before any step completes.
      expect(msg).toContain("IMPORTANT");
      expect(msg).toContain("Do NOT replace or rename these steps");
    });

    it("includes findings section in all tiers", () => {
      state.addFinding("Memory leak", "Buffer not freed in auth", 5);
      state.addFinding("Race condition", "Mutex not held during write", 7);

      const full = state.toSystemMessage("full")!;
      expect(full).toContain("## Findings");
      expect(full).toContain("Memory leak");
      expect(full).toContain("Buffer not freed in auth");

      // Also appears in compact tier
      const compact = state.toSystemMessage("compact")!;
      expect(compact).toContain("## Findings");

      // Also appears in minimal tier
      state.setPlan([{ description: "Test", status: "pending" }]);
      const minimal = state.toSystemMessage("minimal")!;
      expect(minimal).toContain("## Findings");
    });

    it("flattens multi-line summaries with ' | ' separator", () => {
      state.addToolSummary({
        tool: "search_files",
        target: "search:createPlanTool",
        summary: "5 matches for \"createPlanTool\"\nplan-tool.ts:42: export function createPlanTool(ctx)\ntask-loop.ts:105: const tool = createPlanTool(this.context)",
        iteration: 3,
      });
      const msg = state.toSystemMessage("full")!;
      expect(msg).toContain("## Recent activity");
      // Multi-line summary should be flattened to single line with ' | '
      expect(msg).not.toContain("5 matches for \"createPlanTool\"\nplan-tool.ts");
      expect(msg).toContain("5 matches for \"createPlanTool\" | plan-tool.ts:42");
    });

    it("includes anti-re-read instruction when tool summaries exist", () => {
      state.addToolSummary({
        tool: "git_diff",
        target: "/src/foo.ts",
        summary: "diff: 50 lines",
        iteration: 3,
      });
      const msg = state.toSystemMessage("full")!;
      expect(msg).toContain("already been examined");
      expect(msg).toContain("Do NOT re-read");
    });

    it("compact tier keeps all tracked modified files (up to cap)", () => {
      for (let i = 0; i < 20; i++) {
        state.recordModifiedFile(`/src/file-${i}.ts`);
      }
      const msg = state.toSystemMessage("compact")!;
      expect(msg).toContain("/src/file-19.ts");
      expect(msg).toContain("/src/file-0.ts");
    });

    it("leads with evidence section listing modified files with tool summary context", () => {
      state.recordModifiedFile("src/foo.ts");
      state.recordModifiedFile("src/bar.ts");
      state.addToolSummary({
        tool: "edit_file",
        target: "src/foo.ts",
        summary: "replaced 3 patterns",
        iteration: 5,
      });
      state.setPlan([
        { description: "Edit foo.ts", status: "completed" },
        { description: "Edit bar.ts", status: "in_progress" },
      ]);

      const msg = state.toSystemMessage()!;
      // Evidence section must appear before Plan section
      const evidenceIdx = msg.indexOf("## Completed work");
      const planIdx = msg.indexOf("## Plan");
      expect(evidenceIdx).toBeGreaterThanOrEqual(0);
      expect(planIdx).toBeGreaterThan(evidenceIdx);
      // Should include file + tool summary reference
      expect(msg).toContain("src/foo.ts");
      expect(msg).toContain("replaced 3 patterns");
      expect(msg).toContain("iter 5");
    });

    it("evidence section omitted when no files have been modified", () => {
      state.setPlan([{ description: "Step 1", status: "pending" }]);
      const msg = state.toSystemMessage()!;
      expect(msg).not.toContain("## Completed work");
    });

    it("evidence section lists file without detail when no matching tool summary", () => {
      state.recordModifiedFile("src/baz.ts");
      state.setPlan([{ description: "Step 1", status: "completed" }]);
      const msg = state.toSystemMessage()!;
      expect(msg).toContain("## Completed work");
      expect(msg).toContain("src/baz.ts");
      // Bare file line — no summary detail appended
      expect(msg).not.toContain(": (iter");
    });

    it("evidence section omitted in minimal tier", () => {
      state.recordModifiedFile("src/foo.ts");
      state.setPlan([{ description: "Step 1", status: "in_progress" }]);
      const msg = state.toSystemMessage("minimal")!;
      expect(msg).not.toContain("## Completed work");
    });
  });

  // ─── JSON Serialization ──────────────────────────────────────

  describe("JSON serialization", () => {
    it("round-trips empty state", () => {
      const json = state.toJSON();
      const restored = SessionState.fromJSON(json);
      expect(restored.getPlan()).toBeNull();
      expect(restored.getModifiedFiles()).toEqual([]);
      expect(restored.getEnvFacts()).toEqual([]);
      expect(restored.getToolSummaries()).toEqual([]);
    });

    it("round-trips full state", () => {
      state.setPlan([
        { description: "Step 1", status: "completed" },
        { description: "Step 2", status: "in_progress" },
      ]);
      state.recordModifiedFile("/src/foo.ts");
      state.recordModifiedFile("/src/bar.ts");
      state.addEnvFact("cmd-not-found:rg", "rg is not installed");
      state.addToolSummary({
        tool: "edit_file",
        target: "/src/foo.ts",
        summary: "Added bar()",
        iteration: 3,
      });

      const json = state.toJSON();
      const restored = SessionState.fromJSON(json);

      expect(restored.getPlan()).toEqual(state.getPlan());
      expect(restored.getModifiedFiles()).toEqual(state.getModifiedFiles());
      expect(restored.getEnvFacts()).toEqual(state.getEnvFacts());
      expect(restored.getToolSummaries()).toEqual(state.getToolSummaries());
    });

    it("round-trips findings", () => {
      state.addFinding("Bug found", "Off by one in loop", 3);
      state.addFinding("Security issue", "SQL injection via user input", 5);
      const json = state.toJSON();
      const restored = SessionState.fromJSON(json);
      expect(restored.getFindings()).toEqual(state.getFindings());
    });

    it("toJSON returns a plain object with version", () => {
      state.setPlan([{ description: "Test", status: "pending" }]);
      const json = state.toJSON();
      expect(json).toEqual({
        version: 1,
        plan: [{ description: "Test", status: "pending" }],
        modifiedFiles: [],
        envFacts: [],
        toolSummaries: [],
      });
    });

    it("fromJSON handles missing optional fields (forward compat)", () => {
      const partial = { version: 1 as const, plan: null, modifiedFiles: [], envFacts: [], toolSummaries: [] };
      const restored = SessionState.fromJSON(partial);
      expect(restored.getModifiedFiles()).toEqual([]);
      expect(restored.getEnvFacts()).toEqual([]);
      expect(restored.getToolSummaries()).toEqual([]);
    });

    it("fromJSON re-applies caps when config limits are smaller than persisted data", () => {
      // Build a JSON payload that exceeds the configured limits
      const data: SessionStateJSON = {
        version: 1,
        plan: null,
        modifiedFiles: Array.from({ length: 20 }, (_, i) => `/src/file-${i}.ts`),
        envFacts: Array.from({ length: 15 }, (_, i) => ({ key: `k${i}`, value: `v${i}` })),
        toolSummaries: Array.from({ length: 10 }, (_, i) => ({
          tool: "edit_file",
          target: `/src/f${i}.ts`,
          summary: `summary ${i}`,
          iteration: i,
        })),
        findings: Array.from({ length: 10 }, (_, i) => ({
          title: `Finding ${i}`,
          detail: `Detail ${i}`,
          iteration: i,
        })),
      };

      // Restore with strict limits
      const ss = SessionState.fromJSON(data, {
        maxModifiedFiles: 5,
        maxEnvFacts: 3,
        maxToolSummaries: 4,
        maxFindings: 2,
      });

      expect(ss.getModifiedFiles().length).toBe(5);
      expect(ss.getEnvFacts().length).toBe(3);
      expect(ss.getToolSummaries().length).toBe(4);
      expect(ss.getFindings().length).toBe(2);

      // Should keep the most recent entries (tail)
      expect(ss.getModifiedFiles()[0]).toBe("/src/file-15.ts");
      expect(ss.getToolSummaries()[0]!.target).toBe("/src/f6.ts");
      expect(ss.getFindings()[0]!.title).toBe("Finding 8");
    });

    it("fromJSON rejects unknown version", () => {
      expect(() => SessionState.fromJSON({ version: 99 } as any)).toThrow(
        "Unsupported SessionState version: 99",
      );
    });

    it("serializes envFacts as key-value pairs", () => {
      state.addEnvFact("cmd-not-found:rg", "rg is not installed");
      state.addEnvFact("build-fail:cargo", "cargo fails");
      const json = state.toJSON();
      expect(json.envFacts).toEqual([
        { key: "cmd-not-found:rg", value: "rg is not installed" },
        { key: "build-fail:cargo", value: "cargo fails" },
      ]);
    });
  });

  // ─── Persistence Binding ────────────────────────────────────

  describe("disk persistence", () => {
    it("auto-saves on setPlan when bound", () => {
      const saved: SessionStateJSON[] = [];
      const persistence: SessionStatePersistence = {
        save: (_id, data) => { saved.push(structuredClone(data)); },
        load: () => null,
      };

      state.bind("test-session", persistence);
      state.setPlan([{ description: "Step 1", status: "pending" }]);

      expect(saved.length).toBe(1);
      expect(saved[0]!.plan).toEqual([{ description: "Step 1", status: "pending" }]);
    });

    it("auto-saves on recordModifiedFile when bound", () => {
      const saved: SessionStateJSON[] = [];
      const persistence: SessionStatePersistence = {
        save: (_id, data) => { saved.push(structuredClone(data)); },
        load: () => null,
      };

      state.bind("test-session", persistence);
      state.recordModifiedFile("/src/foo.ts");

      expect(saved.length).toBe(1);
      expect(saved[0]!.modifiedFiles).toEqual(["/src/foo.ts"]);
    });

    it("auto-saves on addEnvFact when bound", () => {
      const saved: SessionStateJSON[] = [];
      const persistence: SessionStatePersistence = {
        save: (_id, data) => { saved.push(structuredClone(data)); },
        load: () => null,
      };

      state.bind("test-session", persistence);
      state.addEnvFact("k", "v");

      expect(saved.length).toBe(1);
      expect(saved[0]!.envFacts).toEqual([{ key: "k", value: "v" }]);
    });

    it("auto-saves on addToolSummary when bound", () => {
      const saved: SessionStateJSON[] = [];
      const persistence: SessionStatePersistence = {
        save: (_id, data) => { saved.push(structuredClone(data)); },
        load: () => null,
      };

      state.bind("test-session", persistence);
      state.addToolSummary({
        tool: "edit_file",
        target: "/x.ts",
        summary: "test",
        iteration: 1,
      });

      expect(saved.length).toBe(1);
      expect(saved[0]!.toolSummaries.length).toBe(1);
    });

    it("does not save when persist is false even when bound", () => {
      const saved: SessionStateJSON[] = [];
      const persistence: SessionStatePersistence = {
        save: (_id, data) => { saved.push(structuredClone(data)); },
        load: () => null,
      };

      const ss = new SessionState({ persist: false });
      ss.bind("test-session", persistence);
      ss.setPlan([{ description: "Step 1", status: "pending" }]);
      ss.recordModifiedFile("/src/foo.ts");
      ss.addEnvFact("k", "v");
      ss.addToolSummary({ tool: "t", target: "t", summary: "s", iteration: 1 });
      ss.addFinding("Test", "Detail", 1);

      expect(saved.length).toBe(0);
    });

    it("does not save when not bound", () => {
      // No bind call — just verifying no error
      state.setPlan([{ description: "Step 1", status: "pending" }]);
      state.recordModifiedFile("/src/foo.ts");
      state.addEnvFact("k", "v");
      state.addToolSummary({ tool: "t", target: "t", summary: "s", iteration: 1 });
      // No assertion needed — just verifying no crash
    });

    it("loadOrCreate loads existing state from persistence", () => {
      const existing: SessionStateJSON = {
        version: 1,
        plan: [{ description: "Loaded", status: "completed" }],
        modifiedFiles: ["/x.ts"],
        envFacts: [{ key: "k", value: "v" }],
        toolSummaries: [{ tool: "t", target: "/x.ts", summary: "s", iteration: 1 }],
      };
      const persistence: SessionStatePersistence = {
        save: () => {},
        load: () => existing,
      };

      const ss = SessionState.loadOrCreate("test-session", persistence);
      expect(ss.getPlan()).toEqual([{ description: "Loaded", status: "completed" }]);
      expect(ss.getModifiedFiles()).toEqual(["/x.ts"]);
      expect(ss.getEnvFacts()).toEqual(["v"]);
      expect(ss.getToolSummaries().length).toBe(1);
    });

    it("loadOrCreate creates fresh state when no disk data", () => {
      const persistence: SessionStatePersistence = {
        save: () => {},
        load: () => null,
      };
      const ss = SessionState.loadOrCreate("test-session", persistence);
      expect(ss.getPlan()).toBeNull();
      expect(ss.getModifiedFiles()).toEqual([]);
    });

    it("loadOrCreate auto-binds for ongoing saves", () => {
      const saved: SessionStateJSON[] = [];
      const persistence: SessionStatePersistence = {
        save: (_id, data) => { saved.push(structuredClone(data)); },
        load: () => null,
      };

      const ss = SessionState.loadOrCreate("test-session", persistence);
      ss.setPlan([{ description: "New", status: "pending" }]);
      expect(saved.length).toBe(1);
    });
  });

  // ─── Findings ──────────────────────────────────────────────

  describe("findings tracking", () => {
    it("returns empty array when no findings stored", () => {
      expect(state.getFindings()).toEqual([]);
    });

    it("stores and retrieves a finding", () => {
      state.addFinding("Memory leak in auth handler", "The auth handler allocates a buffer on each request without releasing it.", 5);
      const findings = state.getFindings();
      expect(findings.length).toBe(1);
      expect(findings[0]!.title).toBe("Memory leak in auth handler");
      expect(findings[0]!.detail).toContain("allocates a buffer");
      expect(findings[0]!.iteration).toBe(5);
    });

    it("deduplicates by title, keeping latest", () => {
      state.addFinding("Bug in parser", "Off by one in line count", 3);
      state.addFinding("Other issue", "Type error", 4);
      state.addFinding("Bug in parser", "Actually off by two", 5);
      const findings = state.getFindings();
      expect(findings.length).toBe(2);
      // Deduplicated entry moves to end
      expect(findings[0]!.title).toBe("Other issue");
      expect(findings[1]!.title).toBe("Bug in parser");
      expect(findings[1]!.detail).toBe("Actually off by two");
      expect(findings[1]!.iteration).toBe(5);
    });

    it("truncates detail exceeding 500 chars", () => {
      const longDetail = "x".repeat(600);
      state.addFinding("Test", longDetail, 1);
      expect(state.getFindings()[0]!.detail.length).toBe(500);
    });

    it("respects max cap of 20", () => {
      for (let i = 0; i < 25; i++) {
        state.addFinding(`Finding ${i}`, `Detail ${i}`, i);
      }
      const findings = state.getFindings();
      expect(findings.length).toBe(20);
      expect(findings[0]!.title).toBe("Finding 5");
    });

    it("does not track when trackFindings is disabled", () => {
      const ss = new SessionState({ trackFindings: false });
      ss.addFinding("Test", "Detail", 1);
      expect(ss.getFindings()).toEqual([]);
    });

    it("auto-saves when bound", () => {
      const saved: SessionStateJSON[] = [];
      const persistence: SessionStatePersistence = {
        save: (_id, data) => { saved.push(structuredClone(data)); },
        load: () => null,
      };

      state.bind("test-session", persistence);
      state.addFinding("Test", "Detail", 1);
      expect(saved.length).toBe(1);
      expect(saved[0]!.findings!.length).toBe(1);
    });
  });

  // ─── Configurable Sections ──────────────────────────────────

  describe("configurable sections", () => {
    it("disables plan tracking when trackPlan is false", () => {
      const ss = new SessionState({ trackPlan: false });
      ss.setPlan([{ description: "Test", status: "pending" }]);
      expect(ss.getPlan()).toBeNull();
    });

    it("disables file tracking when trackFiles is false", () => {
      const ss = new SessionState({ trackFiles: false });
      ss.recordModifiedFile("/src/foo.ts");
      expect(ss.getModifiedFiles()).toEqual([]);
    });

    it("disables env fact tracking when trackEnv is false", () => {
      const ss = new SessionState({ trackEnv: false });
      ss.addEnvFact("k", "v");
      expect(ss.getEnvFacts()).toEqual([]);
    });

    it("respects custom maxModifiedFiles cap", () => {
      const ss = new SessionState({ maxModifiedFiles: 5 });
      for (let i = 0; i < 10; i++) {
        ss.recordModifiedFile(`/src/file-${i}.ts`);
      }
      expect(ss.getModifiedFiles().length).toBe(5);
      expect(ss.getModifiedFiles()[0]).toBe("/src/file-5.ts");
    });

    it("respects custom maxToolSummaries cap", () => {
      const ss = new SessionState({ maxToolSummaries: 3 });
      for (let i = 0; i < 5; i++) {
        ss.addToolSummary({
          tool: "edit_file",
          target: `/src/file-${i}.ts`,
          summary: `edit ${i}`,
          iteration: i,
        });
      }
      expect(ss.getToolSummaries().length).toBe(3);
    });
  });

  // ─── Deep-Clone Isolation ──────────────────────────────────
  describe("deep-clone isolation", () => {
    it("setPlan deep-clones input so external mutation does not affect internal state", () => {
      const steps: PlanStep[] = [
        { description: "Step 1", status: "pending" },
      ];
      state.setPlan(steps);
      // Mutate the original object
      (steps[0] as any).status = "completed";
      // Internal state should be unaffected
      expect(state.getPlan()![0]!.status).toBe("pending");
    });

    it("getPlan returns a readonly reference to internal state", () => {
      state.setPlan([{ description: "Step 1", status: "pending" }]);
      const plan = state.getPlan()!;
      // Returns the same reference (no clone) — callers must not mutate through ReadonlyArray.
      expect(plan).toBe(state.getPlan());
    });

    it("addToolSummary deep-clones input so external mutation does not affect internal state", () => {
      const summary = {
        tool: "edit_file",
        target: "/src/foo.ts",
        summary: "original",
        iteration: 1,
      } as ToolResultSummary;
      state.addToolSummary(summary);
      (summary as any).summary = "mutated";
      expect(state.getToolSummaries()[0]!.summary).toBe("original");
    });

    it("getToolSummaries returns a readonly reference to internal state", () => {
      state.addToolSummary({
        tool: "edit_file",
        target: "/src/foo.ts",
        summary: "original",
        iteration: 1,
      });
      const summaries = state.getToolSummaries();
      // Returns the same reference (no clone) — callers must not mutate through ReadonlyArray.
      expect(summaries).toBe(state.getToolSummaries());
    });

    it("fromJSON deep-clones restored objects so mutation of source data does not leak", () => {
      const data: SessionStateJSON = {
        version: 1,
        plan: [{ description: "Step 1", status: "pending" }],
        modifiedFiles: ["/foo.ts"],
        envFacts: [{ key: "k", value: "v" }],
        toolSummaries: [{ tool: "t", target: "/x.ts", summary: "s", iteration: 1 }],
        findings: [{ title: "Bug", detail: "Detail", iteration: 1 }],
      };
      const restored = SessionState.fromJSON(data);
      // Mutate the source data
      (data.plan![0] as any).status = "completed";
      (data.toolSummaries[0] as any).summary = "mutated";
      (data.findings![0] as any).detail = "mutated";
      // Restored state should be unaffected
      expect(restored.getPlan()![0]!.status).toBe("pending");
      expect(restored.getToolSummaries()[0]!.summary).toBe("s");
      expect(restored.getFindings()[0]!.detail).toBe("Detail");
    });
  });
});

describe("extractEnvFact", () => {
  it("detects 'command not found' pattern", () => {
    const fact = extractEnvFact(
      "run_command",
      "Command exited with code 127",
      "Exit code: 127\nstdout: \nstderr: sh: rg: command not found",
    );
    expect(fact).not.toBeNull();
    expect(fact!.key).toBe("cmd-not-found:rg");
    expect(fact!.message).toContain("rg");
    expect(fact!.message).toContain("not installed");
  });

  it("detects 'permission denied' pattern", () => {
    const fact = extractEnvFact(
      "run_command",
      "Command exited with code 1",
      "Exit code: 1\nstdout: \nstderr: /usr/bin/foo: Permission denied",
    );
    expect(fact).not.toBeNull();
    expect(fact!.key).toContain("permission-denied");
  });

  it("returns null for generic failures", () => {
    const fact = extractEnvFact(
      "run_command",
      "Command exited with code 1",
      "Exit code: 1\nstdout: no matches found\nstderr: ",
    );
    expect(fact).toBeNull();
  });

  it("detects cargo/build failures with missing deps", () => {
    const fact = extractEnvFact(
      "run_command",
      "Command exited with code 101",
      "Exit code: 101\nstdout: \nstderr: error[E0463]: can't find crate for `ani_sys`",
    );
    expect(fact).not.toBeNull();
    expect(fact!.key).toContain("build-fail");
  });

  it("returns null for non-run_command tools", () => {
    const fact = extractEnvFact(
      "search_files",
      "Tool error: no results",
      "",
    );
    expect(fact).toBeNull();
  });

  // ─── Enhanced patterns (Task 7) ─────────────────────────────

  it("detects network failure pattern", () => {
    const fact = extractEnvFact(
      "run_command",
      "Command exited with code 1",
      "curl: (6) Could not resolve host: api.example.com",
    );
    expect(fact).not.toBeNull();
    expect(fact!.key).toBe("network-failure");
    expect(fact!.message).toContain("Network");
  });

  it("detects ECONNREFUSED pattern", () => {
    const fact = extractEnvFact(
      "run_command",
      "Command exited with code 1",
      "Error: connect ECONNREFUSED 127.0.0.1:3000",
    );
    expect(fact).not.toBeNull();
    expect(fact!.key).toBe("network-failure");
  });

  it("detects disk full pattern", () => {
    const fact = extractEnvFact(
      "run_command",
      "Command exited with code 1",
      "write error: No space left on device",
    );
    expect(fact).not.toBeNull();
    expect(fact!.key).toBe("disk-full");
  });

  it("detects Node.js version mismatch pattern", () => {
    const fact = extractEnvFact(
      "run_command",
      "Command exited with code 1",
      "error: package requires Node.js 18 or higher",
    );
    expect(fact).not.toBeNull();
    expect(fact!.key).toBe("version-mismatch");
  });

  it("detects git state issue", () => {
    const fact = extractEnvFact(
      "run_command",
      "Command exited with code 128",
      "fatal: not a git repository (or any of the parent directories): .git",
    );
    expect(fact).not.toBeNull();
    expect(fact!.key).toBe("git-issue");
  });

  it("detects timeout pattern", () => {
    const fact = extractEnvFact(
      "run_command",
      "Command timed out",
      "Process received SIGTERM",
    );
    expect(fact).not.toBeNull();
    expect(fact!.key).toBe("tool-timeout");
  });
});
