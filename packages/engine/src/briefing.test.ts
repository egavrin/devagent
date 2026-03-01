import { describe, it, expect } from "vitest";
import {
  synthesizeBriefing,
  extractHeuristicBriefing,
  formatBriefing,
} from "./briefing.js";
import type { TurnBriefing } from "./briefing.js";
import type { Message, LLMProvider, StreamChunk } from "@devagent/core";
import { MessageRole } from "@devagent/core";

// ─── Helpers ────────────────────────────────────────────────

function makeMessages(
  ...items: Array<
    | { role: "system"; content: string }
    | { role: "user"; content: string }
    | { role: "assistant"; content: string }
    | {
        role: "assistant";
        content: string;
        toolCalls: Array<{ name: string; arguments: Record<string, unknown>; callId: string }>;
      }
    | { role: "tool"; content: string; toolCallId: string }
  >
): Message[] {
  return items.map((item) => {
    if ("toolCalls" in item) {
      return {
        role: item.role as MessageRole,
        content: item.content,
        toolCalls: item.toolCalls,
      };
    }
    if ("toolCallId" in item) {
      return {
        role: item.role as MessageRole,
        content: item.content,
        toolCallId: item.toolCallId,
      };
    }
    return {
      role: item.role as MessageRole,
      content: item.content,
    };
  });
}

function makeMockProvider(response: string): LLMProvider {
  return {
    id: "mock",
    chat: () => {
      async function* generate(): AsyncIterable<StreamChunk> {
        yield { type: "text", content: response };
        yield { type: "done", content: "" };
      }
      return generate();
    },
    abort: () => {},
  };
}

// ─── Tests ──────────────────────────────────────────────────

describe("briefing", () => {
  describe("heuristic strategy", () => {
    it("extracts user query and final response", () => {
      const messages = makeMessages(
        { role: "system", content: "You are a helpful agent." },
        { role: "user", content: "Find all TypeScript files in src/" },
        {
          role: "assistant",
          content: "I found 15 TypeScript files in src/.",
        },
      );

      const briefing = extractHeuristicBriefing(messages, 1);

      expect(briefing.turnNumber).toBe(1);
      expect(briefing.priorTaskSummary).toContain(
        "Find all TypeScript files in src/",
      );
      expect(briefing.priorTaskSummary).toContain(
        "I found 15 TypeScript files",
      );
    });

    it("collects file paths from tool calls", () => {
      const messages = makeMessages(
        { role: "system", content: "System prompt" },
        { role: "user", content: "Read the config files" },
        {
          role: "assistant",
          content: "",
          toolCalls: [
            {
              name: "read_file",
              arguments: { path: "src/config.ts" },
              callId: "c1",
            },
            {
              name: "read_file",
              arguments: { path: "src/types.ts" },
              callId: "c2",
            },
          ],
        },
        { role: "tool", content: "file content 1", toolCallId: "c1" },
        { role: "tool", content: "file content 2", toolCallId: "c2" },
        { role: "assistant", content: "Both files look good." },
      );

      const briefing = extractHeuristicBriefing(messages, 2);

      expect(briefing.keyArtifacts).toContain("src/config.ts");
      expect(briefing.keyArtifacts).toContain("src/types.ts");
      expect(briefing.priorTaskSummary).toContain("read_file(x2)");
    });

    it("handles empty messages gracefully", () => {
      const briefing = extractHeuristicBriefing([], 1);

      expect(briefing.turnNumber).toBe(1);
      expect(briefing.priorTaskSummary).toBe("");
      expect(briefing.activeContext).toBe("");
      expect(briefing.pendingWork).toBeNull();
      expect(briefing.keyArtifacts).toEqual([]);
    });

    it("caps output at token budget", () => {
      // Create a message with very long content
      const longContent = "x".repeat(10000);
      const messages = makeMessages(
        { role: "user", content: longContent },
        { role: "assistant", content: longContent },
      );

      const briefing = extractHeuristicBriefing(messages, 1, 500);

      // Total briefing text should be under the budget
      const totalChars =
        briefing.priorTaskSummary.length +
        briefing.activeContext.length +
        (briefing.pendingWork?.length ?? 0);
      expect(totalChars).toBeLessThan(800); // Some overhead from labels
    });

    it("extracts plan steps from update_plan tool calls", () => {
      const messages = makeMessages(
        { role: "system", content: "System prompt" },
        { role: "user", content: "Refactor the auth module" },
        {
          role: "assistant",
          content: "",
          toolCalls: [
            {
              name: "update_plan",
              arguments: {
                steps: [
                  { description: "Extract interfaces", status: "completed" },
                  { description: "Update imports", status: "in_progress" },
                  { description: "Run tests", status: "pending" },
                ],
              },
              callId: "c1",
            },
          ],
        },
        { role: "tool", content: "Plan updated", toolCallId: "c1" },
        { role: "assistant", content: "Phase 1 complete." },
      );

      const briefing = extractHeuristicBriefing(messages, 1);

      expect(briefing.priorTaskSummary).toContain("Extract interfaces");
      expect(briefing.priorTaskSummary).toContain("completed");
      expect(briefing.priorTaskSummary).toContain("in_progress");
    });

    it("captures tool errors in activeContext", () => {
      const messages = makeMessages(
        { role: "system", content: "System prompt" },
        { role: "user", content: "Build the project" },
        {
          role: "assistant",
          content: "",
          toolCalls: [
            {
              name: "run_command",
              arguments: { command: "bun run build" },
              callId: "c1",
            },
          ],
        },
        {
          role: "tool",
          content: "Error: Cannot find module 'missing-dep'",
          toolCallId: "c1",
        },
        { role: "assistant", content: "Build failed due to missing dep." },
      );

      const briefing = extractHeuristicBriefing(messages, 1);

      expect(briefing.activeContext).toContain("missing-dep");
    });

    it("detects pending work from assistant response", () => {
      const messages = makeMessages(
        { role: "system", content: "System prompt" },
        { role: "user", content: "Fix all bugs" },
        {
          role: "assistant",
          content:
            "I fixed the null pointer bug. We still need to address the race condition in the worker thread.",
        },
      );

      const briefing = extractHeuristicBriefing(messages, 1);

      expect(briefing.pendingWork).not.toBeNull();
      expect(briefing.pendingWork).toContain("still need");
    });

    it("extracts plan from update_plan with string steps argument", () => {
      const messages = makeMessages(
        { role: "system", content: "System prompt." },
        { role: "user", content: "Do the task." },
        {
          role: "assistant",
          content: "",
          toolCalls: [{
            name: "update_plan",
            arguments: {
              steps: JSON.stringify([
                { description: "Step 1", status: "completed" },
                { description: "Step 2", status: "in_progress" },
              ]),
            },
            callId: "plan-1",
          }],
        },
        { role: "tool", content: "Plan updated (1/2 completed):\n[x] Step 1\n[>] Step 2", toolCallId: "plan-1" },
      );

      const briefing = extractHeuristicBriefing(messages, 1);
      expect(briefing.planSteps).not.toBeNull();
      expect(briefing.planSteps!).toHaveLength(2);
      expect(briefing.planSteps![0]!.status).toBe("completed");
      expect(briefing.planSteps![1]!.status).toBe("in_progress");
    });
  });

  describe("auto strategy", () => {
    it("uses heuristic when fewer than 5 tool calls", async () => {
      const messages = makeMessages(
        { role: "user", content: "Simple question" },
        { role: "assistant", content: "Simple answer" },
      );

      const briefing = await synthesizeBriefing(messages, 1, {
        strategy: "auto",
      });

      expect(briefing.turnNumber).toBe(1);
      expect(briefing.priorTaskSummary).toContain("Simple question");
    });

    it("uses LLM when 5+ tool calls and provider available", async () => {
      const messages = makeMessages(
        { role: "user", content: "Explore the codebase" },
        {
          role: "assistant",
          content: "",
          toolCalls: [
            { name: "find_files", arguments: {}, callId: "c1" },
            { name: "read_file", arguments: { path: "a.ts" }, callId: "c2" },
            { name: "read_file", arguments: { path: "b.ts" }, callId: "c3" },
            { name: "search_files", arguments: {}, callId: "c4" },
            { name: "read_file", arguments: { path: "c.ts" }, callId: "c5" },
          ],
        },
        { role: "tool", content: "files found", toolCallId: "c1" },
        { role: "tool", content: "content a", toolCallId: "c2" },
        { role: "tool", content: "content b", toolCallId: "c3" },
        { role: "tool", content: "search results", toolCallId: "c4" },
        { role: "tool", content: "content c", toolCallId: "c5" },
        { role: "assistant", content: "Analysis complete." },
      );

      const mockProvider = makeMockProvider(
        `## Goal
Explore and understand the codebase structure.

## Key Decisions & Constraints
- Uses TypeScript
- Monorepo structure

## Accomplished
- Read 3 files: a.ts, b.ts, c.ts
- Searched for patterns

## Pending
Nothing pending.

## Relevant Files
\`a.ts\` — read
\`b.ts\` — read
\`c.ts\` — read`,
      );

      const briefing = await synthesizeBriefing(messages, 1, {
        strategy: "auto",
        provider: mockProvider,
      });

      expect(briefing.turnNumber).toBe(1);
      expect(briefing.priorTaskSummary).toContain("Goal");
      expect(briefing.priorTaskSummary).toContain("Explore");
      expect(briefing.pendingWork).toBeNull(); // "Nothing pending"
    });
  });

  describe("formatBriefing", () => {
    it("formats a briefing into a readable string", () => {
      const briefing: TurnBriefing = {
        turnNumber: 3,
        priorTaskSummary: "Explored the project structure and identified key files.",
        activeContext: "Project uses TypeScript with Bun runtime.",
        pendingWork: "Still need to update the imports.",
        keyArtifacts: ["src/main.ts", "src/config.ts"],
        planSteps: null,
      };

      const formatted = formatBriefing(briefing);

      expect(formatted).toContain("Turn: 3");
      expect(formatted).toContain("Explored the project structure");
      expect(formatted).toContain("TypeScript with Bun");
      expect(formatted).toContain("update the imports");
      expect(formatted).toContain("src/main.ts");
      expect(formatted).toContain("src/config.ts");
      expect(formatted).not.toContain("Plan status:");
    });

    it("includes plan status section when planSteps present", () => {
      const briefing: TurnBriefing = {
        turnNumber: 2,
        priorTaskSummary: "Implemented feature A.",
        activeContext: "",
        pendingWork: null,
        keyArtifacts: [],
        planSteps: [
          { description: "Write tests", status: "completed" },
          { description: "Update docs", status: "in_progress" },
          { description: "Deploy", status: "pending" },
        ],
      };

      const formatted = formatBriefing(briefing);

      expect(formatted).toContain("Plan status:");
      expect(formatted).toContain("- [completed] Write tests");
      expect(formatted).toContain("- [in_progress] Update docs");
      expect(formatted).toContain("- [pending] Deploy");
    });
  });

  // ─── Plan Step Extraction ──────────────────────────────────

  describe("plan step extraction", () => {
    it("extracts plan steps as structured array from last update_plan", () => {
      const messages = makeMessages(
        { role: "system", content: "System prompt" },
        { role: "user", content: "Implement the feature" },
        {
          role: "assistant",
          content: "",
          toolCalls: [
            {
              name: "update_plan",
              arguments: {
                steps: [
                  { description: "Design API", status: "pending" },
                  { description: "Write code", status: "pending" },
                ],
              },
              callId: "c1",
            },
          ],
        },
        { role: "tool", content: "Plan updated", toolCallId: "c1" },
        {
          role: "assistant",
          content: "",
          toolCalls: [
            {
              name: "update_plan",
              arguments: {
                steps: [
                  { description: "Design API", status: "completed" },
                  { description: "Write code", status: "in_progress" },
                  { description: "Run tests", status: "pending" },
                ],
              },
              callId: "c2",
            },
          ],
        },
        { role: "tool", content: "Plan updated", toolCallId: "c2" },
        { role: "assistant", content: "Working on the code now." },
      );

      const briefing = extractHeuristicBriefing(messages, 1);

      // Should use the LAST plan, not the first
      expect(briefing.planSteps).not.toBeNull();
      expect(briefing.planSteps).toHaveLength(3);
      expect(briefing.planSteps![0]!.description).toBe("Design API");
      expect(briefing.planSteps![0]!.status).toBe("completed");
      expect(briefing.planSteps![1]!.status).toBe("in_progress");
      expect(briefing.planSteps![2]!.description).toBe("Run tests");
      expect(briefing.planSteps![2]!.status).toBe("pending");

      // Plan should also appear in priorTaskSummary
      expect(briefing.priorTaskSummary).toContain("[completed] Design API");
      expect(briefing.priorTaskSummary).toContain("[in_progress] Write code");
    });

    it("returns null planSteps when no update_plan calls exist", () => {
      const messages = makeMessages(
        { role: "user", content: "Quick question" },
        { role: "assistant", content: "Quick answer" },
      );

      const briefing = extractHeuristicBriefing(messages, 1);

      expect(briefing.planSteps).toBeNull();
    });

    it("skips denied update_plan calls and extracts only successful ones", () => {
      const messages = makeMessages(
        { role: "user", content: "Plan the work" },
        {
          role: "assistant",
          content: "",
          toolCalls: [
            {
              name: "update_plan",
              arguments: {
                steps: [
                  { description: "Denied step", status: "pending" },
                ],
              },
              callId: "denied-1",
            },
          ],
        },
        {
          role: "tool",
          content: "Error: Tool execution denied: approval mode requires manual confirmation",
          toolCallId: "denied-1",
        },
        {
          role: "assistant",
          content: "",
          toolCalls: [
            {
              name: "update_plan",
              arguments: {
                steps: [
                  { description: "Approved step", status: "in_progress" },
                ],
              },
              callId: "ok-1",
            },
          ],
        },
        { role: "tool", content: "Plan updated (0/1 completed)", toolCallId: "ok-1" },
      );

      const briefing = extractHeuristicBriefing(messages, 1);

      expect(briefing.planSteps).not.toBeNull();
      expect(briefing.planSteps).toHaveLength(1);
      expect(briefing.planSteps![0]!.description).toBe("Approved step");
    });

    it("returns null planSteps when all update_plan calls were denied", () => {
      const messages = makeMessages(
        { role: "user", content: "Plan the work" },
        {
          role: "assistant",
          content: "",
          toolCalls: [
            {
              name: "update_plan",
              arguments: {
                steps: [
                  { description: "Denied step", status: "pending" },
                ],
              },
              callId: "denied-1",
            },
          ],
        },
        {
          role: "tool",
          content: "Error: Tool execution denied: user rejected",
          toolCallId: "denied-1",
        },
      );

      const briefing = extractHeuristicBriefing(messages, 1);

      expect(briefing.planSteps).toBeNull();
    });

    it("handles single update_plan call correctly", () => {
      const messages = makeMessages(
        { role: "user", content: "Plan the work" },
        {
          role: "assistant",
          content: "",
          toolCalls: [
            {
              name: "update_plan",
              arguments: {
                steps: [
                  { description: "Step A", status: "completed" },
                  { description: "Step B", status: "pending" },
                ],
              },
              callId: "c1",
            },
          ],
        },
        { role: "tool", content: "Plan updated", toolCallId: "c1" },
      );

      const briefing = extractHeuristicBriefing(messages, 1);

      expect(briefing.planSteps).toHaveLength(2);
      expect(briefing.planSteps![0]!.status).toBe("completed");
      expect(briefing.planSteps![1]!.status).toBe("pending");
    });
  });
});
