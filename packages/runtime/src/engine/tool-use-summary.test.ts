import { describe, it, expect, beforeEach } from "vitest";
import { ToolUseSummaryGenerator, DEFAULT_TOOL_USE_SUMMARY_INTERVAL, TOOL_USE_SUMMARY_MARKER } from "./tool-use-summary.js";
import { SessionState } from "./session-state.js";
import { MessageRole } from "../core/index.js";
import type { Message } from "../core/index.js";

// ─── Helpers ────────────────────────────────────────────────

/** Build messages with tool calls and results. */
function makeToolMessages(): Message[] {
  return [
    { role: MessageRole.USER, content: "Fix the bug" },
    {
      role: MessageRole.ASSISTANT,
      content: null,
      toolCalls: [
        { name: "read_file", arguments: { path: "/src/foo.ts" }, callId: "tc1" },
        { name: "search_files", arguments: { pattern: "bug" }, callId: "tc2" },
      ],
    },
    { role: MessageRole.TOOL, content: "file contents here", toolCallId: "tc1" },
    { role: MessageRole.TOOL, content: "found 2 matches", toolCallId: "tc2" },
    {
      role: MessageRole.ASSISTANT,
      content: null,
      toolCalls: [
        { name: "write_file", arguments: { path: "/src/foo.ts" }, callId: "tc3" },
      ],
    },
    { role: MessageRole.TOOL, content: "file written", toolCallId: "tc3" },
    {
      role: MessageRole.ASSISTANT,
      content: null,
      toolCalls: [
        { name: "read_file", arguments: { path: "/src/bar.ts" }, callId: "tc4" },
      ],
    },
    { role: MessageRole.TOOL, content: "Error: file not found", toolCallId: "tc4" },
  ];
}

// ─── Tests ──────────────────────────────────────────────────

describe("ToolUseSummaryGenerator", () => {
  let generator: ToolUseSummaryGenerator;

  beforeEach(() => {
    generator = new ToolUseSummaryGenerator();
  });

  it("returns null before interval is reached", () => {
    const messages = makeToolMessages();
    const result = generator.maybeSummarize(5, messages, null);
    expect(result).toBeNull();
  });

  it("generates summary at the correct interval", () => {
    const messages = makeToolMessages();

    // At iteration 10 (DEFAULT_TOOL_USE_SUMMARY_INTERVAL), should generate
    const result = generator.maybeSummarize(DEFAULT_TOOL_USE_SUMMARY_INTERVAL, messages, null);
    expect(result).not.toBeNull();
    expect(result!.role).toBe(MessageRole.SYSTEM);
    expect(result!.content).toContain(TOOL_USE_SUMMARY_MARKER);
  });

  it("returns null again until next interval after generating", () => {
    const messages = makeToolMessages();

    // First generation at interval
    generator.maybeSummarize(DEFAULT_TOOL_USE_SUMMARY_INTERVAL, messages, null);

    // Too soon — should return null
    const result = generator.maybeSummarize(DEFAULT_TOOL_USE_SUMMARY_INTERVAL + 5, messages, null);
    expect(result).toBeNull();

    // At next interval boundary — should generate again
    const result2 = generator.maybeSummarize(DEFAULT_TOOL_USE_SUMMARY_INTERVAL * 2, messages, null);
    expect(result2).not.toBeNull();
  });

  it("includes tool usage counts in summary", () => {
    const messages = makeToolMessages();
    const result = generator.maybeSummarize(DEFAULT_TOOL_USE_SUMMARY_INTERVAL, messages, null);

    expect(result).not.toBeNull();
    const content = result!.content!;
    expect(content).toContain("read_file: 2 calls");
    expect(content).toContain("write_file: 1 calls");
    expect(content).toContain("search_files: 1 calls");
  });

  it("includes files modified and read", () => {
    const messages = makeToolMessages();
    const result = generator.maybeSummarize(DEFAULT_TOOL_USE_SUMMARY_INTERVAL, messages, null);

    expect(result).not.toBeNull();
    const content = result!.content!;
    expect(content).toContain("Files modified");
    expect(content).toContain("/src/foo.ts");
    expect(content).toContain("Files read");
    expect(content).toContain("/src/bar.ts");
  });

  it("includes plan progress from session state", () => {
    const state = new SessionState();
    state.setPlan([
      { description: "Step 1", status: "completed" },
      { description: "Step 2", status: "in_progress" },
      { description: "Step 3", status: "pending" },
    ]);

    const messages = makeToolMessages();
    const result = generator.maybeSummarize(DEFAULT_TOOL_USE_SUMMARY_INTERVAL, messages, state);

    expect(result).not.toBeNull();
    const content = result!.content!;
    expect(content).toContain("Plan progress");
    expect(content).toContain("1/3 completed");
    expect(content).toContain("1 in progress");
    expect(content).toContain("1 pending");
  });

  it("reset clears the interval counter", () => {
    const messages = makeToolMessages();

    // Generate at interval
    generator.maybeSummarize(DEFAULT_TOOL_USE_SUMMARY_INTERVAL, messages, null);

    // Reset
    generator.reset();

    // Should generate again at the same interval (since counter was reset to 0)
    const result = generator.maybeSummarize(DEFAULT_TOOL_USE_SUMMARY_INTERVAL, messages, null);
    expect(result).not.toBeNull();
  });

  it("is disabled when interval is 0", () => {
    const disabled = new ToolUseSummaryGenerator({ interval: 0 });
    const messages = makeToolMessages();

    const result = disabled.maybeSummarize(100, messages, null);
    expect(result).toBeNull();
  });

  it("returns null when there are no tool calls in messages", () => {
    const messages: Message[] = [
      { role: MessageRole.USER, content: "Hello" },
      { role: MessageRole.ASSISTANT, content: "Hi there" },
    ];

    const result = generator.maybeSummarize(DEFAULT_TOOL_USE_SUMMARY_INTERVAL, messages, null);
    expect(result).toBeNull();
  });

  it("includes tool error counts", () => {
    const messages = makeToolMessages();
    const result = generator.maybeSummarize(DEFAULT_TOOL_USE_SUMMARY_INTERVAL, messages, null);

    expect(result).not.toBeNull();
    const content = result!.content!;
    // read_file tc4 had an "Error:" response
    expect(content).toContain("read_file: 2 calls (1 errors)");
  });
});
