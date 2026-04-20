import { describe, it, expect, vi } from "vitest";

import { SessionState } from "./session-state.js";
import { StagnationDetector } from "./stagnation-detector.js";
import { EventBus, MessageRole } from "../core/index.js";
import type { LLMProvider, Message, StreamChunk } from "../core/index.js";

// ─── Mock helpers ────────────────────────────────────────────

/** Create a mock LLMProvider that yields the given JSON string as text chunks. */
function mockProvider(responseJson: string): LLMProvider {
  return {
    id: "mock",
    chat: vi.fn(async function* (): AsyncIterable<StreamChunk> {
      yield { type: "text", content: responseJson };
      yield { type: "done", content: "" };
    }),
    abort: vi.fn(),
  };
}

/** Create a mock LLMProvider that throws an error. */
function throwingProvider(): LLMProvider {
  return {
    id: "mock-error",
    chat: vi.fn(async function* (): AsyncIterable<StreamChunk> {
      throw new Error("Provider error");
    }),
    abort: vi.fn(),
  };
}

function makeDetector(sessionState: SessionState | null = null) {
  const bus = new EventBus();
  const detector = new StagnationDetector({ bus, sessionState });
  return { detector, bus };
}

/** Build a minimal messages array with a USER message. */
function makeMessages(count = 5): Message[] {
  const msgs: Message[] = [
    { role: MessageRole.USER, content: "Fix the login bug" },
  ];
  for (let i = 1; i < count; i++) {
    msgs.push({
      role: MessageRole.ASSISTANT,
      content: `Searching for files... attempt ${i}`,
    });
  }
  return msgs;
}

/** Build messages with tool calls and tool results for richer context testing. */
function makeMessagesWithToolCalls(): Message[] {
  return [
    { role: MessageRole.USER, content: "Fix the login bug" },
    {
      role: MessageRole.ASSISTANT,
      content: null,
      toolCalls: [
        { name: "search_files", arguments: { pattern: "login", path: "/src" }, callId: "tc1" },
      ],
    },
    {
      role: MessageRole.TOOL,
      content: "Found 3 matches in src/auth/login.ts",
      toolCallId: "tc1",
    },
    {
      role: MessageRole.ASSISTANT,
      content: null,
      toolCalls: [
        { name: "read_file", arguments: { path: "/src/auth/login.ts" }, callId: "tc2" },
      ],
    },
    {
      role: MessageRole.TOOL,
      content: "export function login(user: string, pass: string) { ... }",
      toolCallId: "tc2",
    },
  ];
}

// ─── Tests ───────────────────────────────────────────────────

describe("StagnationDetector — LLM-as-judge", () => {
  it("skips judge below MIN_JUDGE_ITERATION", async () => {
    const { detector } = makeDetector();
    const provider = mockProvider('{"analysis":"ok","stagnation_confidence":0.9}');
    const messages = makeMessages();

    const result = await detector.checkStagnationWithLLM(provider, messages, 5);
    expect(result).toBeNull();
    expect(provider.chat).not.toHaveBeenCalled();
  });

  it("skips judge within interval", async () => {
    const { detector } = makeDetector();
    const provider = mockProvider('{"analysis":"no stagnation","stagnation_confidence":0.1}');
    const messages = makeMessages();

    // First call at iteration 15 — fires (within gating rules)
    const first = await detector.checkStagnationWithLLM(provider, messages, 15);
    expect(first).toBeNull(); // confidence 0.1 < threshold
    expect(provider.chat).toHaveBeenCalledTimes(1);

    // Second call at iteration 16 — skipped (within interval)
    const second = await detector.checkStagnationWithLLM(provider, messages, 16);
    expect(second).toBeNull();
    expect(provider.chat).toHaveBeenCalledTimes(1); // not called again
  });

  it("returns nudge when confidence exceeds threshold", async () => {
    const { detector } = makeDetector();
    const provider = mockProvider(
      '{"analysis":"Agent is repeating search patterns without making edits","stagnation_confidence":0.92}',
    );
    const messages = makeMessages();

    const result = await detector.checkStagnationWithLLM(provider, messages, 15);
    expect(result).not.toBeNull();
    expect(result).toContain("STAGNATION DETECTED");
    expect(result).toContain("repeating search patterns");
    expect(result).toContain("0.92");
  });

  it("returns null when confidence below threshold", async () => {
    const { detector } = makeDetector();
    const provider = mockProvider(
      '{"analysis":"Agent is exploring codebase","stagnation_confidence":0.3}',
    );
    const messages = makeMessages();

    const result = await detector.checkStagnationWithLLM(provider, messages, 15);
    expect(result).toBeNull();
  });

  it("returns null on provider error", async () => {
    const { detector } = makeDetector();
    const provider = throwingProvider();
    const messages = makeMessages();

    const result = await detector.checkStagnationWithLLM(provider, messages, 15);
    expect(result).toBeNull();
  });

  it("adapts interval based on confidence", async () => {
    const { detector } = makeDetector();
    const messages = makeMessages();

    // High confidence (0.7+) → shorter interval (MIN_JUDGE_INTERVAL = 5)
    const highProvider = mockProvider('{"analysis":"stagnating","stagnation_confidence":0.75}');
    await detector.checkStagnationWithLLM(highProvider, messages, 15);

    // Next call at iteration 20 (15 + 5 = 20) should fire
    const midProvider = mockProvider('{"analysis":"ok","stagnation_confidence":0.2}');
    const atShortInterval = await detector.checkStagnationWithLLM(midProvider, messages, 20);
    expect(midProvider.chat).toHaveBeenCalledTimes(1); // called — interval was 5

    // Low confidence (0.2) → longer interval (MAX_JUDGE_INTERVAL = 12)
    // Next call at iteration 25 (20 + 5) should NOT fire — interval is now 12
    const skipProvider = mockProvider('{"analysis":"ok","stagnation_confidence":0.1}');
    const tooSoon = await detector.checkStagnationWithLLM(skipProvider, messages, 25);
    expect(tooSoon).toBeNull();
    expect(skipProvider.chat).not.toHaveBeenCalled(); // skipped — 25-20=5 < 12

    // But at iteration 32 (20 + 12) it should fire
    const fireProvider = mockProvider('{"analysis":"ok","stagnation_confidence":0.1}');
    await detector.checkStagnationWithLLM(fireProvider, messages, 32);
    expect(fireProvider.chat).toHaveBeenCalledTimes(1);
  });

  it("includes session state context in judge call", async () => {
    const state = new SessionState();
    state.setPlan([
      { description: "Step 1", status: "completed" },
      { description: "Step 2", status: "in_progress" },
    ]);
    state.recordModifiedFile("src/foo.ts");
    state.addFinding("Bug found", "null pointer in foo()", 3);

    const { detector } = makeDetector(state);
    const provider = mockProvider('{"analysis":"ok","stagnation_confidence":0.1}');
    const messages = makeMessages();

    await detector.checkStagnationWithLLM(provider, messages, 15);

    // Verify the messages passed to provider contain session state context
    const chatCall = (provider.chat as ReturnType<typeof vi.fn>).mock.calls[0]!;
    const judgeMessages = chatCall[0] as Message[];
    const userMsg = judgeMessages.find((m) => m.role === MessageRole.USER);
    expect(userMsg).toBeDefined();
    expect(userMsg!.content).toContain("Plan progress: 1/2 steps completed");
    expect(userMsg!.content).toContain("Modified files: 1");
    expect(userMsg!.content).toContain("Findings: 1");
  });

  it("resetRunState resets judge state", async () => {
    const { detector } = makeDetector();
    const provider = mockProvider('{"analysis":"ok","stagnation_confidence":0.1}');
    const messages = makeMessages();

    // Fire at iteration 15
    await detector.checkStagnationWithLLM(provider, messages, 15);
    expect(provider.chat).toHaveBeenCalledTimes(1);

    // Reset
    detector.resetRunState();

    // Should fire again at iteration 15 after reset (lastJudgeIteration was reset to 0)
    const provider2 = mockProvider('{"analysis":"ok","stagnation_confidence":0.1}');
    await detector.checkStagnationWithLLM(provider2, messages, 15);
    expect(provider2.chat).toHaveBeenCalledTimes(1);
  });

  it("includes tool call names and arguments in judge context", async () => {
    const { detector } = makeDetector();
    const provider = mockProvider('{"analysis":"ok","stagnation_confidence":0.1}');
    const messages = makeMessagesWithToolCalls();

    await detector.checkStagnationWithLLM(provider, messages, 15);

    const chatCall = (provider.chat as ReturnType<typeof vi.fn>).mock.calls[0]!;
    const judgeMessages = chatCall[0] as Message[];
    const userMsg = judgeMessages.find((m) => m.role === MessageRole.USER);
    expect(userMsg).toBeDefined();
    // Verify tool call names and args are present
    expect(userMsg!.content).toContain("tool_call: search_files");
    expect(userMsg!.content).toContain("pattern=login");
    expect(userMsg!.content).toContain("path=/src");
    expect(userMsg!.content).toContain("tool_call: read_file");
    expect(userMsg!.content).toContain("path=/src/auth/login.ts");
    // Verify tool results are present
    expect(userMsg!.content).toContain("Found 3 matches");
    expect(userMsg!.content).toContain("tool_result [tc1]");
    expect(userMsg!.content).toContain("tool_result [tc2]");
  });

  it("emits LLM_STAGNATION_DETECTED error event when stagnation detected", async () => {
    const { detector, bus } = makeDetector();
    const provider = mockProvider(
      '{"analysis":"stuck in search loop","stagnation_confidence":0.95}',
    );
    const messages = makeMessages();

    const errors: Array<{ code: string }> = [];
    bus.on("error", (e) => errors.push(e as { code: string }));

    await detector.checkStagnationWithLLM(provider, messages, 15);

    expect(errors).toHaveLength(1);
    expect(errors[0]!.code).toBe("LLM_STAGNATION_DETECTED");
  });
});
