import { describe, it, expect } from "vitest";
import { extractPreCompactionKnowledge, KNOWLEDGE_EXTRACTION_SYSTEM_PROMPT } from "./knowledge-extractor.js";
import type { KnowledgeExtractionResult } from "./knowledge-extractor.js";
import type { LLMProvider, StreamChunk, Message } from "../core/index.js";
import { MessageRole } from "../core/index.js";
import { SessionState } from "./session-state.js";

function createMockProvider(responseText: string): LLMProvider {
  return {
    id: "mock",
    async *chat(): AsyncIterable<StreamChunk> {
      yield { type: "text", content: responseText };
      yield { type: "done", content: "" };
    },
    abort() {},
  };
}

function createFailingProvider(): LLMProvider {
  return {
    id: "mock",
    async *chat(): AsyncIterable<StreamChunk> {
      throw new Error("Provider unavailable");
    },
    abort() {},
  };
}

const VALID_RESPONSE = JSON.stringify({
  entries: [
    { key: "inventory", content: "Found 5 ANI descriptor files in src/ani/" },
    { key: "decisions", content: "Using visitor pattern for AST transformation" },
    { key: "progress", content: "3/7 files transformed, 4 remaining" },
    { key: "next_action", content: "Transform src/ani/type_resolver.cpp next" },
  ],
});

function makeMessages(count: number): Message[] {
  const msgs: Message[] = [];
  for (let i = 0; i < count; i++) {
    msgs.push({
      role: i % 2 === 0 ? MessageRole.USER : MessageRole.ASSISTANT,
      content: `Message ${i}`,
    });
  }
  return msgs;
}

describe("extractPreCompactionKnowledge", () => {
  it("returns null on provider error (graceful degradation)", async () => {
    const provider = createFailingProvider();
    const result = await extractPreCompactionKnowledge(
      provider,
      "summary text",
      null,
      [],
      "original task",
    );
    expect(result).toBeNull();
  });

  it("returns null on invalid JSON response", async () => {
    const provider = createMockProvider("This is not JSON at all");
    const result = await extractPreCompactionKnowledge(
      provider,
      "summary text",
      null,
      [],
      "original task",
    );
    expect(result).toBeNull();
  });

  it("returns null when response has no entries array", async () => {
    const provider = createMockProvider('{"quality": "good"}');
    const result = await extractPreCompactionKnowledge(
      provider,
      "summary text",
      null,
      [],
      "original task",
    );
    expect(result).toBeNull();
  });

  it("extracts knowledge entries from valid response", async () => {
    const provider = createMockProvider(VALID_RESPONSE);
    const result = await extractPreCompactionKnowledge(
      provider,
      "Iteration: 15\nActive plan step: Transform descriptors",
      null,
      makeMessages(4),
      "Find and transform ANI type descriptors",
    );
    expect(result).not.toBeNull();
    expect(result!.entries.length).toBe(4);
    expect(result!.entries[0]!.key).toBe("inventory");
    expect(result!.entries[0]!.content).toContain("5 ANI descriptor");
    expect(result!.entries[3]!.key).toBe("next_action");
  });

  it("filters out entries with empty key or content", async () => {
    const response = JSON.stringify({
      entries: [
        { key: "inventory", content: "Found 5 files" },
        { key: "", content: "Bad entry with empty key" },
        { key: "decisions", content: "" },
        { key: "progress", content: "3/7 done" },
      ],
    });
    const provider = createMockProvider(response);
    const result = await extractPreCompactionKnowledge(
      provider,
      "summary",
      null,
      [],
      "task",
    );
    expect(result).not.toBeNull();
    expect(result!.entries.length).toBe(2);
    expect(result!.entries[0]!.key).toBe("inventory");
    expect(result!.entries[1]!.key).toBe("progress");
  });

  it("limits input messages to last 20", async () => {
    let capturedInput = "";
    const provider: LLMProvider = {
      id: "mock",
      async *chat(msgs): AsyncIterable<StreamChunk> {
        const userMsg = msgs.find((m) => m.role === MessageRole.USER);
        capturedInput = userMsg?.content ?? "";
        yield { type: "text", content: VALID_RESPONSE };
        yield { type: "done", content: "" };
      },
      abort() {},
    };

    const manyMessages = makeMessages(40);
    await extractPreCompactionKnowledge(
      provider,
      "summary",
      null,
      manyMessages,
      "task",
    );

    // Should only include last 20 messages (Message 20..39)
    expect(capturedInput).toContain("Message 39");
    expect(capturedInput).toContain("Message 20");
    expect(capturedInput).not.toContain("Message 0]");
    expect(capturedInput).not.toContain("Message 19]");
  });

  it("includes session state context when provided", async () => {
    let capturedInput = "";
    const provider: LLMProvider = {
      id: "mock",
      async *chat(msgs): AsyncIterable<StreamChunk> {
        const userMsg = msgs.find((m) => m.role === MessageRole.USER);
        capturedInput = userMsg?.content ?? "";
        yield { type: "text", content: VALID_RESPONSE };
        yield { type: "done", content: "" };
      },
      abort() {},
    };

    const ss = new SessionState({ persist: false });
    ss.setPlan([
      { description: "Step 1", status: "completed" },
      { description: "Step 2", status: "in_progress" },
    ]);
    ss.recordModifiedFile("src/foo.ts");

    await extractPreCompactionKnowledge(
      provider,
      "summary",
      ss,
      [],
      "task",
    );

    expect(capturedInput).toContain("Plan progress");
    expect(capturedInput).toContain("Modified files");
  });

  it("handles response wrapped in markdown fences", async () => {
    const fenced = "```json\n" + VALID_RESPONSE + "\n```";
    const provider = createMockProvider(fenced);
    const result = await extractPreCompactionKnowledge(
      provider,
      "summary",
      null,
      [],
      "task",
    );
    expect(result).not.toBeNull();
    expect(result!.entries.length).toBe(4);
  });

  it("system prompt instructs extraction of INVENTORY, DECISIONS, PROGRESS, NEXT_ACTION", () => {
    expect(KNOWLEDGE_EXTRACTION_SYSTEM_PROMPT).toContain("INVENTORY");
    expect(KNOWLEDGE_EXTRACTION_SYSTEM_PROMPT).toContain("DECISIONS");
    expect(KNOWLEDGE_EXTRACTION_SYSTEM_PROMPT).toContain("PROGRESS");
    expect(KNOWLEDGE_EXTRACTION_SYSTEM_PROMPT).toContain("NEXT_ACTION");
  });
});
