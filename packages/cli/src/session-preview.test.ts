import { describe, expect, it } from "vitest";

import {
  buildSessionPreview,
  deriveSessionTitle,
  formatRelativeUpdatedAt,
  renderSessionPreview,
} from "./session-preview.js";

describe("deriveSessionTitle", () => {
  it("keeps a normal query readable", () => {
    expect(deriveSessionTitle("Fix failing auth test")).toBe("Fix failing auth test");
  });

  it("normalizes multiline queries and trims trailing punctuation", () => {
    expect(deriveSessionTitle("\"Fix\n failing tests in auth module???\""))
      .toBe("Fix failing tests in auth module");
  });

  it("truncates long queries with an ellipsis", () => {
    expect(deriveSessionTitle("This is a very long query that keeps going past the title limit for session previews"))
      .toBe("This is a very long query that keeps going pa...");
  });

  it("falls back for empty or placeholder queries", () => {
    expect(deriveSessionTitle("   ")).toBe("Interactive session");
    expect(deriveSessionTitle("(interactive query)")).toBe("Interactive session");
    expect(deriveSessionTitle("(file query)")).toBe("Prompt from file");
  });
});

describe("session previews", () => {
  it("derives legacy preview data from stored query and repo root", () => {
    const preview = buildSessionPreview({
      id: "12345678-aaaa-bbbb-cccc-1234567890ab",
      updatedAt: 4_000,
      metadata: {
        query: "Review parser failure output",
        repoRoot: "/Users/egavrin/Documents/devagent",
      },
    });

    expect(preview).toEqual({
      id: "12345678-aaaa-bbbb-cccc-1234567890ab",
      updatedAt: 4_000,
      title: "Review parser failure output",
      repoLabel: "devagent",
      cost: undefined,
    });
  });

  it("uses a later substantive user message when the stored query is low-signal", () => {
    const preview = buildSessionPreview({
      id: "12345678-aaaa-bbbb-cccc-1234567890ab",
      updatedAt: 4_000,
      metadata: {
        query: "hi",
      },
      messages: [
        { role: "user", content: "hi" },
        { role: "assistant", content: "How can I help?" },
        { role: "user", content: "Fix the ArkTS validation errors in devagent_lsp" },
      ],
    });

    expect(preview.title).toBe("Fix the ArkTS validation errors in devagent_lsp");
  });

  it("falls back to an untitled label when only trivial prompts exist", () => {
    const preview = buildSessionPreview({
      id: "12345678-aaaa-bbbb-cccc-1234567890ab",
      updatedAt: 4_000,
      metadata: {
        query: "2+2",
      },
      messages: [
        { role: "user", content: "hello" },
        { role: "assistant", content: "Hi there" },
      ],
    });

    expect(preview.title).toBe("Untitled session");
  });

  it("renders multi-line session previews with relative time and cost", () => {
    const text = renderSessionPreview({
      id: "12345678-aaaa-bbbb-cccc-1234567890ab",
      updatedAt: 60_000,
      title: "Fix auth retry loop",
      repoLabel: "devagent",
      cost: 0.125,
    }, 2 * 60_000);

    expect(text).toBe([
      "  Fix auth retry loop",
      "    12345678  devagent  1m ago  $0.1250",
    ].join("\n"));
  });

  it("omits the repo label when it is unknown", () => {
    const text = renderSessionPreview({
      id: "12345678-aaaa-bbbb-cccc-1234567890ab",
      updatedAt: 60_000,
      title: "Untitled session",
      repoLabel: "unknown repo",
    }, 2 * 60_000);

    expect(text).toBe([
      "  Untitled session",
      "    12345678  1m ago",
    ].join("\n"));
  });

  it("formats relative timestamps in stable buckets", () => {
    expect(formatRelativeUpdatedAt(95_000, 100_000)).toBe("just now");
    expect(formatRelativeUpdatedAt(0, 2 * 60 * 60 * 1000)).toBe("2h ago");
    expect(formatRelativeUpdatedAt(0, 10 * 24 * 60 * 60 * 1000)).toBe("1w ago");
  });
});
