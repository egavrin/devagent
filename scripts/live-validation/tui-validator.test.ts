import { describe, expect, it } from "bun:test";

import { assertTuiFrame, extractSettledFrame } from "./tui-validator";

describe("extractSettledFrame", () => {
  it("returns the final settled frame instead of earlier repaint history", () => {
    const raw = [
      "old noise",
      "╔══════════════════╗",
      "║    devagent      ║",
      "╚══════════════════╝",
      "cortex • v0.1.0",
      "Shift+Tab safety │ Ctrl+K cmds │ Ctrl+C exit │ cortex │ workspace-123  ╔══════════════════╗",
      "╔══════════════════╗",
      "║    devagent      ║",
      "╚══════════════════╝",
      "cortex • v0.2.0",
      "Type /help for commands • Ctrl+K palette • Shift+Tab safety",
      "╭────────────────────╮",
      "│ ❯ /help           │",
      "╰────────────────────╯",
    ].join("\n");

    const frame = extractSettledFrame(raw);
    expect(frame).toContain("cortex • v0.2.0");
    expect(frame).not.toContain("workspace-123  ╔");
    expect(frame.startsWith("╔══════════════════╗")).toBe(true);
  });
});

describe("assertTuiFrame", () => {
  it("accepts a clean settled frame", () => {
    const frame = [
      "╔══════════════════╗",
      "║    devagent      ║",
      "╚══════════════════╝",
      "cortex • v0.2.0",
      "Type /help for commands • Ctrl+K palette • Shift+Tab safety",
      "No sessions found.",
      "╭────────────────────╮",
      "│ ❯ /               │",
      "│ … s               │",
      "│ … e               │",
      "│ … s               │",
      "│ … s               │",
      "│ … i               │",
      "│ … o               │",
      "│ … n               │",
      "│ … s               │",
      "╰────────────────────╯",
    ].join("\n");

    expect(() => assertTuiFrame(frame, { expectedVersion: "0.2.0", requiredText: "No sessions found." })).not.toThrow();
  });

  it("accepts command output matched by regex", () => {
    const frame = [
      "╔══════════════════╗",
      "║    devagent      ║",
      "╚══════════════════╝",
      "cortex • v0.2.0",
      "Type /help for commands • Ctrl+K palette • Shift+Tab safety",
      "Recent sessions:",
      "╭────────────────────╮",
      "│ ❯                 │",
      "╰────────────────────╯",
    ].join("\n");

    expect(() => assertTuiFrame(frame, { expectedVersion: "0.2.0", requiredText: /No sessions found\.|Recent sessions:/ })).not.toThrow();
  });

  it("rejects duplicate welcome blocks and banner collisions", () => {
    const frame = [
      "╔══════════════════╗",
      "║    devagent      ║",
      "╚══════════════════╝",
      "cortex • v0.2.0",
      "Shift+Tab safety │ Ctrl+K cmds │ Ctrl+C exit │ cortex │ workspace-9BtpJh  ╔══════════════════╗",
      "║    devagent      ║",
      "╚══════════════════╝",
      "╭────────────────────╮",
      "│ ❯ /clear          │",
      "╰────────────────────╯",
    ].join("\n");

    expect(() => assertTuiFrame(frame, { expectedVersion: "0.2.0", requiredText: "/clear" })).toThrow();
  });
});
