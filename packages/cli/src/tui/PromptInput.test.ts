import { mkdtempSync, mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import stringWidth from "string-width";
import { describe, expect, it } from "vitest";

import { TUI_HELP_MESSAGE } from "./App.js";
import {
  buildPromptRows,
  getCompletions,
  shouldInsertPromptNewline,
  SLASH_COMMANDS,
} from "./PromptInput.js";
import { cycleApprovalMode, resolvePromptTabAction } from "./shared.js";

describe("PromptInput slash completions", () => {
  it("includes review and simplify prompt shortcuts", () => {
    expect(SLASH_COMMANDS).toContain("/review");
    expect(SLASH_COMMANDS).toContain("/simplify");
    expect(SLASH_COMMANDS).toContain("/continue");
    expect(SLASH_COMMANDS).not.toContain("/rename");
  });

  it("completes embedded review commands without breaking control commands", () => {
    expect(getCompletions("please /r")).toContain("please /review");
    expect(getCompletions("/s")).toContain("/sessions");
    expect(getCompletions("/s")).toContain("/simplify");
  });

  it("keeps absolute-path completion working inside normal prompts", () => {
    const tempRoot = mkdtempSync(join(tmpdir(), "devagent-prompt-input-"));
    mkdirSync(join(tempRoot, "foo-dir"));

    expect(getCompletions(`edit ${tempRoot}/fo`)).toContain(`edit ${tempRoot}/foo-dir`);
  });
});

describe("TUI help message", () => {
  it("mentions prompt shortcuts and existing control commands", () => {
    expect(TUI_HELP_MESSAGE).toContain("/review");
    expect(TUI_HELP_MESSAGE).toContain("/simplify");
    expect(TUI_HELP_MESSAGE).toContain("/continue");
    expect(TUI_HELP_MESSAGE).toContain("anywhere");
    expect(TUI_HELP_MESSAGE).toContain("/clear");
    expect(TUI_HELP_MESSAGE).toContain("/sessions");
    expect(TUI_HELP_MESSAGE).toContain("Shift+Enter");
    expect(TUI_HELP_MESSAGE).toContain("Option+Enter");
    expect(TUI_HELP_MESSAGE).toContain("Shift+Tab");
  });
});

describe("PromptInput layout helpers", () => {
  it("treats modified Return keys as newline insertion", () => {
    expect(shouldInsertPromptNewline({ return: true, shift: true })).toBe(true);
    expect(shouldInsertPromptNewline({ return: true, meta: true })).toBe(true);
    expect(shouldInsertPromptNewline({ return: true, super: true })).toBe(true);
    expect(shouldInsertPromptNewline({ return: true, hyper: true })).toBe(true);
    expect(shouldInsertPromptNewline({ return: true })).toBe(false);
    expect(shouldInsertPromptNewline({ shift: true })).toBe(false);
  });

  it("wraps long prompt text into explicit continuation rows", () => {
    const rows = buildPromptRows("abcdefghij", 5, "placeholder", 5);

    expect(rows).toEqual([
      { prefix: "❯ ", text: "abcde", cursorOffset: null, dim: false },
      { prefix: "… ", text: "fghij", cursorOffset: 0, dim: false },
    ]);
  });

  it("keeps explicit blank lines as continuation rows", () => {
    const rows = buildPromptRows("hi\n\nthere", 3, "placeholder", 5);

    expect(rows).toEqual([
      { prefix: "❯ ", text: "hi", cursorOffset: null, dim: false },
      { prefix: "… ", text: "", cursorOffset: 0, dim: false },
      { prefix: "… ", text: "there", cursorOffset: null, dim: false },
    ]);
  });

  it("wraps emoji without splitting surrogate pairs", () => {
    const rows = buildPromptRows("😀😀😀", 4, "placeholder", 4);

    expect(rows).toEqual([
      { prefix: "❯ ", text: "😀😀", cursorOffset: null, dim: false },
      { prefix: "… ", text: "😀", cursorOffset: 0, dim: false },
    ]);
    expect(rows.every((row) => stringWidth(row.text) <= 4)).toBe(true);
  });

  it("wraps full-width CJK characters by terminal cell width", () => {
    const rows = buildPromptRows("你好a", 2, "placeholder", 4);

    expect(rows).toEqual([
      { prefix: "❯ ", text: "你好", cursorOffset: null, dim: false },
      { prefix: "… ", text: "a", cursorOffset: 0, dim: false },
    ]);
    expect(rows.map((row) => stringWidth(row.text))).toEqual([4, 1]);
  });
});

describe("safety mode helpers", () => {
  it("toggles default -> autopilot -> default", () => {
    expect(cycleApprovalMode("default")).toBe("autopilot");
    expect(cycleApprovalMode("autopilot")).toBe("default");
    expect(cycleApprovalMode("legacy" as never)).toBe("autopilot");
  });

  it("routes Shift+Tab to mode cycling while plain Tab stays on completion", () => {
    expect(resolvePromptTabAction({ tab: true, shift: true })).toBe("cycle-mode");
    expect(resolvePromptTabAction({ tab: true, shift: false })).toBe("complete");
    expect(resolvePromptTabAction({ tab: false, shift: true })).toBe("none");
  });
});
