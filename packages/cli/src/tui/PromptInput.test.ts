import { mkdtempSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { describe, expect, it } from "vitest";

import { TUI_HELP_MESSAGE } from "./App.js";
import { getCompletions, SLASH_COMMANDS } from "./PromptInput.js";
import { cycleApprovalMode, resolvePromptTabAction } from "./shared.js";

describe("PromptInput slash completions", () => {
  it("includes review and simplify prompt shortcuts", () => {
    expect(SLASH_COMMANDS).toContain("/review");
    expect(SLASH_COMMANDS).toContain("/simplify");
    expect(SLASH_COMMANDS).toContain("/continue");
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
    expect(TUI_HELP_MESSAGE).toContain("Shift+Tab");
  });
});

describe("safety mode helpers", () => {
  it("toggles default -> autopilot -> default", () => {
    expect(cycleApprovalMode("default")).toBe("autopilot");
    expect(cycleApprovalMode("autopilot")).toBe("default");
    expect(cycleApprovalMode("legacy" as never)).toBe("default");
  });

  it("routes Shift+Tab to mode cycling while plain Tab stays on completion", () => {
    expect(resolvePromptTabAction({ tab: true, shift: true })).toBe("cycle-mode");
    expect(resolvePromptTabAction({ tab: true, shift: false })).toBe("complete");
    expect(resolvePromptTabAction({ tab: false, shift: true })).toBe("none");
  });
});
