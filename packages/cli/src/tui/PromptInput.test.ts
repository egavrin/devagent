import { describe, expect, it } from "vitest";

import { TUI_HELP_MESSAGE } from "./App.js";
import { getCompletions, SLASH_COMMANDS } from "./PromptInput.js";

describe("PromptInput slash completions", () => {
  it("includes review and simplify prompt shortcuts", () => {
    expect(SLASH_COMMANDS).toContain("/review");
    expect(SLASH_COMMANDS).toContain("/simplify");
  });

  it("completes embedded review commands without breaking control commands", () => {
    expect(getCompletions("please /r")).toContain("please /review");
    expect(getCompletions("/s")).toContain("/sessions");
    expect(getCompletions("/s")).toContain("/simplify");
  });
});

describe("TUI help message", () => {
  it("mentions prompt shortcuts and existing control commands", () => {
    expect(TUI_HELP_MESSAGE).toContain("/review");
    expect(TUI_HELP_MESSAGE).toContain("/simplify");
    expect(TUI_HELP_MESSAGE).toContain("anywhere");
    expect(TUI_HELP_MESSAGE).toContain("/clear");
    expect(TUI_HELP_MESSAGE).toContain("/sessions");
  });
});
