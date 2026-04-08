import { describe, expect, it } from "vitest";

import * as core from "./index.js";
import * as skills from "./skills/index.js";

describe("runtime core skill prompt-format exports", () => {
  it("exports skill prompt-format helpers from the skills barrel", () => {
    expect(typeof skills.formatSkillMatchLine).toBe("function");
    expect(typeof skills.formatSkillPromptGuidance).toBe("function");
  });

  it("re-exports skill prompt-format helpers from the runtime core barrel", () => {
    expect(core.formatSkillMatchLine).toBe(skills.formatSkillMatchLine);
    expect(core.formatSkillPromptGuidance).toBe(skills.formatSkillPromptGuidance);
  });
});
