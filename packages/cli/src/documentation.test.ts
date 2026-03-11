import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const cliRoot = dirname(fileURLToPath(import.meta.url));
const repoRoot = join(cliRoot, "..", "..", "..");

describe("documentation parity", () => {
  it("documents execute as the public machine contract", () => {
    const readme = readFileSync(join(repoRoot, "README.md"), "utf-8");
    expect(readme).toContain("devagent execute --request request.json --artifact-dir");
    expect(readme).not.toContain("desktop/");
  });

  it("does not expose legacy workflow-run guidance in live docs", () => {
    const files = [
      join(repoRoot, "README.md"),
      join(repoRoot, "AGENTS.md"),
    ];

    for (const file of files) {
      const contents = readFileSync(file, "utf-8");
      expect(contents).not.toContain("devagent workflow run");
      expect(contents).not.toContain("workflow run --phase");
    }
  });
});
