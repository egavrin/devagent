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

  it("documents the devagent-api gateway pairing with cortex", () => {
    const readme = readFileSync(join(repoRoot, "README.md"), "utf-8");
    expect(readme).toContain('provider = "devagent-api"');
    expect(readme).toContain('model = "cortex"');
    expect(readme).toContain("DEVAGENT_API_KEY=ilg_");
    expect(readme).toContain("OpenAI-compatible under the hood");
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
