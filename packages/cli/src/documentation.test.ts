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

  it("documents the canonical global config path and provider env vars", () => {
    const readme = readFileSync(join(repoRoot, "README.md"), "utf-8");
    expect(readme).toContain("Global config: `~/.config/devagent/config.toml`");
    expect(readme).toContain("| `OPENAI_API_KEY` | OpenAI API key |");
    expect(readme).toContain("| `ANTHROPIC_API_KEY` | Anthropic API key |");
  });

  it("describes the interactive surface as the TUI, not a REPL", () => {
    const readme = readFileSync(join(repoRoot, "README.md"), "utf-8");
    expect(readme).toContain("Interactive TUI");
    expect(readme).toContain("max_iterations = 0");
    expect(readme).not.toContain("Interactive mode");
    expect(readme).not.toContain("REPL");
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
