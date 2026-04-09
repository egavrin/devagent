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
    expect(readme).toContain("The primary DevAgent product surface is the staged executor contract");
    expect(readme).toContain("`design -> breakdown -> issue-generation -> implement -> review -> repair`");
    expect(readme).toContain("## Stage Matrix");
    expect(readme).not.toContain("desktop/");
  });

  it("documents the fixed stage set and dynamic request context for execute", () => {
    const readme = readFileSync(join(repoRoot, "README.md"), "utf-8");
    expect(readme).toContain("The workflow is a fixed supported stage set.");
    expect(readme).toContain("Stage prompts follow a fixed shape plus dynamic request context");
    expect(readme).toContain("repo/work item metadata");
    expect(readme).toContain("extraInstructions");
    expect(readme).toContain("This pass does not expose stage prompts as user-editable templates.");
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

  it("documents the supported runtime floor and Ubuntu-safe Node setup", () => {
    const readme = readFileSync(join(repoRoot, "README.md"), "utf-8");
    expect(readme).toContain("DevAgent requires Node.js 20+ or Bun 1.3+.");
    expect(readme).toContain("do not rely on `apt install nodejs` for this project");
    expect(readme).toContain("nvm install 20");
    expect(readme).toContain("nvm use 20");
    expect(readme).toContain("bunx @egavrin/devagent");
  });

  it("describes the interactive surface as the TUI, not a REPL", () => {
    const readme = readFileSync(join(repoRoot, "README.md"), "utf-8");
    expect(readme).toContain("Interactive TUI");
    expect(readme).toContain("devagent --mode autopilot");
    expect(readme).toContain("max_iterations = 0");
    expect(readme).toContain('[safety]');
    expect(readme).toContain('mode = "autopilot"');
    expect(readme).not.toContain("Interactive mode");
    expect(readme).not.toContain("REPL");
  });

  it("documents setup as the public onboarding command", () => {
    const readme = readFileSync(join(repoRoot, "README.md"), "utf-8");
    expect(readme).toContain("devagent help");
    expect(readme).toContain("| `devagent help` | Show top-level help |");
    expect(readme).toContain("devagent setup");
    expect(readme).toContain("| `devagent setup` | Guided global configuration wizard |");
    expect(readme).toContain("| `devagent config get/set/path` | Inspect or edit global config directly |");
    expect(readme).toContain("| `devagent install-lsp` | Install LSP servers for code intelligence |");
    expect(readme).toContain("| `devagent auth login/status/logout` | Manage provider credentials |");
    expect(readme).toContain("| `devagent sessions` | List recent sessions |");
    expect(readme).toContain("| `devagent execute --request <file> --artifact-dir <dir>` | Execute an SDK request and write artifacts |");
    expect(readme).toContain("devagent auth logout chatgpt");
    expect(readme).toContain("devagent auth logout --all");
    expect(readme).not.toContain("`devagent init`");
  });

  it("keeps workflow prose aligned with public safety-mode language", () => {
    const workflow = readFileSync(join(repoRoot, "WORKFLOW.md"), "utf-8");
    expect(workflow).toContain("`design -> breakdown -> issue-generation -> implement -> review -> repair`");
    expect(workflow).toContain("fixed rather than user-defined");
    expect(workflow).toContain("Public stage semantics are code-defined by `taskType`");
    expect(workflow).toContain("`devagent --mode default`");
    expect(workflow).toContain("`devagent --mode autopilot`");
    expect(workflow).toContain("legacy approval-mode flags");
  });

  it("describes project instructions as optional manual files", () => {
    const readme = readFileSync(join(repoRoot, "README.md"), "utf-8");
    expect(readme).toContain("Project instructions are optional.");
    expect(readme).toContain("Create `AGENTS.md` manually");
    expect(readme).not.toContain(".devagent/instructions.md");
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
