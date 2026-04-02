/**
 * CLI subcommands: doctor, config, init.
 */

import { existsSync, readFileSync, writeFileSync, mkdirSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";
import { execSync } from "node:child_process";
import { createInterface } from "node:readline";
import {
  loadConfig,
  loadModelRegistry,
  getRegisteredModels,
} from "@devagent/runtime";

// ─── doctor ─────────────────────────────────────────────────

export async function runDoctor(version: string): Promise<void> {
  let allOk = true;

  function check(label: string, fn: () => string | null): void {
    const err = fn();
    if (err) {
      console.log(`  ✗ ${label}: ${err}`);
      allOk = false;
    } else {
      console.log(`  ✓ ${label}`);
    }
  }

  console.log(`devagent v${version}\n`);
  console.log("Checks:\n");

  // Runtime
  const runtime = typeof Bun !== "undefined" ? `Bun ${Bun.version}` : `Node ${process.version}`;
  check(`Runtime: ${runtime}`, () => {
    const major = parseInt(process.version.replace("v", ""), 10);
    if (major < 20 && typeof Bun === "undefined") return "Node.js >= 20 required";
    return null;
  });

  // Git
  check("Git", () => {
    try {
      const v = execSync("git --version", { encoding: "utf-8", timeout: 5000 }).trim();
      return null;
    } catch {
      return "git not found in PATH";
    }
  });

  // Config
  const configPaths = [
    join(homedir(), ".config", "devagent", "config.toml"),
    join(homedir(), ".devagent.toml"),
    ".devagent.toml",
    "devagent.toml",
  ];
  const foundConfig = configPaths.find((p) => existsSync(p));
  check(`Config file`, () => {
    if (foundConfig) return null;
    return `not found (searched: ${configPaths.join(", ")})`;
  });

  // Provider credentials
  const config = loadConfig();
  check(`Provider: ${config.provider}`, () => {
    const envKey = getProviderEnvKey(config.provider);
    if (envKey && process.env[envKey]) return null;
    const prov = config.providers[config.provider];
    if (prov?.apiKey) return null;
    return `no API key (set ${envKey ?? "DEVAGENT_API_KEY"} or run devagent auth login)`;
  });

  // Show available providers
  console.log("\n  Available providers:");
  for (const p of PROVIDERS) {
    const envKey = getProviderEnvKey(p.id);
    const hasKey = (envKey && process.env[envKey]) || config.providers[p.id]?.apiKey;
    const status = hasKey ? "✓" : "·";
    const active = p.id === config.provider ? " (active)" : "";
    console.log(`    ${status} ${p.id}${active} — ${p.hint}`);
  }
  console.log("");

  // Model registry
  try {
    loadModelRegistry();
    const models = getRegisteredModels();
    check(`Model registry: ${models.length} models loaded`, () => {
      if (models.length === 0) return "no models found";
      return null;
    });

    // Check configured model exists
    check(`Model: ${config.model}`, () => {
      if (models.includes(config.model)) return null;
      return `model "${config.model}" not in registry`;
    });
  } catch (err) {
    check("Model registry", () => String(err));
  }

  // LSP servers — check each individually
  console.log("  LSP servers:");
  let lspCount = 0;
  for (const lsp of LSP_SERVERS) {
    const found = commandExists(lsp.command);
    if (found) lspCount++;
    const status = found ? "✓" : "·";
    const install = found ? "" : ` — install: ${lsp.install}`;
    console.log(`    ${status} ${lsp.label}${install}`);
  }
  if (lspCount === 0) {
    console.log("    (none found — code intelligence will be limited)");
    allOk = false;
  }
  console.log("");

  // Platform
  check(`Platform: ${process.platform} ${process.arch}`, () => null);

  console.log("");
  if (allOk) {
    console.log("All checks passed.");
    process.exit(0);
  } else {
    console.log("Some checks failed.");
    process.exit(1);
  }
}

const PROVIDERS = [
  { id: "anthropic", env: "ANTHROPIC_API_KEY", hint: "set ANTHROPIC_API_KEY or devagent auth login" },
  { id: "openai", env: "OPENAI_API_KEY", hint: "set OPENAI_API_KEY or devagent auth login" },
  { id: "deepseek", env: "DEEPSEEK_API_KEY", hint: "set DEEPSEEK_API_KEY or devagent auth login" },
  { id: "openrouter", env: "OPENROUTER_API_KEY", hint: "set OPENROUTER_API_KEY or devagent auth login" },
  { id: "chatgpt", env: "CHATGPT_API_KEY", hint: "devagent auth login (ChatGPT Plus/Pro)" },
  { id: "github-copilot", env: "GITHUB_TOKEN", hint: "devagent auth login (GitHub device flow)" },
  { id: "ollama", env: "", hint: "local — no API key needed (ollama must be running)" },
];

const LSP_SERVERS = [
  { command: "typescript-language-server", label: "TypeScript/JavaScript", install: "npm i -g typescript-language-server typescript" },
  { command: "pyright-langserver", label: "Python (Pyright)", install: "npm i -g pyright" },
  { command: "clangd", label: "C/C++ (clangd)", install: "apt install clangd / brew install llvm" },
  { command: "rust-analyzer", label: "Rust", install: "rustup component add rust-analyzer" },
  { command: "bash-language-server", label: "Bash/Shell", install: "npm i -g bash-language-server" },
];

function getProviderEnvKey(provider: string): string | null {
  const p = PROVIDERS.find((x) => x.id === provider);
  return p?.env || null;
}

function commandExists(cmd: string): boolean {
  try {
    execSync(`which ${cmd}`, { encoding: "utf-8", timeout: 3000, stdio: "pipe" });
    return true;
  } catch {
    return false;
  }
}

// ─── config ─────────────────────────────────────────────────

const GLOBAL_CONFIG_DIR = join(homedir(), ".config", "devagent");
const GLOBAL_CONFIG_PATH = join(GLOBAL_CONFIG_DIR, "config.json");

export function runConfig(args: string[]): void {
  const sub = args[0];

  if (!sub || sub === "path") {
    console.log(GLOBAL_CONFIG_PATH);
    return;
  }

  if (sub === "get") {
    const key = args[1];
    const data = loadConfigJson();
    if (!key) {
      // Dump all
      if (Object.keys(data).length === 0) {
        console.log("(no config set)");
      } else {
        for (const [k, v] of flatEntries(data)) {
          console.log(`${k} = ${formatValue(v)}`);
        }
      }
      return;
    }
    const value = getNestedValue(data, key);
    if (value === undefined) {
      console.log(`(not set)`);
    } else {
      console.log(formatValue(value));
    }
    return;
  }

  if (sub === "set") {
    const key = args[1];
    const value = args[2];
    if (!key || value === undefined) {
      console.error("Usage: devagent config set <key> <value>");
      process.exit(2);
    }
    const data = loadConfigJson();
    setNestedValue(data, key, parseValue(value));
    saveConfigJson(data);
    console.log(`${key} = ${value}`);
    return;
  }

  console.error(`Unknown config subcommand: ${sub}`);
  console.error("Usage: devagent config {get|set|path}");
  process.exit(2);
}

function loadConfigJson(): Record<string, unknown> {
  if (!existsSync(GLOBAL_CONFIG_PATH)) return {};
  try {
    return JSON.parse(readFileSync(GLOBAL_CONFIG_PATH, "utf-8")) as Record<string, unknown>;
  } catch {
    return {};
  }
}

function saveConfigJson(data: Record<string, unknown>): void {
  mkdirSync(GLOBAL_CONFIG_DIR, { recursive: true });
  writeFileSync(GLOBAL_CONFIG_PATH, JSON.stringify(data, null, 2) + "\n");
}

function getNestedValue(obj: Record<string, unknown>, path: string): unknown {
  const parts = path.split(".");
  let current: unknown = obj;
  for (const part of parts) {
    if (current == null || typeof current !== "object") return undefined;
    current = (current as Record<string, unknown>)[part];
  }
  return current;
}

function setNestedValue(obj: Record<string, unknown>, path: string, value: unknown): void {
  const parts = path.split(".");
  let current = obj;
  for (let i = 0; i < parts.length - 1; i++) {
    const part = parts[i]!;
    if (!(part in current) || typeof current[part] !== "object" || current[part] === null) {
      current[part] = {};
    }
    current = current[part] as Record<string, unknown>;
  }
  current[parts[parts.length - 1]!] = value;
}

function parseValue(s: string): unknown {
  if (s === "true") return true;
  if (s === "false") return false;
  const n = Number(s);
  if (!isNaN(n) && s.trim() !== "") return n;
  return s;
}

function formatValue(v: unknown): string {
  if (typeof v === "string") return v;
  if (typeof v === "object" && v !== null) return JSON.stringify(v);
  return String(v);
}

function flatEntries(obj: Record<string, unknown>, prefix = ""): Array<[string, unknown]> {
  const entries: Array<[string, unknown]> = [];
  for (const [k, v] of Object.entries(obj)) {
    const key = prefix ? `${prefix}.${k}` : k;
    if (typeof v === "object" && v !== null && !Array.isArray(v)) {
      entries.push(...flatEntries(v as Record<string, unknown>, key));
    } else {
      entries.push([key, v]);
    }
  }
  return entries;
}

// ─── init ───────────────────────────────────────────────────

export function runInit(): void {
  const cwd = process.cwd();
  const devagentDir = join(cwd, ".devagent");

  if (existsSync(devagentDir)) {
    console.log(".devagent/ already exists.");
  } else {
    mkdirSync(devagentDir, { recursive: true });
    console.log("Created .devagent/");
  }

  // Instructions file
  const instructionsPath = join(devagentDir, "instructions.md");
  if (!existsSync(instructionsPath)) {
    writeFileSync(instructionsPath, INSTRUCTIONS_TEMPLATE);
    console.log("Created .devagent/instructions.md");
  } else {
    console.log(".devagent/instructions.md already exists.");
  }

  // AGENTS.md
  const agentsPath = join(cwd, "AGENTS.md");
  if (!existsSync(agentsPath)) {
    const projectType = detectProjectType(cwd);
    writeFileSync(agentsPath, generateAgentsMd(projectType));
    console.log(`Created AGENTS.md (detected: ${projectType})`);
  } else {
    console.log("AGENTS.md already exists.");
  }

  console.log("\nDone. Edit these files to customize agent behavior.");
}

function detectProjectType(dir: string): string {
  if (existsSync(join(dir, "package.json"))) return "node";
  if (existsSync(join(dir, "Cargo.toml"))) return "rust";
  if (existsSync(join(dir, "go.mod"))) return "go";
  if (existsSync(join(dir, "pyproject.toml")) || existsSync(join(dir, "setup.py"))) return "python";
  if (existsSync(join(dir, "pom.xml")) || existsSync(join(dir, "build.gradle"))) return "java";
  if (existsSync(join(dir, "*.sln"))) return "dotnet";
  return "generic";
}

function generateAgentsMd(projectType: string): string {
  const buildCmds: Record<string, string> = {
    node: "npm install\nnpm run build\nnpm test",
    rust: "cargo build\ncargo test",
    go: "go build ./...\ngo test ./...",
    python: "pip install -e .\npytest",
    java: "mvn compile\nmvn test",
    dotnet: "dotnet build\ndotnet test",
    generic: "# Add your build/test commands here",
  };

  return `# Project Agent Instructions

## Build and Test

\`\`\`bash
${buildCmds[projectType] ?? buildCmds.generic}
\`\`\`

## Conventions

- Follow existing code style and patterns
- Keep changes minimal and focused
- Write tests for new functionality
- Run tests before considering work complete

## Architecture

<!-- Describe your project's architecture, key directories, and design decisions -->
`;
}

// ─── setup ──────────────────────────────────────────────────

const SETUP_PROVIDERS = [
  { id: "anthropic", name: "Anthropic", envVar: "ANTHROPIC_API_KEY", defaultModel: "claude-sonnet-4-20250514", hint: "Get key at https://console.anthropic.com/settings/keys" },
  { id: "openai", name: "OpenAI", envVar: "OPENAI_API_KEY", defaultModel: "gpt-4.1", hint: "Get key at https://platform.openai.com/api-keys" },
  { id: "deepseek", name: "DeepSeek", envVar: "DEEPSEEK_API_KEY", defaultModel: "deepseek-chat", hint: "Get key at https://platform.deepseek.com/api_keys" },
  { id: "openrouter", name: "OpenRouter", envVar: "OPENROUTER_API_KEY", defaultModel: "anthropic/claude-sonnet-4-20250514", hint: "Get key at https://openrouter.ai/keys" },
  { id: "ollama", name: "Ollama (local)", envVar: "", defaultModel: "qwen3:32b", hint: "No API key needed — ollama must be running locally" },
  { id: "chatgpt", name: "ChatGPT (Pro/Plus)", envVar: "", defaultModel: "gpt-4.1", hint: "Use 'devagent auth login' after setup" },
  { id: "github-copilot", name: "GitHub Copilot", envVar: "", defaultModel: "gpt-4.1", hint: "Use 'devagent auth login' after setup" },
];

export async function runSetup(): Promise<void> {
  const rl = createInterface({ input: process.stdin, output: process.stderr });
  const ask = (prompt: string): Promise<string> =>
    new Promise((resolve) => rl.question(prompt, resolve));

  const configDir = join(homedir(), ".config", "devagent");
  const configPath = join(configDir, "config.toml");
  const isUpdate = existsSync(configPath);

  console.log("DevAgent Setup\n");
  if (isUpdate) console.log(`(Existing config at ${configPath} will be updated)\n`);

  // 1. Provider selection
  console.log("Select your LLM provider:\n");
  for (let i = 0; i < SETUP_PROVIDERS.length; i++) {
    const p = SETUP_PROVIDERS[i]!;
    console.log(`  ${i + 1}. ${p.name}`);
    console.log(`     ${p.hint}`);
  }
  console.log("");

  const providerChoice = await ask("> Provider (1-7) [1]: ");
  const providerIdx = (parseInt(providerChoice.trim(), 10) || 1) - 1;
  const provider = SETUP_PROVIDERS[Math.max(0, Math.min(providerIdx, SETUP_PROVIDERS.length - 1))]!;
  console.log(`\n  ✓ Provider: ${provider.name}\n`);

  // 2. API key (for key-based providers)
  let apiKey = "";
  if (provider.envVar) {
    const existing = process.env[provider.envVar];
    if (existing) {
      console.log(`  ${provider.envVar} already set in environment.`);
      console.log(`  (Will use environment variable at runtime)\n`);
    } else {
      apiKey = await ask(`> ${provider.envVar}: `);
      apiKey = apiKey.trim();
      if (apiKey) {
        console.log(`  ✓ API key stored\n`);
      } else {
        console.log(`  (Skipped — set ${provider.envVar} in your shell profile later)\n`);
      }
    }
  } else if (provider.id === "ollama") {
    console.log("  No API key needed for Ollama.\n");
  } else {
    console.log(`  Run 'devagent auth login' after setup to authenticate.\n`);
  }

  // 3. Model selection
  const modelPrompt = `> Model [${provider.defaultModel}]: `;
  const modelChoice = await ask(modelPrompt);
  const model = modelChoice.trim() || provider.defaultModel;
  console.log(`  ✓ Model: ${model}\n`);

  // 4. Approval mode
  console.log("Approval mode:");
  console.log("  1. suggest — ask before writing files (recommended)");
  console.log("  2. auto-edit — auto-approve file writes, ask for commands");
  console.log("  3. full-auto — auto-approve everything");
  console.log("");
  const approvalChoice = await ask("> Approval mode (1-3) [1]: ");
  const approvalModes = ["suggest", "auto-edit", "full-auto"];
  const approvalIdx = (parseInt(approvalChoice.trim(), 10) || 1) - 1;
  const approvalMode = approvalModes[Math.max(0, Math.min(approvalIdx, 2))]!;
  console.log(`  ✓ Approval mode: ${approvalMode}\n`);

  // 5. Max iterations
  const iterChoice = await ask("> Max iterations per query [30]: ");
  const maxIterations = parseInt(iterChoice.trim(), 10) || 30;
  console.log(`  ✓ Max iterations: ${maxIterations}\n`);

  // 6. Subagent configuration
  const AGENT_TYPES = [
    { id: "general", label: "General", desc: "default agent for code tasks" },
    { id: "explore", label: "Explore", desc: "fast codebase search, read-only" },
    { id: "reviewer", label: "Reviewer", desc: "code review, read-only" },
    { id: "architect", label: "Architect", desc: "design and planning, read-only" },
  ];

  // Sensible defaults per provider
  const subagentDefaults: Record<string, { models: Record<string, string>; reasoning: Record<string, string> }> = {
    anthropic: {
      models: { general: model, explore: "claude-haiku-4-20250414", reviewer: model, architect: model },
      reasoning: { general: "medium", explore: "low", reviewer: "high", architect: "high" },
    },
    openai: {
      models: { general: model, explore: "gpt-5.4-mini", reviewer: "gpt-5.4", architect: "gpt-5.4" },
      reasoning: { general: "medium", explore: "low", reviewer: "high", architect: "high" },
    },
    deepseek: {
      models: { general: model, explore: model, reviewer: model, architect: model },
      reasoning: { general: "medium", explore: "low", reviewer: "high", architect: "high" },
    },
  };
  const defaults = subagentDefaults[provider.id] ?? {
    models: { general: model, explore: model, reviewer: model, architect: model },
    reasoning: { general: "medium", explore: "low", reviewer: "high", architect: "high" },
  };

  console.log("Subagent configuration:");
  console.log("  DevAgent spawns specialized subagents for different tasks.");
  console.log("  You can use cheaper/faster models for simple tasks like exploration.\n");

  console.log("  Defaults for your provider:");
  for (const agent of AGENT_TYPES) {
    const m = defaults.models[agent.id] ?? model;
    const r = defaults.reasoning[agent.id] ?? "medium";
    console.log(`    ${agent.label.padEnd(10)} model=${m}  reasoning=${r}`);
  }
  console.log("");

  const customizeSub = await ask("> Customize subagent models? (y/N) [N]: ");
  const agentModels: Record<string, string> = { ...defaults.models };
  const agentReasoning: Record<string, string> = { ...defaults.reasoning };

  if (customizeSub.trim().toLowerCase() === "y") {
    console.log("");
    for (const agent of AGENT_TYPES) {
      const defModel = defaults.models[agent.id] ?? model;
      const defReasoning = defaults.reasoning[agent.id] ?? "medium";

      const mChoice = await ask(`  > ${agent.label} model [${defModel}]: `);
      agentModels[agent.id] = mChoice.trim() || defModel;

      const rChoice = await ask(`  > ${agent.label} reasoning (low/medium/high) [${defReasoning}]: `);
      const r = rChoice.trim().toLowerCase();
      agentReasoning[agent.id] = (r === "low" || r === "medium" || r === "high") ? r : defReasoning;

      console.log(`    ✓ ${agent.label}: ${agentModels[agent.id]} (${agentReasoning[agent.id]})\n`);
    }
  } else {
    console.log("  ✓ Using defaults\n");
  }

  rl.close();

  // Write config.toml
  const lines: string[] = [
    "# DevAgent global configuration",
    `# Generated by 'devagent setup' on ${new Date().toISOString().split("T")[0]}`,
    "",
    `provider = "${provider.id}"`,
    `model = "${model}"`,
    "",
    "[approval]",
    `mode = "${approvalMode}"`,
    "",
    "[budget]",
    `max_iterations = ${maxIterations}`,
  ];

  // Subagent config
  const hasCustomModels = Object.entries(agentModels).some(([k, v]) => v !== model);
  const hasCustomReasoning = true; // Always write reasoning defaults
  if (hasCustomModels || hasCustomReasoning) {
    lines.push("", "[subagents]", "# Per-agent model and reasoning overrides");
    lines.push("# Agent types: general, explore, reviewer, architect");

    if (hasCustomModels) {
      lines.push("", "[subagents.agent_model_overrides]");
      for (const agent of AGENT_TYPES) {
        const m = agentModels[agent.id];
        if (m && m !== model) {
          lines.push(`${agent.id} = "${m}"`);
        }
      }
    }

    lines.push("", "[subagents.agent_reasoning_overrides]");
    for (const agent of AGENT_TYPES) {
      lines.push(`${agent.id} = "${agentReasoning[agent.id] ?? "medium"}"`);
    }
  }

  // Provider-specific config
  if (apiKey) {
    lines.push("", `[providers.${provider.id}]`, `api_key = "${apiKey}"`);
  }
  if (provider.id === "ollama") {
    lines.push("", "[providers.ollama]", 'base_url = "http://localhost:11434/v1"');
  }

  mkdirSync(configDir, { recursive: true });
  writeFileSync(configPath, lines.join("\n") + "\n");

  console.log(`Config written to ${configPath}\n`);
  console.log("Next steps:");
  if (!apiKey && provider.envVar) {
    console.log(`  1. Set ${provider.envVar} in your shell profile`);
    console.log(`  2. Run 'devagent doctor' to verify`);
    console.log(`  3. Run 'devagent "hello"' to test`);
  } else if (provider.id === "chatgpt" || provider.id === "github-copilot") {
    console.log(`  1. Run 'devagent auth login' to authenticate`);
    console.log(`  2. Run 'devagent doctor' to verify`);
    console.log(`  3. Run 'devagent "hello"' to test`);
  } else {
    console.log(`  1. Run 'devagent doctor' to verify`);
    console.log(`  2. Run 'devagent "hello"' to test`);
  }
  console.log(`  - Run 'devagent init' in a project to add project-level config`);
}

// ─── update ─────────────────────────────────────────────────

export async function runUpdate(): Promise<void> {
  const PACKAGE = "@egavrin/devagent";

  console.log("Checking for updates...");

  try {
    const res = await fetch(`https://registry.npmjs.org/${PACKAGE}/latest`, {
      signal: AbortSignal.timeout(5000),
    });
    const data = (await res.json()) as { version?: string };
    const latest = data.version;

    if (!latest) {
      console.error("Could not determine latest version.");
      process.exit(1);
    }

    const current = getCurrentVersion();
    if (latest === current) {
      console.log(`Already up to date (v${current}).`);
      return;
    }

    console.log(`Updating: v${current} → v${latest}\n`);

    // Detect package manager
    const isBun = typeof globalThis.Bun !== "undefined";
    const cmd = isBun
      ? `bun install -g ${PACKAGE}@latest`
      : `npm install -g ${PACKAGE}@latest`;

    console.log(`$ ${cmd}\n`);
    execSync(cmd, { stdio: "inherit" });
    console.log(`\n✓ Updated to v${latest}`);
  } catch (err) {
    console.error(`Update failed: ${err instanceof Error ? err.message : String(err)}`);
    process.exit(1);
  }
}

function getCurrentVersion(): string {
  try {
    const dir = new URL(".", import.meta.url).pathname;
    const pkgPath = join(dir, "package.json");
    if (existsSync(pkgPath)) {
      return JSON.parse(readFileSync(pkgPath, "utf-8")).version ?? "0.0.0";
    }
  } catch { /* ignore */ }
  return "0.0.0";
}

// ─── completions ────────────────────────────────────────────

const COMMANDS = [
  "setup", "init", "doctor", "config", "update", "completions",
  "version", "sessions", "review", "auth", "execute",
];
const FLAGS = [
  "--help", "--version", "--provider", "--model", "--max-iterations",
  "--reasoning", "--resume", "--continue", "--suggest", "--auto-edit",
  "--full-auto", "--verbose", "--quiet", "--file",
];

export function runCompletions(shell: string): void {
  switch (shell) {
    case "bash":
      console.log(bashCompletions());
      console.log("\n# Add to ~/.bashrc:\n#   eval \"$(devagent completions bash)\"");
      break;
    case "zsh":
      console.log(zshCompletions());
      console.log("\n# Add to ~/.zshrc:\n#   eval \"$(devagent completions zsh)\"");
      break;
    case "fish":
      console.log(fishCompletions());
      console.log("\n# Save to ~/.config/fish/completions/devagent.fish:\n#   devagent completions fish > ~/.config/fish/completions/devagent.fish");
      break;
    default:
      console.log("Usage: devagent completions <bash|zsh|fish>");
      console.log("\nExamples:");
      console.log("  eval \"$(devagent completions bash)\"   # Add to ~/.bashrc");
      console.log("  eval \"$(devagent completions zsh)\"    # Add to ~/.zshrc");
      console.log("  devagent completions fish > ~/.config/fish/completions/devagent.fish");
      break;
  }
}

function bashCompletions(): string {
  return `_devagent_completions() {
  local cur="\${COMP_WORDS[COMP_CWORD]}"
  local prev="\${COMP_WORDS[COMP_CWORD-1]}"

  case "\${prev}" in
    devagent)
      COMPREPLY=( $(compgen -W "${COMMANDS.join(" ")} ${FLAGS.join(" ")}" -- "\${cur}") )
      return 0
      ;;
    config)
      COMPREPLY=( $(compgen -W "get set path" -- "\${cur}") )
      return 0
      ;;
    auth)
      COMPREPLY=( $(compgen -W "login status logout" -- "\${cur}") )
      return 0
      ;;
    completions)
      COMPREPLY=( $(compgen -W "bash zsh fish" -- "\${cur}") )
      return 0
      ;;
    --provider)
      COMPREPLY=( $(compgen -W "anthropic openai deepseek openrouter ollama chatgpt github-copilot" -- "\${cur}") )
      return 0
      ;;
    --reasoning)
      COMPREPLY=( $(compgen -W "low medium high" -- "\${cur}") )
      return 0
      ;;
  esac

  if [[ "\${cur}" == -* ]]; then
    COMPREPLY=( $(compgen -W "${FLAGS.join(" ")}" -- "\${cur}") )
  else
    COMPREPLY=( $(compgen -W "${COMMANDS.join(" ")}" -- "\${cur}") )
  fi
}
complete -F _devagent_completions devagent`;
}

function zshCompletions(): string {
  return `#compdef devagent

_devagent() {
  local -a commands flags

  commands=(
${COMMANDS.map((c) => `    '${c}:${c} command'`).join("\n")}
  )

  flags=(
    '--help[Show help]'
    '--version[Show version]'
    '--provider[LLM provider]:provider:(anthropic openai deepseek openrouter ollama chatgpt github-copilot)'
    '--model[Model ID]:model:'
    '--max-iterations[Max iterations]:number:'
    '--reasoning[Reasoning effort]:level:(low medium high)'
    '--resume[Resume session]:session_id:'
    '--continue[Resume most recent session]'
    '--suggest[Suggest mode]'
    '--auto-edit[Auto-edit mode]'
    '--full-auto[Full-auto mode]'
    '--verbose[Verbose output]'
    '--quiet[Quiet output]'
    '--file[Read query from file]:file:_files'
  )

  _arguments -s \\
    '1:command:->command' \\
    '*::arg:->args' \\
    \${flags}

  case \$state in
    command)
      _describe 'command' commands
      ;;
    args)
      case \$words[1] in
        config) _values 'subcommand' get set path ;;
        auth) _values 'subcommand' login status logout ;;
        completions) _values 'shell' bash zsh fish ;;
      esac
      ;;
  esac
}

_devagent`;
}

function fishCompletions(): string {
  const lines = [
    "# devagent completions for fish",
    "complete -c devagent -e",
    "",
    "# Commands",
    ...COMMANDS.map((c) => `complete -c devagent -n '__fish_use_subcommand' -a '${c}' -d '${c}'`),
    "",
    "# Flags",
    "complete -c devagent -l help -s h -d 'Show help'",
    "complete -c devagent -l version -s V -d 'Show version'",
    "complete -c devagent -l provider -x -a 'anthropic openai deepseek openrouter ollama chatgpt github-copilot' -d 'LLM provider'",
    "complete -c devagent -l model -x -d 'Model ID'",
    "complete -c devagent -l max-iterations -x -d 'Max iterations'",
    "complete -c devagent -l reasoning -x -a 'low medium high' -d 'Reasoning effort'",
    "complete -c devagent -l resume -x -d 'Resume session by ID'",
    "complete -c devagent -l continue -d 'Resume most recent session'",
    "complete -c devagent -l suggest -d 'Suggest mode'",
    "complete -c devagent -l auto-edit -d 'Auto-edit mode'",
    "complete -c devagent -l full-auto -d 'Full-auto mode'",
    "complete -c devagent -l verbose -s v -d 'Verbose output'",
    "complete -c devagent -l quiet -s q -d 'Quiet output'",
    "complete -c devagent -l file -s f -r -d 'Read query from file'",
    "",
    "# Subcommands",
    "complete -c devagent -n '__fish_seen_subcommand_from config' -a 'get set path'",
    "complete -c devagent -n '__fish_seen_subcommand_from auth' -a 'login status logout'",
    "complete -c devagent -n '__fish_seen_subcommand_from completions' -a 'bash zsh fish'",
  ];
  return lines.join("\n");
}

const INSTRUCTIONS_TEMPLATE = `# DevAgent Instructions

<!--
  This file provides project-specific instructions to devagent.
  It is loaded automatically when devagent runs in this directory.
-->

## Guidelines

- Follow the project's existing coding conventions
- Prefer editing existing files over creating new ones
- Run tests after making changes
`;
