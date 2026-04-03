import { describe, expect, it } from "vitest";
import { ApprovalMode } from "@devagent/runtime";
import type { DevAgentConfig } from "@devagent/runtime";
import type { DoctorReportInput } from "./commands.js";
import { buildDoctorReport, renderDoctorReport } from "./commands.js";

function makeConfig(overrides: Partial<DevAgentConfig> = {}): DevAgentConfig {
  return {
    provider: "openai",
    model: "gpt-4.1",
    providers: {},
    approval: {
      mode: ApprovalMode.SUGGEST,
      auditLog: false,
      toolOverrides: {},
      pathRules: [],
    },
    budget: {
      maxIterations: 30,
      maxContextTokens: 100_000,
      responseHeadroom: 2_000,
      costWarningThreshold: 1,
      enableCostTracking: true,
    },
    context: {
      pruningStrategy: "hybrid",
      triggerRatio: 0.9,
      keepRecentMessages: 40,
      turnIsolation: true,
      midpointBriefingInterval: 15,
      briefingStrategy: "auto",
    },
    arkts: {
      enabled: false,
      strictMode: false,
      targetVersion: "5.0",
    },
    ...overrides,
  };
}

function makeInput(overrides: Partial<DoctorReportInput> = {}): DoctorReportInput {
  const config = overrides.config ?? makeConfig();
  return {
    version: "0.1.0",
    runtimeLabel: "Bun 1.3.9",
    configPath: "/tmp/.config/devagent/config.toml",
    configSearchPaths: ["/tmp/.config/devagent/config.toml"],
    config,
    providerStatuses: [
      { id: "openai", hint: "set OPENAI_API_KEY or devagent auth login", active: config.provider === "openai", hasCredential: true },
      { id: "devagent-api", hint: "set DEVAGENT_API_KEY or devagent auth login", active: config.provider === "devagent-api", hasCredential: false },
      { id: "chatgpt", hint: "devagent auth login (ChatGPT Plus/Pro)", active: config.provider === "chatgpt", hasCredential: false },
      { id: "ollama", hint: "local — no API key needed (ollama must be running)", active: config.provider === "ollama", hasCredential: true },
    ],
    modelRegistryCount: 110,
    modelRegistered: true,
    modelOwner: "openai",
    lspStatuses: [
      { label: "TypeScript/JavaScript", found: true, install: "npm i -g typescript-language-server typescript" },
      { label: "Python (Pyright)", found: true, install: "npm i -g pyright" },
      { label: "C/C++ (clangd)", found: true, install: "apt install clangd / brew install llvm" },
      { label: "Rust", found: true, install: "rustup component add rust-analyzer" },
      { label: "Bash/Shell", found: true, install: "npm i -g bash-language-server" },
    ],
    platformLabel: "darwin arm64",
    providerSource: "config",
    modelSource: "config",
    credentialSource: "missing",
    ...overrides,
  };
}

describe("doctor report", () => {
  it("prioritizes provider/model mismatch and shows gateway next steps", () => {
    const report = buildDoctorReport(makeInput({
      config: makeConfig({ provider: "openai", model: "cortex" }),
      providerStatuses: [
        { id: "openai", hint: "set OPENAI_API_KEY or devagent auth login", active: true, hasCredential: false },
        { id: "devagent-api", hint: "set DEVAGENT_API_KEY or devagent auth login", active: false, hasCredential: false },
        { id: "ollama", hint: "local — no API key needed (ollama must be running)", active: false, hasCredential: true },
      ],
      providerCredentialIssue: {
        status: "advisory",
        detail: "no API key (set OPENAI_API_KEY or run devagent auth login). Secondary until provider/model pairing is fixed.",
      },
      modelOwner: "devagent-api",
      providerModelIssue: {
        model: "cortex",
        configuredProvider: "openai",
        expectedProvider: "devagent-api",
      },
    }));

    const output = renderDoctorReport(report);

    expect(output).toBe(`devagent v0.1.0

Blocking issues:

  - Provider/model pairing: Configured model "cortex" belongs to provider "devagent-api"; current provider is "openai". Switch provider or choose a model registered for "openai".

What to do next:

  Provider/model pairing:
    Run now: devagent --provider devagent-api --model cortex "<your prompt>"
    Set in ~/.config/devagent/config.toml:
    provider = "devagent-api"
    model = "cortex"
    Export credentials: export DEVAGENT_API_KEY=ilg_...
    Or store credentials: devagent auth login

Effective config:

  Provider: openai (config)
  Model: cortex (config)
  Credential: missing
  Model owner: devagent-api

Checks:

  ✓ Runtime: Bun 1.3.9
  ✓ Git
  ✓ Config file
  ! Provider: openai: no API key (set OPENAI_API_KEY or run devagent auth login). Secondary until provider/model pairing is fixed.

  Available providers:
    · openai (active) — set OPENAI_API_KEY or devagent auth login
    · devagent-api — set DEVAGENT_API_KEY or devagent auth login
    ✓ ollama — local — no API key needed (ollama must be running)

  ✓ Model registry: 110 models loaded
  ✓ Model: cortex
  ✗ Provider/model pairing: Configured model "cortex" belongs to provider "devagent-api"; current provider is "openai". Switch provider or choose a model registered for "openai". Try "--provider devagent-api --model cortex" for the deployed Devagent API gateway.
  LSP servers:
    ✓ TypeScript/JavaScript
    ✓ Python (Pyright)
    ✓ C/C++ (clangd)
    ✓ Rust
    ✓ Bash/Shell

  ✓ Platform: darwin arm64

Some checks failed.`);
  });

  it("omits remediation block when doctor passes", () => {
    const report = buildDoctorReport(makeInput({
      config: makeConfig({ provider: "devagent-api", model: "cortex" }),
      providerStatuses: [
        { id: "openai", hint: "set OPENAI_API_KEY or devagent auth login", active: false, hasCredential: false },
        { id: "devagent-api", hint: "set DEVAGENT_API_KEY or devagent auth login", active: true, hasCredential: true },
        { id: "ollama", hint: "local — no API key needed (ollama must be running)", active: false, hasCredential: true },
      ],
      modelOwner: "devagent-api",
      credentialSource: "env (DEVAGENT_API_KEY)",
    }));

    const output = renderDoctorReport(report);

    expect(report.ok).toBe(true);
    expect(output).toBe(`devagent v0.1.0

Effective config:

  Provider: devagent-api (config)
  Model: cortex (config)
  Credential: env (DEVAGENT_API_KEY)
  Model owner: devagent-api

Checks:

  ✓ Runtime: Bun 1.3.9
  ✓ Git
  ✓ Config file
  ✓ Provider: devagent-api

  Available providers:
    · openai — set OPENAI_API_KEY or devagent auth login
    ✓ devagent-api (active) — set DEVAGENT_API_KEY or devagent auth login
    ✓ ollama — local — no API key needed (ollama must be running)

  ✓ Model registry: 110 models loaded
  ✓ Model: cortex
  ✓ Provider/model pairing
  LSP servers:
    ✓ TypeScript/JavaScript
    ✓ Python (Pyright)
    ✓ C/C++ (clangd)
    ✓ Rust
    ✓ Bash/Shell

  ✓ Platform: darwin arm64

All checks passed.`);
  });

  it("shows provider-specific next steps for missing credentials", () => {
    const report = buildDoctorReport(makeInput({
      config: makeConfig({ provider: "openai", model: "gpt-4.1" }),
      providerStatuses: [
        { id: "openai", hint: "set OPENAI_API_KEY or devagent auth login", active: true, hasCredential: false },
        { id: "devagent-api", hint: "set DEVAGENT_API_KEY or devagent auth login", active: false, hasCredential: false },
        { id: "ollama", hint: "local — no API key needed (ollama must be running)", active: false, hasCredential: true },
      ],
      providerCredentialIssue: {
        status: "blocking",
        detail: "no API key (set OPENAI_API_KEY or run devagent auth login)",
      },
    }));

    const output = renderDoctorReport(report);

    expect(output).toContain("Blocking issues:");
    expect(output).toContain("Effective config:");
    expect(output).toContain("  Provider: openai (config)");
    expect(output).toContain("  Model: gpt-4.1 (config)");
    expect(output).toContain("  Credential: missing");
    expect(output).toContain("  Model owner: openai");
    expect(output).toContain('Provider credentials: no API key (set OPENAI_API_KEY or run devagent auth login)');
    expect(output).toContain("Export credentials: export OPENAI_API_KEY=<your_api_key>");
    expect(output).toContain("Or store credentials: devagent auth login");
    expect(output).toContain('Then retry: devagent "<your prompt>"');
    expect(output).toContain("Some checks failed.");
  });
});
