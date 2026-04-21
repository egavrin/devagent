#!/usr/bin/env bun

import { spawnSync } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { mkdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";

import {
  CredentialStore,
  getProviderCredentialDescriptor,
  type CredentialInfo,
} from "../../packages/runtime/src/index.ts";

type SupportedProvider =
  | "devagent-api"
  | "openai"
  | "openrouter"
  | "deepseek"
  | "chatgpt"
  | "github-copilot";

interface CliOptions {
  readonly outputRoot?: string;
  readonly provider?: SupportedProvider;
}

interface TuiProviderSelection {
  readonly provider: SupportedProvider;
  readonly model: string;
  readonly credential: CredentialInfo;
}

const ROOT = resolve(import.meta.dirname, "..", "..");
const DIST = join(ROOT, "dist");

const DEFAULT_MODELS: Readonly<Record<SupportedProvider, string>> = {
  "devagent-api": "cortex",
  openai: "gpt-5.4-mini",
  openrouter: "openai/gpt-4o-mini",
  deepseek: "deepseek-chat",
  chatgpt: "gpt-5.4-mini",
  "github-copilot": "gpt-4o",
};

function parseArgs(argv: string[]): CliOptions {
  let outputRoot: string | undefined;
  let provider: SupportedProvider | undefined;

  for (let index = 2; index < argv.length; index++) {
    const arg = argv[index]!;
    if (arg === "--output-dir" && argv[index + 1]) {
      outputRoot = argv[++index]!;
      continue;
    }
    if (arg === "--provider" && argv[index + 1]) {
      const value = argv[++index]!;
      if (!(value in DEFAULT_MODELS)) {
        throw new Error(`Unsupported TUI validation provider: ${value}`);
      }
      provider = value as SupportedProvider;
      continue;
    }
    if (arg === "--help" || arg === "-h") {
      process.stdout.write(
        [
          "Tarball TUI validator",
          "",
          "Usage:",
          "  bun run scripts/live-validation/tui-validator.ts",
          "  bun run scripts/live-validation/tui-validator.ts --provider devagent-api",
          "  bun run scripts/live-validation/tui-validator.ts --output-dir <path>",
          "",
        ].join("\n"),
      );
      process.exit(0);
    }
    throw new Error(`Unknown argument: ${arg}`);
  }

  return { outputRoot, provider };
}

function ensureBundleExists(): void {
  for (const requiredPath of [join(DIST, "bootstrap.js"), join(DIST, "devagent.js"), join(DIST, "package.json")]) {
    if (!existsSync(requiredPath)) {
      throw new Error(`Missing publish artifact: ${requiredPath}. Run "bun run build:publish" before TUI validation.`);
    }
  }
}

function resolveNpmBinary(): string {
  const explicit = process.env["DEVAGENT_NPM_BIN"];
  if (explicit) {
    validateCommandVersion(explicit, ["--version"], "npm");
    return explicit;
  }

  const candidates = uniquePathCandidates("npm");
  for (const candidate of candidates) {
    if (validateCommandVersion(candidate, ["--version"], "npm", false)) {
      return candidate;
    }
  }

  throw new Error("Could not find npm for tarball TUI validation. Set DEVAGENT_NPM_BIN to an npm executable.");
}

function resolveNodeBinary(): string {
  const explicit = process.env["DEVAGENT_NODE_BIN"];
  if (explicit) {
    validateRealNodeBinary(explicit);
    return explicit;
  }

  const candidates = uniquePathCandidates("node");
  for (const candidate of candidates) {
    if (validateRealNodeBinary(candidate, false)) {
      return candidate;
    }
  }

  throw new Error(
    "Could not find a real Node.js binary for tarball TUI validation. Set DEVAGENT_NODE_BIN to Node >= 20.",
  );
}

function uniquePathCandidates(commandName: string): string[] {
  const envPath = process.env["PATH"] ?? "";
  const separator = process.platform === "win32" ? ";" : ":";
  const suffix = process.platform === "win32" ? ".exe" : "";
  return [...new Set(
    envPath
      .split(separator)
      .filter(Boolean)
      .map((dirPath) => join(dirPath, `${commandName}${suffix}`)),
  )];
}

function validateCommandVersion(
  candidate: string,
  args: string[],
  label: string,
  throwOnFailure: boolean = true,
): boolean {
  const result = spawnSync(candidate, args, { encoding: "utf-8" });
  if (result.status === 0) {
    return true;
  }
  if (throwOnFailure) {
    throw new Error(`Failed to run ${label} command ${candidate}: ${`${result.stdout}${result.stderr}`.trim()}`);
  }
  return false;
}

function validateRealNodeBinary(candidate: string, throwOnFailure: boolean = true): boolean {
  const releaseName = spawnSync(
    candidate,
    ["-p", "process.versions?.bun ? 'bun' : (process.release?.name ?? '')"],
    { encoding: "utf-8" },
  );
  if (releaseName.status !== 0 || releaseName.stdout.trim() !== "node") {
    if (throwOnFailure) {
      throw new Error(`Expected a real Node.js binary, got: ${candidate}`);
    }
    return false;
  }

  const majorCheck = spawnSync(
    candidate,
    ["-p", "const major = Number.parseInt(process.versions.node.split('.')[0], 10); if (major >= 20) { console.log('ok'); } else { process.exit(1); }"],
    { encoding: "utf-8" },
  );
  if (majorCheck.status !== 0) {
    if (throwOnFailure) {
      throw new Error(`Node.js >= 20 is required for tarball TUI validation: ${candidate}`);
    }
    return false;
  }

  return true;
}

function buildNodePreferredEnv(nodeBin: string): NodeJS.ProcessEnv {
  const nodeDir = dirname(nodeBin);
  const pathSeparator = process.platform === "win32" ? ";" : ":";
  return {
    ...process.env,
    PATH: [nodeDir, process.env["PATH"] ?? ""].filter(Boolean).join(pathSeparator),
  };
}

function isCredentialExpired(credential: CredentialInfo): boolean {
  return credential.type === "oauth"
    && credential.expiresAt !== undefined
    && credential.expiresAt <= Date.now();
}

function resolveProviderSelection(preferredProvider?: SupportedProvider): TuiProviderSelection {
  const storedCredentials = new CredentialStore().all();
  const providerOrder = preferredProvider
    ? [preferredProvider]
    : ["devagent-api", "chatgpt", "openai", "github-copilot", "openrouter", "deepseek"] as SupportedProvider[];

  for (const provider of providerOrder) {
    const descriptor = getProviderCredentialDescriptor(provider);
    if (!descriptor || descriptor.credentialMode === "none") {
      continue;
    }
    const credential = storedCredentials[provider];
    if (!credential || isCredentialExpired(credential)) {
      continue;
    }
    return {
      provider,
      model: DEFAULT_MODELS[provider],
      credential,
    };
  }

  const hint = preferredProvider
    ? `No stored non-expired credential found for ${preferredProvider}.`
    : "No stored non-expired credentials found for devagent-api, chatgpt, openai, github-copilot, openrouter, or deepseek.";
  throw new Error(`${hint} Run "devagent auth login" first.`);
}

function packTarball(npmBin: string, nodeBin: string, outputDir: string): string {
  const packDir = mkdtempSync(join(outputDir, "tarball-"));
  const pack = spawnSync(
    npmBin,
    ["pack", "--pack-destination", packDir],
    {
      cwd: DIST,
      env: buildNodePreferredEnv(nodeBin),
      encoding: "utf-8",
      stdio: "pipe",
    },
  );
  if (pack.status !== 0) {
    throw new Error(`Failed to pack dist tarball for TUI validation: ${`${pack.stdout}${pack.stderr}`.trim()}`);
  }
  const tarball = pack.stdout
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .at(-1);
  if (!tarball) {
    throw new Error("npm pack did not report a tarball filename for TUI validation.");
  }
  return join(packDir, tarball);
}

function installTarballIntoPrefix(npmBin: string, prefixDir: string, tarballPath: string, nodeBin: string): void {
  const install = spawnSync(
    npmBin,
    ["install", "-g", "--prefix", prefixDir, tarballPath],
    {
      env: buildNodePreferredEnv(nodeBin),
      encoding: "utf-8",
      stdio: "pipe",
    },
  );
  if (install.status !== 0) {
    throw new Error(`Failed to install TUI validation tarball: ${`${install.stdout}${install.stderr}`.trim()}`);
  }
}

async function seedCredential(homeDir: string, provider: SupportedProvider, credential: CredentialInfo): Promise<void> {
  const configDir = join(homeDir, ".config", "devagent");
  await mkdir(configDir, { recursive: true });
  const store = new CredentialStore({
    filePath: join(configDir, "credentials.json"),
  });
  store.set(provider, credential);
}

function normalizeTranscript(text: string): string {
  return text
    .replace(/\u001B\][^\u0007]*(?:\u0007|\u001B\\)/g, "")
    .replace(/\u001B\[[0-9;?]*[ -/]*[@-~]/g, "")
    .replace(/\u001B[@-_]/g, "")
    .replace(/\r/g, "");
}

function countOccurrences(text: string, needle: string): number {
  if (needle.length === 0) return 0;
  let count = 0;
  let index = text.indexOf(needle);
  while (index !== -1) {
    count++;
    index = text.indexOf(needle, index + needle.length);
  }
  return count;
}

function transcriptMatchesPattern(text: string, pattern: string): boolean {
  try {
    return new RegExp(pattern).test(normalizeTranscript(text));
  } catch {
    return normalizeTranscript(text).includes(pattern);
  }
}

function escapeExpectDoubleQuoted(value: string): string {
  return value
    .replace(/\\/g, "\\\\")
    .replace(/\r/g, "\\r")
    .replace(/\n/g, "\\n")
    .replace(/"/g, '\\"')
    .replace(/\$/g, "\\$");
}

export function extractSettledFrame(text: string): string {
  const normalized = normalizeTranscript(text).trim();
  if (normalized.length === 0) {
    return normalized;
  }

  const welcomeAnchor = "╔══════════════════╗";
  const promptAnchor = "╭────────────────";
  const welcomeIndex = normalized.lastIndexOf(welcomeAnchor);
  const promptIndex = normalized.lastIndexOf(promptAnchor);
  const startIndex = welcomeIndex !== -1 ? welcomeIndex : promptIndex;
  if (startIndex === -1) {
    return normalized;
  }
  return normalized.slice(startIndex).trim();
}

export function assertTuiFrame(
  frame: string,
  options: {
    readonly expectedVersion: string;
    readonly requiredText: string | RegExp;
  },
): void {
  if (!frame.includes("devagent")) {
    throw new Error("TUI frame did not include the devagent banner.");
  }
  if (!frame.includes(`v${options.expectedVersion}`)) {
    throw new Error(`TUI frame did not include expected version v${options.expectedVersion}.`);
  }
  const hasRequiredText = typeof options.requiredText === "string"
    ? frame.includes(options.requiredText)
    : options.requiredText.test(frame);
  if (!hasRequiredText) {
    throw new Error(`TUI frame did not include expected text: ${String(options.requiredText)}`);
  }
  if (!(frame.includes("Type /help for commands") || frame.includes("Type /help for all commands"))) {
    throw new Error("TUI frame did not include help guidance.");
  }
  if (!frame.includes("Shift+Tab")) {
    throw new Error("TUI frame did not include Shift+Tab safety guidance.");
  }
  if (countOccurrences(frame, "║    devagent") !== 1) {
    throw new Error("TUI frame contained duplicate welcome/header blocks.");
  }
  if (/workspace-[^\n]*╔/.test(frame)) {
    throw new Error("TUI frame contained a status-bar/banner collision.");
  }
  if (!/╭[─]+╮[\s\S]*❯[\s\S]*╰[─]+╯/.test(frame)) {
    throw new Error("TUI frame did not contain an intact prompt box.");
  }
}

async function runTuiTranscript(
  outputRoot: string,
  executable: string,
  args: string[],
  cwd: string,
  env: NodeJS.ProcessEnv,
  transcriptPath: string,
  typedCommand?: string,
  expectedOutputPattern?: string,
): Promise<void> {
  const expectScriptPath = join(outputRoot, "tui.expect");
  const expectLines = [
    "#!/usr/bin/expect -f",
    "set timeout 10",
    "match_max 200000",
    "set transcript [lindex $argv 0]",
    "set executable [lindex $argv 1]",
    "set args [lrange $argv 2 end]",
    "log_user 1",
    "spawn -noecho $executable {*}$args",
    'expect { -re "Type /help|Shift\\+Tab toggles default and autopilot|Shift\\+Tab safety" {} timeout { exit 124 } }',
  ];
  if (typedCommand) {
    expectLines.push("after 300");
    expectLines.push(`send -- "${escapeExpectDoubleQuoted(`${typedCommand}\n`)}"`);
  }
  if (expectedOutputPattern) {
    expectLines.push(`expect { -re "${escapeExpectDoubleQuoted(expectedOutputPattern)}" { puts "\\nVALIDATOR_MATCH: $expect_out(0,string)" } timeout { exit 125 } }`);
  }
  expectLines.push(
    "after 300",
    'send -- "\\003"',
    "expect eof",
    "set wait_status [wait]",
    "exit [lindex $wait_status 3]",
    "",
  );
  await writeFile(expectScriptPath, expectLines.join("\n"));

  const result = spawnSync(
    "expect",
    [expectScriptPath, transcriptPath, executable, ...args],
    {
      cwd,
      env,
      encoding: "utf-8",
      stdio: "pipe",
      timeout: 20_000,
    },
  );
  if (result.stdout.trim().length > 0) {
    await writeFile(transcriptPath, result.stdout);
  }
  const observedExpectedOutput = expectedOutputPattern
    ? result.stdout.includes("VALIDATOR_MATCH:") || transcriptMatchesPattern(result.stdout, expectedOutputPattern)
    : true;
  if (expectedOutputPattern && !observedExpectedOutput) {
    throw new Error(`TUI validator did not observe expected output: ${expectedOutputPattern}`);
  }
  if (result.status !== 0 && result.status !== 130 && result.status !== 143 && !(result.status === 125 && observedExpectedOutput)) {
    throw new Error(`TUI validator exited with ${result.status}.${result.stderr ? `\n${result.stderr.trim()}` : ""}`);
  }
}

async function main(): Promise<void> {
  const options = parseArgs(process.argv);
  ensureBundleExists();

  const nodeBin = resolveNodeBinary();
  const npmBin = resolveNpmBinary();
  const selection = resolveProviderSelection(options.provider);
  const outputRoot = options.outputRoot
    ? resolve(options.outputRoot)
    : mkdtempSync(join(tmpdir(), "devagent-live-tui-"));
  await mkdir(outputRoot, { recursive: true });

  const tarballPath = packTarball(npmBin, nodeBin, outputRoot);
  const prefixDir = mkdtempSync(join(outputRoot, "prefix-"));
  const homeDir = mkdtempSync(join(outputRoot, "home-"));
  const workspaceDir = mkdtempSync(join(outputRoot, "workspace-"));
  const transcriptPath = join(outputRoot, "tui-transcript.raw.txt");
  const sessionsTranscriptPath = join(outputRoot, "tui-sessions.raw.txt");
  const clearTranscriptPath = join(outputRoot, "tui-clear.raw.txt");
  const normalizedTranscriptPath = join(outputRoot, "tui-transcript.txt");
  const helpFramePath = join(outputRoot, "tui-help.frame.txt");
  const sessionsFramePath = join(outputRoot, "tui-sessions.frame.txt");
  const clearFramePath = join(outputRoot, "tui-clear.frame.txt");

  try {
    installTarballIntoPrefix(npmBin, prefixDir, tarballPath, nodeBin);
    await seedCredential(homeDir, selection.provider, selection.credential);
    const expectedVersion = JSON.parse(readFileSync(join(DIST, "package.json"), "utf-8")).version as string;

    const executable = nodeBin;
    const installedBootstrap = join(
      prefixDir,
      "lib",
      "node_modules",
      "@egavrin",
      "devagent",
      "bootstrap.js",
    );
    const env: NodeJS.ProcessEnv = {
      ...buildNodePreferredEnv(nodeBin),
      HOME: homeDir,
      XDG_CONFIG_HOME: join(homeDir, ".config"),
      XDG_CACHE_HOME: join(homeDir, ".cache"),
      NO_COLOR: "1",
      FORCE_COLOR: "0",
      TERM: process.env["TERM"] ?? "xterm-256color",
      COLUMNS: "120",
      LINES: "40",
      DEVAGENT_DISABLE_UPDATE_CHECK: "1",
    };

    await runTuiTranscript(
      outputRoot,
      executable,
      [installedBootstrap, "--provider", selection.provider, "--model", selection.model, "--mode", "default"],
      workspaceDir,
      env,
      transcriptPath,
      "/help",
      "Commands: /clear",
    );
    await runTuiTranscript(
      outputRoot,
      executable,
      [installedBootstrap, "--provider", selection.provider, "--model", selection.model, "--mode", "default"],
      workspaceDir,
      env,
      sessionsTranscriptPath,
      "/sessions",
      "No sessions found\\.|Recent sessions:",
    );
    await runTuiTranscript(
      outputRoot,
      executable,
      [installedBootstrap, "--provider", selection.provider, "--model", selection.model, "--mode", "default"],
      workspaceDir,
      env,
      clearTranscriptPath,
      "/clear",
      "Context cleared\\.",
    );

    const helpFrame = extractSettledFrame(readFileSync(transcriptPath, "utf-8"));
    const sessionsFrame = extractSettledFrame(readFileSync(sessionsTranscriptPath, "utf-8"));
    const clearFrame = extractSettledFrame(readFileSync(clearTranscriptPath, "utf-8"));

    await writeFile(helpFramePath, helpFrame);
    await writeFile(sessionsFramePath, sessionsFrame);
    await writeFile(clearFramePath, clearFrame);
    await writeFile(
      normalizedTranscriptPath,
      [
        "=== /help ===",
        helpFrame,
        "",
        "=== /sessions ===",
        sessionsFrame,
        "",
        "=== /clear ===",
        clearFrame,
      ].join("\n"),
    );

    assertTuiFrame(helpFrame, { expectedVersion, requiredText: "Commands: /clear" });
    assertTuiFrame(sessionsFrame, { expectedVersion, requiredText: /No sessions found\.|Recent sessions:/ });
    assertTuiFrame(clearFrame, { expectedVersion, requiredText: "Context cleared." });

    process.stdout.write(
      `Validated tarball TUI with provider ${selection.provider}. Transcript: ${normalizedTranscriptPath}\n`,
    );
  } finally {
    rmSync(prefixDir, { recursive: true, force: true });
    rmSync(homeDir, { recursive: true, force: true });
    rmSync(workspaceDir, { recursive: true, force: true });
  }
}

if (import.meta.main) {
  void main().catch((error) => {
    const message = error instanceof Error ? error.message : String(error);
    console.error(message);
    process.exit(1);
  });
}
