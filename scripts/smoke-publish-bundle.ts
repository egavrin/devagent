#!/usr/bin/env bun

import { spawnSync } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const ROOT = resolve(import.meta.dirname, "..");
const DIST = join(ROOT, "dist");
const BUNDLE_PATH = join(DIST, "devagent.js");
const BOOTSTRAP_PATH = join(DIST, "bootstrap.js");

main();

function main(): void {
  ensureBundleExists();
  verifyBundleMarkers();

  const nodeBin = resolveNodeBinary();
  const npmBin = resolveNpmBinary();

  installPublishDependencies(npmBin);

  const isolatedHome = mkdtempSync(join(tmpdir(), "devagent-bundle-smoke-"));
  try {
    runSmokeCommand(nodeBin, ["bootstrap.js", "--help"], {
      expectedExitCode: 0,
      description: "help smoke",
      homeDir: isolatedHome,
    });
    runSmokeCommand(nodeBin, ["bootstrap.js", "sessions", "list"], {
      expectedExitCode: 0,
      description: "session database smoke",
      homeDir: isolatedHome,
    });
    runSmokeCommand(nodeBin, ["bootstrap.js", "--provider", "devagent-api", "--model", "cortex", "hello"], {
      expectedExitCode: 1,
      expectedOutput: 'No API key configured for provider "devagent-api".',
      description: "provider startup smoke",
      homeDir: isolatedHome,
      clearedEnvKeys: [
        "DEVAGENT_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENROUTER_API_KEY",
        "DEEPSEEK_API_KEY",
      ],
    });
  } finally {
    rmSync(isolatedHome, { recursive: true, force: true });
  }
}

function ensureBundleExists(): void {
  for (const requiredPath of [BUNDLE_PATH, BOOTSTRAP_PATH, join(DIST, "package.json")]) {
    if (!existsSync(requiredPath)) {
      throw new Error(
        `Missing publish artifact: ${requiredPath}. Run "bun run build:publish" before bundle smoke tests.`,
      );
    }
  }
}

function verifyBundleMarkers(): void {
  const bundle = readFileSync(BUNDLE_PATH, "utf-8");
  assertNoAsyncLazyInitNearAnchor(bundle, {
    anchor: "Failed to open session database:",
    label: "SessionStore startup",
  });
  assertNoAsyncLazyInitNearAnchor(bundle, {
    anchor: "registerDeferred",
    label: "tool registry startup",
  });
}

function assertNoAsyncLazyInitNearAnchor(
  bundle: string,
  options: { anchor: string; label: string },
): void {
  const index = bundle.indexOf(options.anchor);
  if (index === -1) {
    throw new Error(`Bundle verification anchor missing for ${options.label}: ${options.anchor}`);
  }

  const windowStart = Math.max(0, index - 20_000);
  const precedingWindow = bundle.slice(windowStart, index);
  if (precedingWindow.includes("=y(async()=>")) {
    throw new Error(
      `Bundle verification failed for ${options.label}: found async lazy-init wrapper near "${options.anchor}".`,
    );
  }
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
    "Could not find a real Node.js binary. Set DEVAGENT_NODE_BIN to a Node >= 20 executable instead of Bun's node shim.",
  );
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

  throw new Error(
    "Could not find npm for publish-bundle smoke tests. Set DEVAGENT_NPM_BIN to an npm executable.",
  );
}

function validateRealNodeBinary(candidate: string, throwOnFailure: boolean = true): boolean {
  const releaseName = spawnSync(
    candidate,
    ["-p", "process.release?.name ?? ''"],
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
      throw new Error(`Node.js >= 20 is required for bundle smoke tests: ${candidate}`);
    }
    return false;
  }

  return true;
}

function validateCommandVersion(
  candidate: string,
  args: string[],
  label: string,
  throwOnFailure: boolean = true,
): boolean {
  const result = spawnSync(candidate, args, { encoding: "utf-8" });
  if (result.status === 0) return true;
  if (throwOnFailure) {
    const output = `${result.stdout}${result.stderr}`.trim();
    throw new Error(`Failed to run ${label} command ${candidate}: ${output}`);
  }
  return false;
}

function uniquePathCandidates(commandName: string): string[] {
  const envPath = process.env["PATH"] ?? "";
  const separator = process.platform === "win32" ? ";" : ":";
  const suffix = process.platform === "win32" ? ".exe" : "";
  const candidates = envPath
    .split(separator)
    .filter(Boolean)
    .map((dirPath) => join(dirPath, `${commandName}${suffix}`));

  return [...new Set(candidates)];
}

function installPublishDependencies(npmBin: string): void {
  const betterSqlite3Path = join(DIST, "node_modules", "better-sqlite3");
  if (existsSync(betterSqlite3Path)) {
    return;
  }

  const install = spawnSync(
    npmBin,
    ["install", "--no-fund", "--no-audit"],
    {
      cwd: DIST,
      encoding: "utf-8",
      stdio: "pipe",
    },
  );
  if (install.status !== 0) {
    const output = `${install.stdout}${install.stderr}`.trim();
    throw new Error(`Failed to install publish-bundle dependencies in dist/: ${output}`);
  }
}

interface SmokeCommandOptions {
  readonly description: string;
  readonly expectedExitCode: number;
  readonly expectedOutput?: string;
  readonly homeDir: string;
  readonly clearedEnvKeys?: ReadonlyArray<string>;
}

function runSmokeCommand(
  nodeBin: string,
  args: string[],
  options: SmokeCommandOptions,
): void {
  const env = {
    ...process.env,
    HOME: options.homeDir,
    XDG_CONFIG_HOME: join(options.homeDir, ".config"),
    XDG_CACHE_HOME: join(options.homeDir, ".cache"),
    NO_COLOR: "1",
    FORCE_COLOR: "0",
  };
  for (const envKey of options.clearedEnvKeys ?? []) {
    delete env[envKey];
  }

  const result = spawnSync(nodeBin, args, {
    cwd: DIST,
    env,
    encoding: "utf-8",
    stdio: "pipe",
  });
  const output = `${result.stdout}${result.stderr}`;

  if (result.status !== options.expectedExitCode) {
    throw new Error(
      `${options.description} exited with ${result.status}, expected ${options.expectedExitCode}.\n${output.trim()}`,
    );
  }

  for (const crashMarker of ["is not iterable", "is not a constructor"]) {
    if (output.includes(crashMarker)) {
      throw new Error(`${options.description} surfaced a startup crash marker: ${crashMarker}\n${output.trim()}`);
    }
  }

  if (options.expectedOutput && !output.includes(options.expectedOutput)) {
    throw new Error(
      `${options.description} did not include expected output: ${options.expectedOutput}\n${output.trim()}`,
    );
  }
}
