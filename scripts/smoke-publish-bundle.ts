#!/usr/bin/env bun

import { spawn, spawnSync } from "node:child_process";
import { cpSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";

import type { ChildProcessWithoutNullStreams } from "node:child_process";

const ROOT = resolve(import.meta.dirname, "..");
const DIST = join(ROOT, "dist");
const BUNDLE_PATH = join(DIST, "devagent.js");
const BOOTSTRAP_PATH = join(DIST, "bootstrap.js");

void main().catch((error) => {
  const message = error instanceof Error ? error.message : String(error);
  console.error(message);
  process.exit(1);
});

async function main(): Promise<void> {
  ensureBundleExists();
  verifyBundleMarkers();

  const nodeBin = resolveNodeBinary();
  const npmBin = resolveNpmBinary();
  const stagedDist = stagePublishRuntime();
  installPublishDependencies(stagedDist, npmBin, nodeBin);

  const isolatedHome = mkdtempSync(join(tmpdir(), "devagent-bundle-smoke-"));
  try {
    runSmokeCommand(stagedDist, nodeBin, ["bootstrap.js", "--help"], {
      expectedExitCode: 0,
      description: "help smoke",
      homeDir: isolatedHome,
    });
    runSmokeCommand(stagedDist, nodeBin, ["bootstrap.js", "sessions"], {
      expectedExitCode: 0,
      description: "session database smoke",
      homeDir: isolatedHome,
    });
    runSmokeCommand(stagedDist, nodeBin, ["bootstrap.js", "--provider", "devagent-api", "--model", "cortex", "hello"], {
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

    const stubGateway = await startStubGateway(nodeBin);
    try {
      writeGatewayConfig(isolatedHome, stubGateway.baseUrl);
      runSmokeCommand(stagedDist, nodeBin, ["bootstrap.js", "--provider", "devagent-api", "--model", "cortex", "hello"], {
        expectedExitCode: 1,
        description: "gateway transport smoke",
        homeDir: isolatedHome,
        envOverrides: {
          DEVAGENT_API_KEY: "ilg_smoke_test_key",
        },
        clearedEnvKeys: [
          "OPENAI_API_KEY",
          "ANTHROPIC_API_KEY",
          "OPENROUTER_API_KEY",
          "DEEPSEEK_API_KEY",
        ],
      });
      stubGateway.assertTransportContract();
    } finally {
      await stubGateway.stop();
    }

    verifyTarballReinstall(stagedDist, npmBin, nodeBin, isolatedHome);
  } finally {
    rmSync(stagedDist, { recursive: true, force: true });
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

function stagePublishRuntime(): string {
  const stagedDist = mkdtempSync(join(tmpdir(), "devagent-publish-runtime-"));
  for (const name of ["bootstrap.js", "devagent.js", "package.json", "README.md", "LICENSE"]) {
    const sourcePath = join(DIST, name);
    if (existsSync(sourcePath)) {
      cpSync(sourcePath, join(stagedDist, name));
    }
  }
  return stagedDist;
}

function installPublishDependencies(stagedDist: string, npmBin: string, nodeBin: string): void {
  const install = spawnSync(
    npmBin,
    ["install", "--no-fund", "--no-audit"],
    {
      cwd: stagedDist,
      env: buildNodePreferredEnv(nodeBin),
      encoding: "utf-8",
      stdio: "pipe",
    },
  );
  if (install.status !== 0) {
    const output = `${install.stdout}${install.stderr}`.trim();
    throw new Error(`Failed to install publish-bundle dependencies in staged dist/: ${output}`);
  }
}

function buildNodePreferredEnv(nodeBin: string): NodeJS.ProcessEnv {
  const nodeDir = dirname(nodeBin);
  const pathSeparator = process.platform === "win32" ? ";" : ":";
  return {
    ...process.env,
    PATH: [nodeDir, process.env["PATH"] ?? ""].filter(Boolean).join(pathSeparator),
  };
}

interface SmokeCommandOptions {
  readonly description: string;
  readonly expectedExitCode: number;
  readonly expectedOutput?: string;
  readonly homeDir: string;
  readonly clearedEnvKeys?: ReadonlyArray<string>;
  readonly envOverrides?: Readonly<Record<string, string>>;
}

function runSmokeCommand(
  stagedDist: string,
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
    DEVAGENT_DISABLE_UPDATE_CHECK: "1",
  };
  for (const envKey of options.clearedEnvKeys ?? []) {
    delete env[envKey];
  }
  Object.assign(env, options.envOverrides ?? {});

  const result = spawnSync(nodeBin, args, {
    cwd: stagedDist,
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

function verifyTarballReinstall(
  stagedDist: string,
  npmBin: string,
  nodeBin: string,
  homeDir: string,
): void {
  const packDir = mkdtempSync(join(tmpdir(), "devagent-pack-smoke-"));
  const prefixDir = mkdtempSync(join(tmpdir(), "devagent-prefix-smoke-"));
  try {
    const pack = spawnSync(
      npmBin,
      ["pack", "--pack-destination", packDir],
      {
        cwd: stagedDist,
        env: buildNodePreferredEnv(nodeBin),
        encoding: "utf-8",
        stdio: "pipe",
      },
    );
    if (pack.status !== 0) {
      throw new Error(`Failed to pack staged publish bundle: ${`${pack.stdout}${pack.stderr}`.trim()}`);
    }
    const tarball = pack.stdout
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean)
      .at(-1);
    if (!tarball) {
      throw new Error("npm pack did not produce a tarball name.");
    }
    const tarballPath = join(packDir, tarball);
    installTarballIntoPrefix(npmBin, prefixDir, tarballPath, nodeBin);
    runInstalledBinarySmokeCommand(prefixDir, ["help"], homeDir, nodeBin, 0, "installed help smoke");
    runInstalledBinarySmokeCommand(prefixDir, ["version"], homeDir, nodeBin, 0, "installed version smoke");
    uninstallTarballFromPrefix(npmBin, prefixDir, nodeBin);
    installTarballIntoPrefix(npmBin, prefixDir, tarballPath, nodeBin);
    runInstalledBinarySmokeCommand(prefixDir, ["help"], homeDir, nodeBin, 0, "reinstalled help smoke");
  } finally {
    rmSync(packDir, { recursive: true, force: true });
    rmSync(prefixDir, { recursive: true, force: true });
  }
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
    throw new Error(`Failed to install tarball smoke bundle: ${`${install.stdout}${install.stderr}`.trim()}`);
  }
}

function uninstallTarballFromPrefix(npmBin: string, prefixDir: string, nodeBin: string): void {
  const uninstall = spawnSync(
    npmBin,
    ["uninstall", "-g", "--prefix", prefixDir, "@egavrin/devagent"],
    {
      env: buildNodePreferredEnv(nodeBin),
      encoding: "utf-8",
      stdio: "pipe",
    },
  );
  if (uninstall.status !== 0) {
    throw new Error(`Failed to uninstall tarball smoke bundle: ${`${uninstall.stdout}${uninstall.stderr}`.trim()}`);
  }
}

function runInstalledBinarySmokeCommand(
  prefixDir: string,
  args: string[],
  homeDir: string,
  nodeBin: string,
  expectedExitCode: number,
  description: string,
): void {
  const executable = join(prefixDir, "bin", process.platform === "win32" ? "devagent.cmd" : "devagent");
  const env = {
    ...buildNodePreferredEnv(nodeBin),
    HOME: homeDir,
    XDG_CONFIG_HOME: join(homeDir, ".config"),
    XDG_CACHE_HOME: join(homeDir, ".cache"),
    NO_COLOR: "1",
    FORCE_COLOR: "0",
    DEVAGENT_DISABLE_UPDATE_CHECK: "1",
  };
  const result = spawnSync(executable, args, {
    env,
    encoding: "utf-8",
    stdio: "pipe",
  });
  const output = `${result.stdout}${result.stderr}`;
  if (result.status !== expectedExitCode) {
    throw new Error(
      `${description} exited with ${result.status}, expected ${expectedExitCode}.\n${output.trim()}`,
    );
  }
}

function writeGatewayConfig(homeDir: string, baseUrl: string): void {
  const configDir = join(homeDir, ".config", "devagent");
  mkdirSync(configDir, { recursive: true });
  writeFileSync(
    join(configDir, "config.toml"),
    [
      'provider = "devagent-api"',
      'model = "cortex"',
      "",
      '[providers.devagent-api]',
      `base_url = "${baseUrl}"`,
      "",
    ].join("\n"),
  );
}

interface StubGatewayHandle {
  readonly baseUrl: string;
  readonly assertTransportContract: () => void;
  readonly stop: () => Promise<void>;
}

async function startStubGateway(nodeBin: string): Promise<StubGatewayHandle> {
  const stubDir = mkdtempSync(join(tmpdir(), "devagent-gateway-stub-"));
  const portPath = join(stubDir, "port.txt");
  const requestLogPath = join(stubDir, "requests.log");
  const serverScriptPath = join(stubDir, "server.mjs");

  writeFileSync(
    serverScriptPath,
    [
      'import { createServer } from "node:http";',
      'import { appendFileSync, writeFileSync } from "node:fs";',
      "",
      "const [portPath, requestLogPath] = process.argv.slice(2);",
      "const server = createServer(async (req, res) => {",
      '  appendFileSync(requestLogPath, `${req.method} ${req.url}\\n`);',
      "  for await (const _chunk of req) {",
      "    void _chunk;",
      "  }",
      '  if (req.url === "/v1/chat/completions") {',
      "    res.statusCode = 401;",
      '    res.setHeader("content-type", "application/json");',
      '    res.end(JSON.stringify({ error: { code: "invalid_auth", message: "Missing runtime bearer token." } }));',
      "    return;",
      "  }",
      '  if (req.url === "/v1/responses") {',
      "    res.statusCode = 404;",
      '    res.setHeader("content-type", "text/plain; charset=UTF-8");',
      '    res.end("404 Not Found");',
      "    return;",
      "  }",
      "  res.statusCode = 404;",
      '  res.end("not found");',
      "});",
      'server.listen(0, "127.0.0.1", () => {',
      "  const address = server.address();",
      '  if (!address || typeof address === "string") {',
      '    throw new Error("Failed to resolve stub gateway address");',
      "  }",
      "  writeFileSync(portPath, String(address.port));",
      "});",
      'process.on("SIGTERM", () => server.close(() => process.exit(0)));',
      'process.on("SIGINT", () => server.close(() => process.exit(0)));',
      "",
    ].join("\n"),
  );

  const child = spawn(nodeBin, [serverScriptPath, portPath, requestLogPath], {
    cwd: stubDir,
    stdio: ["ignore", "pipe", "pipe"],
  });
  const output = captureChildOutput(child);

  try {
    const port = await waitForPortFile(portPath, child, output);
    return {
      baseUrl: `http://127.0.0.1:${port}/v1`,
      assertTransportContract: () => {
        const requests = existsSync(requestLogPath)
          ? readFileSync(requestLogPath, "utf-8").split("\n").filter(Boolean)
          : [];
        if (!requests.includes("POST /v1/chat/completions")) {
          throw new Error(
            `gateway transport smoke did not hit /v1/chat/completions.\nRequests:\n${requests.join("\n")}`,
          );
        }
        if (requests.includes("POST /v1/responses")) {
          throw new Error(
            `gateway transport smoke unexpectedly hit /v1/responses.\nRequests:\n${requests.join("\n")}`,
          );
        }
      },
      stop: async () => {
        await stopStubGateway(child);
        rmSync(stubDir, { recursive: true, force: true });
      },
    };
  } catch (error) {
    await stopStubGateway(child);
    rmSync(stubDir, { recursive: true, force: true });
    throw error;
  }
}

function captureChildOutput(child: ChildProcessWithoutNullStreams): { stdout: string; stderr: string } {
  const output = { stdout: "", stderr: "" };
  child.stdout.on("data", (chunk: Buffer | string) => {
    output.stdout += chunk.toString();
  });
  child.stderr.on("data", (chunk: Buffer | string) => {
    output.stderr += chunk.toString();
  });
  return output;
}

async function waitForPortFile(
  portPath: string,
  child: ChildProcessWithoutNullStreams,
  output: { stdout: string; stderr: string },
): Promise<string> {
  const timeoutAt = Date.now() + 5_000;
  while (Date.now() < timeoutAt) {
    if (existsSync(portPath)) {
      return readFileSync(portPath, "utf-8").trim();
    }
    if (child.exitCode !== null) {
      throw new Error(
        `Stub gateway exited before startup.\n${[output.stdout, output.stderr].filter(Boolean).join("\n").trim()}`,
      );
    }
    await new Promise((resolve) => setTimeout(resolve, 50));
  }

  throw new Error(
    `Timed out waiting for stub gateway startup.\n${[output.stdout, output.stderr].filter(Boolean).join("\n").trim()}`,
  );
}

async function stopStubGateway(child: ChildProcessWithoutNullStreams): Promise<void> {
  if (child.exitCode !== null) {
    return;
  }

  child.kill("SIGTERM");
  const timeoutAt = Date.now() + 2_000;
  while (Date.now() < timeoutAt) {
    if (child.exitCode !== null) {
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 50));
  }

  child.kill("SIGKILL");
}
