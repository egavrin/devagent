/**
 * Wire DoubleCheck providers — adapters that connect LSPClient diagnostics
 * and shell-based test execution to the DoubleCheck validation system.
 *
 * Supports multi-language LSP routing: multiple LSP servers can run
 * simultaneously, each handling different file types.
 */

import { extractErrorMessage, LANGUAGE_EXTENSIONS, LSPClient, spawnAndCapture } from "@devagent/runtime";
import { execFile } from "node:child_process";
import { extname } from "node:path";

import type { DiagnosticProvider, DoubleCheck, LSPServerConfig, TestRunner } from "@devagent/runtime";

// ─── Language Map ───────────────────────────────────────────

interface CompilerFallbackCheck {
  /** Command to run (e.g. "npx", "cargo", "gcc") */
  readonly command: string;
  /** Arguments for the command */
  readonly args: ReadonlyArray<string>;
  /**
   * Regex to parse diagnostic lines from compiler output.
   * Must have named groups: file, line, col (optional), message.
   */
  readonly diagnosticPattern: RegExp;
  /** Max time to wait for the compiler in ms */
  readonly timeout: number;
  /** Retry command if primary fails (e.g. "tsc" as fallback for "npx tsc") */
  readonly retryCommand?: string;
  /** Args for the retry command */
  readonly retryArgs?: ReadonlyArray<string>;
}

interface LanguageEntry {
  readonly languageId: string;
  readonly extensions: ReadonlyArray<string>;
  readonly defaultCommand: string;
  readonly defaultArgs: ReadonlyArray<string>;
  /** Fallback compiler/checker when no LSP is running */
  readonly fallbackCheck?: CompilerFallbackCheck;
}

// ─── Diagnostic patterns for compiler fallback output parsing ──

/** tsc: src/foo.ts(10,5): error TS2451: Cannot redeclare block-scoped variable */
const TSC_PATTERN = /^(?<file>.+)\((?<line>\d+),(?<col>\d+)\):\s*(?<sev>error|warning)\s+\S+:\s*(?<message>.+)$/;

/** pyright: /path/file.py:10:5 - error: Message */
const PYRIGHT_PATTERN = /^(?<file>.+):(?<line>\d+):(?<col>\d+)\s*-\s*(?<sev>error|warning|information):\s*(?<message>.+)$/;

/** gcc/clang: file.c:10:5: error: message */
const GCC_PATTERN = /^(?<file>.+):(?<line>\d+):(?<col>\d+):\s*(?<sev>error|warning):\s*(?<message>.+)$/;

/** Built-in map of supported languages with default LSP servers. */
export const LANGUAGE_MAP: ReadonlyArray<LanguageEntry> = [
  {
    languageId: "typescript",
    extensions: LANGUAGE_EXTENSIONS.typescript,
    defaultCommand: "typescript-language-server",
    defaultArgs: ["--stdio"],
    fallbackCheck: {
      command: "npx",
      args: ["tsc", "--noEmit", "--pretty", "false"],
      diagnosticPattern: TSC_PATTERN,
      timeout: 8_000,
      retryCommand: "tsc",
      retryArgs: ["--noEmit", "--pretty", "false"],
    },
  },
  {
    languageId: "javascript",
    extensions: LANGUAGE_EXTENSIONS.javascript,
    defaultCommand: "typescript-language-server",
    defaultArgs: ["--stdio"],
    // JS projects often lack tsconfig; tsc fallback only works with one
  },
  {
    languageId: "python",
    extensions: LANGUAGE_EXTENSIONS.python,
    defaultCommand: "pyright-langserver",
    defaultArgs: ["--stdio"],
    fallbackCheck: {
      command: "pyright",
      args: [],  // file path appended at runtime
      diagnosticPattern: PYRIGHT_PATTERN,
      timeout: 15_000,
    },
  },
  {
    languageId: "c",
    extensions: LANGUAGE_EXTENSIONS.c,
    defaultCommand: "clangd",
    defaultArgs: [],
    fallbackCheck: {
      command: "gcc",
      args: ["-fsyntax-only", "-Wall"],
      diagnosticPattern: GCC_PATTERN,
      timeout: 10_000,
    },
  },
  {
    languageId: "cpp",
    extensions: LANGUAGE_EXTENSIONS.cpp,
    defaultCommand: "clangd",
    defaultArgs: [],
    fallbackCheck: {
      command: "g++",
      args: ["-fsyntax-only", "-Wall"],
      diagnosticPattern: GCC_PATTERN,
      timeout: 10_000,
    },
  },
  {
    languageId: "rust",
    extensions: LANGUAGE_EXTENSIONS.rust,
    defaultCommand: "rust-analyzer",
    defaultArgs: [],
    fallbackCheck: {
      command: "cargo",
      args: ["check", "--message-format=short"],
      diagnosticPattern: GCC_PATTERN,  // cargo --message-format=short uses same format
      timeout: 30_000,
    },
  },
  {
    languageId: "shellscript",
    extensions: LANGUAGE_EXTENSIONS.shellscript,
    defaultCommand: "bash-language-server",
    defaultArgs: ["start"],
    fallbackCheck: {
      command: "shellcheck",
      args: ["-f", "gcc"],  // gcc output format: file:line:col: severity: message
      diagnosticPattern: GCC_PATTERN,
      timeout: 5_000,
    },
  },
];

/** Detect language ID from file extension. Returns null if unknown. */
export function detectLanguage(filePath: string): string | null {
  const ext = extname(filePath).toLowerCase();
  for (const entry of LANGUAGE_MAP) {
    if (entry.extensions.includes(ext)) return entry.languageId;
  }
  return null;
}

/** Get the LanguageEntry for a language ID. */
export function getLanguageEntry(
  languageId: string,
): LanguageEntry | undefined {
  return LANGUAGE_MAP.find((e) => e.languageId === languageId);
}

// ─── LSP Router ─────────────────────────────────────────────

/**
 * Manages multiple LSP clients and routes requests by file extension.
 * Each server config can handle one or more languages.
 */
export class LSPRouter {
  /** Map from language ID to running LSPClient. */
  private clients = new Map<string, LSPClient>();
  /** Map from file extension to language ID. */
  private extensionMap = new Map<string, string>();
  private readonly rootPath: string;

  constructor(rootPath: string) {
    this.rootPath = rootPath;
  }

  /** Start an LSP server for a given server config. */
  async addServer(config: LSPServerConfig): Promise<void> {
    const primaryLanguage = config.languages[0]!;

    const client = new LSPClient({
      command: config.command,
      args: [...config.args],
      rootPath: this.rootPath,
      languageId: primaryLanguage,
      timeout: config.timeout,
      diagnosticTimeout: config.diagnosticTimeout,
    });

    await client.start();

    // Register all languages this server handles
    for (const lang of config.languages) {
      this.clients.set(lang, client);
    }

    // Register file extensions
    for (const ext of config.extensions) {
      // Find which language this extension belongs to
      const lang = this.findLanguageForExtension(ext, config) ?? primaryLanguage;
      this.extensionMap.set(ext.toLowerCase(), lang);
    }
  }

  /** Get the LSP client for a file path (by extension). */
  getClientForFile(
    filePath: string,
  ): { client: LSPClient; languageId: string } | null {
    const ext = extname(filePath).toLowerCase();
    const lang = this.extensionMap.get(ext);
    if (!lang) return null;
    const client = this.clients.get(lang);
    if (!client) return null;
    return { client, languageId: lang };
  }

  /** Get all unique running clients. */
  getClients(): ReadonlyArray<LSPClient> {
    return [...new Set(this.clients.values())];
  }

  /** Get all registered language IDs. */
  getLanguages(): ReadonlyArray<string> {
    return [...this.clients.keys()];
  }

  /** Stop all LSP servers. */
  async stopAll(): Promise<void> {
    const uniqueClients = new Set(this.clients.values());
    for (const client of uniqueClients) {
      try {
        await client.stop();
      } catch {
        /* already dead */
      }
    }
    this.clients.clear();
    this.extensionMap.clear();
  }

  private findLanguageForExtension(
    ext: string,
    config: LSPServerConfig,
  ): string | undefined {
    for (const lang of config.languages) {
      const entry = getLanguageEntry(lang);
      if (entry?.extensions.includes(ext.toLowerCase())) return lang;
    }
    return undefined;
  }
}

// ─── LSP Diagnostic Adapters ────────────────────────────────

/**
 * Adapt a single LSPClient.getDiagnostics() to the DiagnosticProvider signature.
 * Kept for backward compatibility with single-server setups.
 */
export function createLSPDiagnosticProvider(
  client: LSPClient,
): DiagnosticProvider {
  return async (filePath: string) => {
    const result = await client.getDiagnostics(filePath);
    return result.diagnostics.map((d) => ({
      message: `${filePath}:${d.line}:${d.character}: ${d.message}`,
      severity: d.severity,
    }));
  };
}

/**
 * Create a DiagnosticProvider that routes to the correct LSP server
 * based on file extension. Files with no matching server are skipped.
 */
export function createRoutingDiagnosticProvider(
  router: LSPRouter,
  onLSPDiagnostics?: () => void,
): DiagnosticProvider {
  return async (filePath: string) => {
    const match = router.getClientForFile(filePath);
    if (!match) return []; // No LSP server for this file type

    onLSPDiagnostics?.();

    // Auto-restart crashed servers when runtime client supports health checks.
    // Some unit tests stub only getDiagnostics() without lifecycle methods.
    const lifecycle = match.client as unknown as {
      isRunning?: () => boolean;
      start?: () => Promise<void>;
    };
    if (typeof lifecycle.isRunning === "function" && !lifecycle.isRunning()) {
      if (typeof lifecycle.start !== "function") {
        throw new Error(
          `LSP client for ${filePath} is not running and restart is unavailable`,
        );
      }
      try {
        await lifecycle.start();
      } catch (err) {
        const message = extractErrorMessage(err);
        throw new Error(`Failed to restart LSP client for ${filePath}: ${message}`);
      }
    }

    const result = await match.client.getDiagnostics(filePath, match.languageId);
    return result.diagnostics.map((d) => ({
      message: `${filePath}:${d.line}:${d.character}: ${d.message}`,
      severity: d.severity,
    }));
  };
}

// ─── Compiler Fallback Provider ──────────────────────────────

const COMPILER_MAX_OUTPUT = 50_000;
const PER_FILE_COMPILERS = new Set(["gcc", "g++", "pyright", "shellcheck"]);

interface ParsedDiagnostic {
  readonly message: string;
  readonly severity: string;
}

interface CompilerRunOptions {
  readonly repoRoot: string;
  readonly command: string;
  readonly args: ReadonlyArray<string>;
  readonly diagnosticPattern: RegExp;
  readonly filePath: string;
  readonly timeout: number;
}

/**
 * Create a DiagnosticProvider that runs language-specific compiler/checker
 * commands as a fallback when no LSP server is available.
 *
 * Supports: tsc, pyright, cargo check, gcc/g++, shellcheck.
 * Each language entry in LANGUAGE_MAP may define a fallbackCheck with
 * a command, args, output pattern, and timeout.
 */
export function createCompilerFallbackProvider(
  repoRoot: string,
): DiagnosticProvider {
  return async (filePath: string) => {
    const lang = detectLanguage(filePath);
    if (!lang) return [];

    const entry = getLanguageEntry(lang);
    if (!entry?.fallbackCheck) return [];

    const { command, args, diagnosticPattern, timeout, retryCommand, retryArgs } = entry.fallbackCheck;

    // Try primary command
    const result = await tryCompiler({ repoRoot, command, args, diagnosticPattern, filePath, timeout });
    if (result !== null) return result;

    // Primary failed (compiler not available) — try retry command if configured
    if (retryCommand) {
      const retryResult = await tryCompiler(
        { repoRoot, command: retryCommand, args: retryArgs ?? args, diagnosticPattern, filePath, timeout },
      );
      if (retryResult !== null) return retryResult;
    }

    return []; // No compiler available
  };
}

function parseDiagnostics(
  output: string,
  diagnosticPattern: RegExp,
  command: string,
  filePath: string,
): ParsedDiagnostic[] {
  const diagnostics: ParsedDiagnostic[] = [];
  const seen = new Set<string>();

  for (const line of output.split("\n")) {
    const diagnostic = parseDiagnosticLine(line, diagnosticPattern, command, filePath);
    if (!diagnostic || seen.has(diagnostic.message)) continue;
    seen.add(diagnostic.message);
    diagnostics.push(diagnostic);
  }
  return diagnostics;
}

function parseDiagnosticLine(
  line: string,
  diagnosticPattern: RegExp,
  command: string,
  filePath: string,
): ParsedDiagnostic | null {
  const match = diagnosticPattern.exec(line.trim());
  if (!match?.groups) return null;

  const { file, line: lineNum, col, message, sev } = match.groups;
  if (!file || !lineNum || !isRelevantCompilerFile(command, file, filePath)) {
    return null;
  }

  return {
    message: `${file}:${lineNum}${col ? `:${col}` : ""}: ${message ?? line}`,
    severity: sev?.toLowerCase() === "warning" ? "warning" : "error",
  };
}

function isRelevantCompilerFile(command: string, file: string, filePath: string): boolean {
  if (PER_FILE_COMPILERS.has(command)) {
    return true;
  }
  const normFile = file.replace(/\\/g, "/");
  const normTarget = filePath.replace(/\\/g, "/");
  return normTarget.endsWith(normFile) || normFile.endsWith(normTarget.split("/").pop()!);
}

async function tryCompiler(
  options: CompilerRunOptions,
): Promise<ParsedDiagnostic[] | null> {
  const finalArgs = PER_FILE_COMPILERS.has(options.command)
    ? [...options.args, options.filePath]
    : [...options.args];

  try {
    const result = await spawnAndCapture(options.command, finalArgs, {
      cwd: options.repoRoot,
      timeout: options.timeout,
      maxBytes: COMPILER_MAX_OUTPUT,
    });
    if (result.exitCode === 0) return [];
    return parseDiagnostics(
      result.stderr || result.stdout,
      options.diagnosticPattern,
      options.command,
      options.filePath,
    );
  } catch {
    return null;
  }
}

// ─── Shell Test Runner ─────────────────────────────────────

const TEST_TIMEOUT_MS = 60_000;
const TEST_MAX_OUTPUT_BYTES = 50_000;

/**
 * Run a single shell command and return success/output.
 */
async function runShellCommand(
  command: string,
  cwd: string,
): Promise<{ success: boolean; output: string }> {
  const result = await spawnAndCapture("sh", ["-c", command], {
    cwd,
    timeout: TEST_TIMEOUT_MS,
    maxBytes: TEST_MAX_OUTPUT_BYTES,
  });

  if (result.timedOut) {
    return {
      success: false,
      output: `Test command timed out after ${TEST_TIMEOUT_MS}ms`,
    };
  }

  const combined = result.stderr
    ? `${result.stdout}\n--- stderr ---\n${result.stderr}`
    : result.stdout;

  return {
    success: result.exitCode === 0,
    output: combined || "(no output)",
  };
}

/**
 * Create a TestRunner that executes a shell command and returns success/output.
 * Used by DoubleCheck to run the project's test suite after file mutations.
 *
 * When the primary command fails with no meaningful output (empty stdout+stderr),
 * tries fallback commands sequentially — this handles environments where
 * the primary test runner is broken (e.g. corepack/yarn issues).
 */
export function createShellTestRunner(
  repoRoot: string,
  fallbacks?: ReadonlyArray<string>,
): TestRunner {
  return async (command: string) => {
    const result = await runShellCommand(command, repoRoot);

    // If primary succeeded or produced meaningful output, return as-is
    if (result.success || result.output.trim().length > 20) {
      return result;
    }

    // Primary failed with no/minimal output — try fallbacks
    if (fallbacks && fallbacks.length > 0) {
      for (const fallback of fallbacks) {
        const fbResult = await runShellCommand(fallback, repoRoot);
        if (fbResult.success || fbResult.output.trim().length > 20) {
          return fbResult;
        }
      }
    }

    return result;
  };
}

// ─── Lazy LSP Detection ─────────────────────────────────────

/**
 * Check if a binary exists in PATH.
 * Returns true if `which <command>` succeeds, false otherwise.
 */
function commandExists(command: string): Promise<boolean> {
  return new Promise((resolve) => {
    execFile("which", [command], (err) => {
      resolve(!err);
    });
  });
}

interface DetectedLSPServer {
  readonly command: string;
  readonly args: readonly string[];
  readonly languages: readonly string[];
  readonly extensions: readonly string[];
}

/**
 * Scan PATH for known LSP server binaries.
 * Returns configs for servers that are actually installed.
 */
export async function detectAvailableLSPServers(): Promise<ReadonlyArray<DetectedLSPServer>> {
  // Group languages by their default LSP command to avoid duplicate starts
  interface MutableGroup {
    command: string;
    args: string[];
    languages: string[];
    extensions: string[];
  }
  const serverGroups = new Map<string, MutableGroup>();

  for (const entry of LANGUAGE_MAP) {
    const key = entry.defaultCommand;
    const existing = serverGroups.get(key);
    if (existing) {
      existing.languages.push(entry.languageId);
      existing.extensions.push(...entry.extensions);
    } else {
      serverGroups.set(key, {
        command: entry.defaultCommand,
        args: [...entry.defaultArgs],
        languages: [entry.languageId],
        extensions: [...entry.extensions],
      });
    }
  }

  // Check which server binaries exist in PATH (parallel)
  const checks = [...serverGroups.entries()].map(async ([cmd, group]) => {
    const exists = await commandExists(cmd);
    return exists ? group : null;
  });

  const results = await Promise.all(checks);
  return results.filter((r): r is MutableGroup => r !== null);
}

interface LazyLSPUpgradeOptions {
  readonly repoRoot: string;
  readonly doubleCheck: DoubleCheck;
  readonly lspRouter: LSPRouter;
  /** Called when LSP servers start (for logging). */
  readonly onServerStarted?: (server: DetectedLSPServer) => void;
  /** Called when upgrade completes. */
  readonly onUpgradeComplete?: (serverCount: number) => void;
  /** Called on upgrade failure. */
  readonly onError?: (error: Error) => void;
  /** Called when LSP diagnostics run in DoubleCheck via routing provider. */
  readonly onLSPDiagnostics?: () => void;
}

/**
 * Background lazy LSP upgrade: detect available LSP servers in PATH,
 * start them, and swap the diagnostic provider from compiler fallback to LSP.
 *
 * Call this after wiring the compiler fallback — it runs asynchronously and
 * upgrades the diagnostic provider in-place when servers are ready.
 *
 * Returns a promise that resolves when the upgrade attempt is complete.
 */
export async function lazyUpgradeLSP(
  opts: LazyLSPUpgradeOptions,
): Promise<void> {
  const { doubleCheck, lspRouter, onServerStarted, onUpgradeComplete, onError, onLSPDiagnostics } = opts;

  try {
    const available = await detectAvailableLSPServers();
    if (available.length === 0) {
      onUpgradeComplete?.(0);
      return;
    }

    let startedCount = 0;

    for (const server of available) {
      if (await startDetectedLspServer(lspRouter, server)) {
        startedCount++;
        onServerStarted?.(server);
      }
    }

    upgradeDiagnosticProvider(startedCount, doubleCheck, lspRouter, onLSPDiagnostics);

    onUpgradeComplete?.(startedCount);
  } catch (err) {
    onError?.(err instanceof Error ? err : new Error(String(err)));
  }
}

function upgradeDiagnosticProvider(
  startedCount: number,
  doubleCheck: DoubleCheck,
  lspRouter: LSPRouter,
  onLSPDiagnostics: (() => void) | undefined,
): void {
  if (startedCount === 0) {
    return;
  }
  doubleCheck.setDiagnosticProvider(createRoutingDiagnosticProvider(lspRouter, onLSPDiagnostics));
}

async function startDetectedLspServer(
  lspRouter: LSPRouter,
  server: DetectedLSPServer,
): Promise<boolean> {
  try {
    await lspRouter.addServer({
      command: server.command,
      args: [...server.args],
      languages: [...server.languages],
      extensions: [...server.extensions],
      timeout: 10_000,
    });
    return true;
  } catch {
    return false;
  }
}
