/**
 * Wire DoubleCheck providers — adapters that connect LSPClient diagnostics
 * and shell-based test execution to the DoubleCheck validation system.
 *
 * Supports multi-language LSP routing: multiple LSP servers can run
 * simultaneously, each handling different file types.
 */

import { spawn } from "node:child_process";
import { extname } from "node:path";
import { LSPClient } from "@devagent/tools";
import type { LSPServerConfig } from "@devagent/core";
import type { DiagnosticProvider, TestRunner } from "@devagent/engine";

// ─── Language Map ───────────────────────────────────────────

export interface LanguageEntry {
  readonly languageId: string;
  readonly extensions: ReadonlyArray<string>;
  readonly defaultCommand: string;
  readonly defaultArgs: ReadonlyArray<string>;
}

/** Built-in map of supported languages with default LSP servers. */
export const LANGUAGE_MAP: ReadonlyArray<LanguageEntry> = [
  {
    languageId: "typescript",
    extensions: [".ts", ".tsx", ".mts", ".cts"],
    defaultCommand: "typescript-language-server",
    defaultArgs: ["--stdio"],
  },
  {
    languageId: "javascript",
    extensions: [".js", ".jsx", ".mjs", ".cjs"],
    defaultCommand: "typescript-language-server",
    defaultArgs: ["--stdio"],
  },
  {
    languageId: "python",
    extensions: [".py", ".pyi"],
    defaultCommand: "pyright-langserver",
    defaultArgs: ["--stdio"],
  },
  {
    languageId: "c",
    extensions: [".c", ".h"],
    defaultCommand: "clangd",
    defaultArgs: [],
  },
  {
    languageId: "cpp",
    extensions: [".cpp", ".cxx", ".cc", ".hpp", ".hxx", ".hh"],
    defaultCommand: "clangd",
    defaultArgs: [],
  },
  {
    languageId: "rust",
    extensions: [".rs"],
    defaultCommand: "rust-analyzer",
    defaultArgs: [],
  },
  {
    languageId: "shellscript",
    extensions: [".sh", ".bash", ".zsh"],
    defaultCommand: "bash-language-server",
    defaultArgs: ["start"],
  },
  {
    languageId: "arkts",
    extensions: [".ets"],
    defaultCommand: "typescript-language-server",
    defaultArgs: ["--stdio"],
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
): DiagnosticProvider {
  return async (filePath: string) => {
    const match = router.getClientForFile(filePath);
    if (!match) return []; // No LSP server for this file type

    const result = await match.client.getDiagnostics(filePath, match.languageId);
    return result.diagnostics.map((d) => ({
      message: `${filePath}:${d.line}:${d.character}: ${d.message}`,
      severity: d.severity,
    }));
  };
}

// ─── Shell Test Runner ─────────────────────────────────────

const TEST_TIMEOUT_MS = 60_000;
const MAX_OUTPUT_BYTES = 50_000;

/**
 * Create a TestRunner that executes a shell command and returns success/output.
 * Used by DoubleCheck to run the project's test suite after file mutations.
 */
export function createShellTestRunner(repoRoot: string): TestRunner {
  return (command: string) => {
    return new Promise((resolveP) => {
      const child = spawn("sh", ["-c", command], {
        cwd: repoRoot,
        stdio: ["ignore", "pipe", "pipe"],
      });

      let stdout = "";
      let stderr = "";
      let killed = false;

      child.stdout.on("data", (data: Buffer) => {
        const chunk = data.toString();
        if (stdout.length < MAX_OUTPUT_BYTES) {
          stdout += chunk.substring(0, MAX_OUTPUT_BYTES - stdout.length);
        }
      });

      child.stderr.on("data", (data: Buffer) => {
        const chunk = data.toString();
        if (stderr.length < MAX_OUTPUT_BYTES) {
          stderr += chunk.substring(0, MAX_OUTPUT_BYTES - stderr.length);
        }
      });

      child.on("close", (code) => {
        if (killed) return;
        const combined = stderr
          ? `${stdout}\n--- stderr ---\n${stderr}`
          : stdout;
        resolveP({
          success: code === 0,
          output: combined || "(no output)",
        });
      });

      child.on("error", (err) => {
        if (killed) return;
        resolveP({
          success: false,
          output: `Failed to run test: ${err.message}`,
        });
      });

      // Timeout
      const timer = setTimeout(() => {
        killed = true;
        child.kill("SIGTERM");
        resolveP({
          success: false,
          output: `Test command timed out after ${TEST_TIMEOUT_MS}ms`,
        });
      }, TEST_TIMEOUT_MS);

      child.on("close", () => clearTimeout(timer));
    });
  };
}
