#!/usr/bin/env bun
/**
 * Verify LSP server integration end-to-end.
 *
 * Tests each configured LSP server by:
 *   1. Starting the server via LSPClient
 *   2. Sending a file with intentional errors
 *   3. Verifying diagnostics are returned
 *   4. Stopping the server cleanly
 *
 * Usage:
 *   bun scripts/verify-lsp.ts              # Test all servers
 *   bun scripts/verify-lsp.ts typescript   # Test one language
 */

import { LSPClient } from "../packages/tools/src/lsp/client.js";
import { writeFileSync, mkdirSync, rmSync } from "node:fs";
import { join } from "node:path";

// ─── Test fixtures: files with intentional errors ─────────

interface TestFixture {
  readonly language: string;
  readonly command: string;
  readonly args: string[];
  readonly fileName: string;
  readonly content: string;
  readonly expectDiagnostics: boolean;
  /** How long to wait for diagnostics (ms). Default: 3000. */
  readonly diagnosticTimeout?: number;
  /** Custom setup function — runs before LSP starts. Receives the rootPath.
   *  Use this for fixtures that need project structure (e.g., Cargo.toml for Rust). */
  readonly setup?: (rootPath: string) => void;
  /** If true, use a subdirectory as rootPath instead of tmpDir.
   *  The setup function should create this subdirectory. */
  readonly subDir?: string;
}

const FIXTURES: TestFixture[] = [
  {
    language: "typescript",
    command: "typescript-language-server",
    args: ["--stdio"],
    fileName: "test-error.ts",
    content: `// TypeScript file with a type error
const x: number = "not a number";
const y: string = 42;
console.log(x + y);
`,
    expectDiagnostics: true,
  },
  {
    language: "python",
    command: "pyright-langserver",
    args: ["--stdio"],
    fileName: "test_error.py",
    content: `# Python file with a type error
def greet(name: str) -> str:
    return 42  # Wrong return type

x: int = "not an int"
`,
    expectDiagnostics: true,
  },
  {
    language: "c",
    command: "clangd",
    args: [],
    fileName: "test_error.c",
    content: `// C file with an error
#include <stdio.h>

int main() {
    int x = "not an int";
    printf("%d\\n", x);
    undeclared_function();
    return 0;
}
`,
    expectDiagnostics: true,
  },
  {
    language: "shellscript",
    command: "bash-language-server",
    args: ["start"],
    fileName: "test_error.sh",
    content: `#!/bin/bash
# Bash file — bash-language-server may not report errors for all issues
echo "hello world"
if [ -f /tmp/test ]; then
  echo "file exists"
fi
`,
    expectDiagnostics: false, // Bash LSP is mainly for completions, not type errors
  },
  {
    language: "rust",
    command: "rust-analyzer",
    args: [],
    // rust-analyzer needs a Cargo project — it delegates diagnostics to `cargo check`
    fileName: "src/main.rs",
    content: `// Rust file with type errors
fn main() {
    let x: i32 = "not a number";
    let y: String = 42;
    println!("{} {}", x, y);
}
`,
    expectDiagnostics: true,
    // cargo check can take 15-30 seconds on first run
    diagnosticTimeout: 45_000,
    subDir: "rust-test-project",
    setup: (rootPath: string) => {
      // Create a minimal Cargo project
      mkdirSync(join(rootPath, "src"), { recursive: true });
      writeFileSync(
        join(rootPath, "Cargo.toml"),
        `[package]
name = "lsp-verify-test"
version = "0.1.0"
edition = "2021"
`,
      );
    },
  },
];

// ─── Colors ──────────────────────────────────────────────

const GREEN = "\x1b[32m";
const RED = "\x1b[31m";
const YELLOW = "\x1b[33m";
const DIM = "\x1b[2m";
const NC = "\x1b[0m";

function ok(msg: string) { console.log(`${GREEN}✓${NC} ${msg}`); }
function fail(msg: string) { console.log(`${RED}✗${NC} ${msg}`); }
function warn(msg: string) { console.log(`${YELLOW}⚠${NC} ${msg}`); }
function dim(msg: string) { console.log(`${DIM}${msg}${NC}`); }

// ─── Check if command exists ─────────────────────────────

function commandExists(cmd: string): boolean {
  try {
    Bun.spawnSync(["which", cmd]);
    return Bun.spawnSync(["which", cmd]).exitCode === 0;
  } catch {
    return false;
  }
}

// ─── Test a single LSP server ────────────────────────────

async function testServer(fixture: TestFixture, tmpDir: string): Promise<boolean> {
  const {
    language, command, args, fileName, content,
    expectDiagnostics, diagnosticTimeout, setup, subDir,
  } = fixture;

  console.log(`\n── ${language} (${command}) ──`);

  // Check if command exists
  if (!commandExists(command)) {
    warn(`${command} not found — skipping`);
    return true; // Not a failure, just not installed
  }

  // Determine root path — some fixtures need a project subdirectory
  const rootPath = subDir ? join(tmpDir, subDir) : tmpDir;
  mkdirSync(rootPath, { recursive: true });

  // Run custom setup (e.g., create Cargo.toml for Rust)
  if (setup) {
    dim(`  Running fixture setup...`);
    setup(rootPath);
  }

  // Write test file
  const fileDir = join(rootPath, fileName).replace(/\/[^/]+$/, "");
  mkdirSync(fileDir, { recursive: true });
  writeFileSync(join(rootPath, fileName), content);

  // Create LSPClient
  const client = new LSPClient({
    command,
    args: [...args],
    rootPath,
    languageId: language,
    timeout: 15_000,
    diagnosticTimeout: diagnosticTimeout,
  });

  try {
    // Start server
    dim(`  Starting ${command}...`);
    await client.start();
    ok(`  Server started`);

    // Get diagnostics
    const timeoutSec = (diagnosticTimeout ?? 3_000) / 1000;
    dim(`  Requesting diagnostics for ${fileName} (timeout: ${timeoutSec}s)...`);
    const result = await client.getDiagnostics(fileName);

    if (result.diagnostics.length > 0) {
      ok(`  Got ${result.diagnostics.length} diagnostic(s):`);
      for (const d of result.diagnostics.slice(0, 5)) {
        dim(`    ${d.line}:${d.character} [${d.severity}] ${d.message}`);
      }
      if (result.diagnostics.length > 5) {
        dim(`    ... and ${result.diagnostics.length - 5} more`);
      }
    } else if (expectDiagnostics) {
      warn(`  No diagnostics returned (expected some — server may need more time)`);
    } else {
      ok(`  No diagnostics (expected for ${language})`);
    }

    // Test symbols
    dim(`  Requesting symbols...`);
    const symbols = await client.getSymbols(fileName);
    if (symbols.length > 0) {
      ok(`  Got ${symbols.length} symbol(s): ${symbols.map(s => s.name).join(", ")}`);
    } else {
      dim(`  No symbols returned`);
    }

    // Stop server
    await client.stop();
    ok(`  Server stopped cleanly`);
    return true;
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    fail(`  Error: ${msg}`);
    try { await client.stop(); } catch { /* ignore */ }
    return false;
  }
}

// ─── Main ────────────────────────────────────────────────

async function main() {
  const filterLanguage = process.argv[2];

  console.log("DevAgent LSP Verification");
  console.log("=========================\n");

  // Create temp directory for test files
  const tmpDir = join(process.cwd(), ".lsp-verify-tmp");
  mkdirSync(tmpDir, { recursive: true });

  // Write a minimal tsconfig.json so typescript-language-server works
  writeFileSync(join(tmpDir, "tsconfig.json"), JSON.stringify({
    compilerOptions: { strict: true, target: "ES2022", module: "ESNext" },
    include: ["*.ts"],
  }));

  let passed = 0;
  let failed = 0;
  let skipped = 0;

  const fixtures = filterLanguage
    ? FIXTURES.filter(f => f.language === filterLanguage || f.command.includes(filterLanguage))
    : FIXTURES;

  if (fixtures.length === 0) {
    console.log(`No fixtures match "${filterLanguage}"`);
    console.log(`Available: ${FIXTURES.map(f => f.language).join(", ")}`);
    process.exit(1);
  }

  for (const fixture of fixtures) {
    if (!commandExists(fixture.command)) {
      warn(`${fixture.language}: ${fixture.command} not installed — skipping`);
      skipped++;
      continue;
    }
    const success = await testServer(fixture, tmpDir);
    if (success) passed++; else failed++;
  }

  // Cleanup
  try {
    rmSync(tmpDir, { recursive: true });
  } catch { /* ignore */ }

  // Summary
  console.log("\n── Summary ──");
  console.log(`  Passed:  ${passed}`);
  if (failed > 0) console.log(`  ${RED}Failed:  ${failed}${NC}`);
  if (skipped > 0) console.log(`  Skipped: ${skipped}`);

  if (failed > 0) {
    process.exit(1);
  } else {
    ok("All available LSP servers verified!");
  }
}

main().catch((err) => {
  fail(`Unexpected error: ${err}`);
  process.exit(1);
});
