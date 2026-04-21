import { execSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

import { hasHelpFlag, writeStderr, writeStdout } from "./shared.js";

function renderUpdateHelpText(): string {
  return `Usage:
  devagent update

Check npm for the latest published version and upgrade the installed CLI.`;
}
export async function runUpdate(args: ReadonlyArray<string> = []): Promise<void> {
  if (hasHelpFlag(args)) {
    writeStdout(renderUpdateHelpText());
    return;
  }

  const PACKAGE = "@egavrin/devagent";

  writeStdout("Checking for updates...");

  try {
    const res = await fetch(`https://registry.npmjs.org/${PACKAGE}/latest`, {
      signal: AbortSignal.timeout(5000),
    });
    const data = (await res.json()) as { version?: string };
    const latest = data.version;

    if (!latest) {
      writeStderr("Could not determine latest version.");
      process.exit(1);
    }

    const current = getCurrentVersion();
    if (latest === current) {
      writeStdout(`Already up to date (v${current}).`);
      return;
    }

    writeStdout(`Updating: v${current} → v${latest}\n`);

    // Detect package manager
    const isBun = typeof globalThis.Bun !== "undefined";
    const cmd = isBun
      ? `bun install -g ${PACKAGE}@latest`
      : `npm install -g ${PACKAGE}@latest`;

    writeStdout(`$ ${cmd}\n`);
    execSync(cmd, { stdio: "inherit" });
    writeStdout(`\n✓ Updated to v${latest}`);
  } catch (err) {
    writeStderr(`Update failed: ${err instanceof Error ? err.message : String(err)}`);
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
