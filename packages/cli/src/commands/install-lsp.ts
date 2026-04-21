import { execSync } from "node:child_process";

import { commandExists, LSP_SERVERS } from "./lsp-servers.js";
import type { LspServerDefinition } from "./lsp-servers.js";
import { hasHelpFlag, writeStderr, writeStdout } from "./shared.js";

function renderInstallLspHelpText(): string {
  return `Usage:
  devagent install-lsp

Install npm-managed LSP servers that power DevAgent code intelligence.`;
}
export function runInstallLsp(args: ReadonlyArray<string> = []): void {
  if (hasHelpFlag(args)) {
    writeStdout(renderInstallLspHelpText());
    return;
  }

  const pm = typeof globalThis.Bun !== "undefined" ? "bun" : "npm";
  const groups = groupLspServers();
  writeInstalledLspServers(groups.alreadyInstalled);

  if (groups.toInstall.length === 0) {
    writeStdout("All npm-installable LSP servers are already available.");
    writeSkippedLspServers(groups.skipped);
    return;
  }

  const packages = groups.toInstall.flatMap((lsp) => lsp.npmPackages!);
  const cmd = `${pm} install -g ${packages.join(" ")}`;
  writeStdout(`Installing: ${groups.toInstall.map((l) => l.label).join(", ")}`);
  writeStdout(`$ ${cmd}\n`);
  installLspPackages(cmd);
  writeSkippedLspServers(groups.skipped);
}

function groupLspServers(): {
  readonly toInstall: LspServerDefinition[];
  readonly skipped: LspServerDefinition[];
  readonly alreadyInstalled: LspServerDefinition[];
} {
  const toInstall: LspServerDefinition[] = [];
  const skipped: LspServerDefinition[] = [];
  const alreadyInstalled: LspServerDefinition[] = [];
  for (const lsp of LSP_SERVERS) {
    getLspTargetGroup(lsp, { toInstall, skipped, alreadyInstalled }).push(lsp);
  }
  return { toInstall, skipped, alreadyInstalled };
}

function getLspTargetGroup(
  lsp: LspServerDefinition,
  groups: {
    readonly toInstall: LspServerDefinition[];
    readonly skipped: LspServerDefinition[];
    readonly alreadyInstalled: LspServerDefinition[];
  },
): LspServerDefinition[] {
  if (commandExists(lsp.command)) return groups.alreadyInstalled;
  return lsp.npmPackages ? groups.toInstall : groups.skipped;
}

function writeInstalledLspServers(alreadyInstalled: ReadonlyArray<LspServerDefinition>): void {
  if (alreadyInstalled.length === 0) return;
  writeStdout("Already installed:");
  for (const lsp of alreadyInstalled) {
    writeStdout(`  ✓ ${lsp.label}`);
  }
  writeStdout("");
}

function installLspPackages(cmd: string): void {
  try {
    execSync(cmd, { stdio: "inherit", timeout: 120000 });
    writeStdout("\n✓ LSP servers installed");
  } catch {
    writeStderr("\n✗ Installation failed. Try running manually:");
    writeStderr(`  ${cmd}`);
    process.exit(1);
  }
}

function writeSkippedLspServers(skipped: ReadonlyArray<LspServerDefinition>): void {
  if (skipped.length > 0) {
    writeStdout("\nSystem-level servers (install manually):");
    for (const lsp of skipped) {
      writeStdout(`  · ${lsp.label} — ${lsp.install}`);
    }
  }
}
