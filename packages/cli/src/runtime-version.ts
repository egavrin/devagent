export const MINIMUM_NODE_MAJOR = 20;
export const MINIMUM_BUN_VERSION = "1.3";

export interface RuntimeVersionInfo {
  readonly nodeVersion: string;
  readonly hasBun: boolean;
}

export function parseNodeMajor(version: string): number | null {
  const match = /^v?(\d+)/.exec(version.trim());
  const major = match?.[1];
  if (!major) return null;
  return Number.parseInt(major, 10);
}

export function getUnsupportedRuntimeMessage(
  runtime: RuntimeVersionInfo = {
    nodeVersion: process.version,
    hasBun: typeof Bun !== "undefined",
  },
): string | null {
  if (runtime.hasBun) return null;

  const major = parseNodeMajor(runtime.nodeVersion);
  if (major !== null && major >= MINIMUM_NODE_MAJOR) return null;

  return [
    `Unsupported Node.js runtime: ${runtime.nodeVersion}.`,
    `DevAgent requires Node.js >= ${MINIMUM_NODE_MAJOR} or Bun >= ${MINIMUM_BUN_VERSION}.`,
    'Older Node.js releases can crash before startup with "Invalid regular expression flags".',
    "Upgrade Node.js and run the command again.",
  ].join("\n");
}

export function renderRuntimeBootstrap(targetImportPath: string): string {
  return `#!/usr/bin/env node
const minimumNodeMajor = ${MINIMUM_NODE_MAJOR};
const minimumBunVersion = ${JSON.stringify(MINIMUM_BUN_VERSION)};
const runtimeVersion = process.version || "";
const match = /^v?(\\d+)/.exec(runtimeVersion.trim());
const major = match ? Number.parseInt(match[1], 10) : Number.NaN;

if (!Number.isFinite(major) || major < minimumNodeMajor) {
  process.stderr.write(
    "Unsupported Node.js runtime: " + runtimeVersion + "\\n" +
    "DevAgent requires Node.js >= " + minimumNodeMajor + " or Bun >= " + minimumBunVersion + ".\\n" +
    "Older Node.js releases can crash before startup with \\"Invalid regular expression flags\\".\\n" +
    "Upgrade Node.js and run the command again.\\n"
  );
  process.exit(1);
}

import(${JSON.stringify(targetImportPath)});
`;
}
