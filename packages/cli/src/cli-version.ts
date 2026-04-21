import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

export function getVersion(): string {
  try {
    const moduleDir = dirname(fileURLToPath(import.meta.url));
    for (const pkgPath of [
      join(moduleDir, "package.json"),
      join(moduleDir, "..", "..", "..", "package.json"),
      join(moduleDir, "..", "package.json"),
    ]) {
      if (!existsSync(pkgPath)) continue;
      const pkg = JSON.parse(readFileSync(pkgPath, "utf-8"));
      if (typeof pkg.version === "string" && pkg.version.length > 0) return pkg.version;
    }
  } catch {
    // Fall through to the hard-coded minimum fallback below.
  }
  return "0.1.0";
}

function semverNewer(a: string, b: string): boolean {
  const pa = a.split(".").map(Number);
  const pb = b.split(".").map(Number);
  for (let i = 0; i < 3; i++) {
    if ((pa[i] ?? 0) > (pb[i] ?? 0)) return true;
    if ((pa[i] ?? 0) < (pb[i] ?? 0)) return false;
  }
  return false;
}

export function checkForUpdates(): void {
  if ((process.env["DEVAGENT_DISABLE_UPDATE_CHECK"] ?? "") === "1") return;
  const packageName = "@egavrin/devagent";
  const cacheDir = join(homedir(), ".cache", "devagent");
  const cachePath = join(cacheDir, "update-check.json");
  const oneDayMs = 24 * 60 * 60 * 1000;

  try {
    if (existsSync(cachePath)) {
      const cached = JSON.parse(readFileSync(cachePath, "utf-8"));
      if (Date.now() - cached.checkedAt < oneDayMs) {
        maybeWriteUpdateNotice(packageName, cached.latest);
        return;
      }
    }
  } catch {
    // Ignore cache read errors.
  }

  fetch(`https://registry.npmjs.org/${packageName}/latest`, {
    signal: AbortSignal.timeout(3000),
  })
    .then((res) => res.json())
    .then((data: any) => {
      const latest = data?.version as string | undefined;
      if (!latest) return;
      try {
        mkdirSync(cacheDir, { recursive: true });
        writeFileSync(cachePath, JSON.stringify({ latest, checkedAt: Date.now() }));
      } catch {
        // Ignore cache write errors.
      }
      maybeWriteUpdateNotice(packageName, latest);
    })
    .catch(() => {
      // Network errors never block startup.
    });
}

function maybeWriteUpdateNotice(packageName: string, latest: string | undefined): void {
  const current = getVersion();
  if (latest && semverNewer(latest, current)) {
    process.stderr.write(`\x1b[33mUpdate available: ${current} → ${latest}\x1b[0m — npm i -g ${packageName}\n`);
  }
}
