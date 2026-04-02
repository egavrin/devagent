#!/usr/bin/env bun
/**
 * bundle.ts — Build a single-file distributable CLI bundle.
 *
 * Steps:
 *   1. Run embed-assets.ts to generate embedded constants
 *   2. Bundle CLI entry point with bun build --target=node
 *   3. Generate dist/package.json for npm publish
 *   4. Copy README and LICENSE
 *
 * Output: dist/  (ready for `npm publish`)
 */

import { readFileSync, writeFileSync, copyFileSync, mkdirSync, existsSync } from "node:fs";
import { resolve, join } from "node:path";
import { execSync } from "node:child_process";
import { MINIMUM_NODE_MAJOR, renderRuntimeBootstrap } from "../packages/cli/src/runtime-version.ts";

const ROOT = resolve(import.meta.dirname, "..");
const DIST = join(ROOT, "dist");

// ── Step 0: Clean dist ──────────────────────────────────────

mkdirSync(DIST, { recursive: true });

// ── Step 1: Generate embedded constants ─────────────────────

console.log("→ Embedding assets...");
execSync("bun run scripts/embed-assets.ts", { cwd: ROOT, stdio: "inherit" });

// ── Step 2: Bundle with Bun ─────────────────────────────────

console.log("→ Bundling CLI...");

// Create a stub for react-devtools-core (Ink's optional dev dependency)
const shimDir = join(ROOT, "node_modules", "react-devtools-core");
mkdirSync(shimDir, { recursive: true });
if (!existsSync(join(shimDir, "index.js"))) {
  writeFileSync(join(shimDir, "package.json"), JSON.stringify({ name: "react-devtools-core", version: "0.0.0", main: "index.js" }));
  writeFileSync(join(shimDir, "index.js"), "module.exports = { initialize() {}, connectToDevTools() {} };");
}

const result = Bun.spawnSync([
  "bun", "build",
  "packages/cli/src/index.ts",
  "--target=node",
  "--outfile", "dist/devagent.js",
  "--minify",
  // External: native addons and optional deps that can't be inlined
  "--external", "cli-highlight",
  "--external", "better-sqlite3",
], { cwd: ROOT, stdio: ["inherit", "inherit", "inherit"] });

if (result.exitCode !== 0) {
  console.error("✗ Bundle failed");
  process.exit(1);
}

// Ensure shebang is present
const bundleContent = readFileSync(join(DIST, "devagent.js"), "utf-8");
if (!bundleContent.startsWith("#!/")) {
  writeFileSync(
    join(DIST, "devagent.js"),
    "#!/usr/bin/env node\n" + bundleContent,
  );
}

// Make executable
execSync(`chmod +x ${join(DIST, "devagent.js")}`);

writeFileSync(join(DIST, "bootstrap.js"), renderRuntimeBootstrap("./devagent.js"));
execSync(`chmod +x ${join(DIST, "bootstrap.js")}`);

// ── Step 3: Generate package.json ───────────────────────────

console.log("→ Generating package.json...");
const rootPkg = JSON.parse(readFileSync(join(ROOT, "package.json"), "utf-8"));
const version = rootPkg.version ?? "0.1.0";

const distPkg = {
  name: "@egavrin/devagent",
  version,
  description: "AI coding agent CLI",
  type: "module",
  bin: { devagent: "bootstrap.js" },
  files: ["bootstrap.js", "devagent.js", "README.md", "LICENSE"],
  engines: { node: `>=${MINIMUM_NODE_MAJOR}` },
  publishConfig: { access: "public" },
  repository: rootPkg.repository,
  bugs: rootPkg.bugs,
  homepage: rootPkg.homepage,
  license: rootPkg.license ?? "MIT",
  dependencies: {
    "cli-highlight": "^2.1.11",
    "better-sqlite3": "^11.0.0",
    "typescript-language-server": "^4.3.0",
    "typescript": "^5.7.0",
    "pyright": "^1.1.0",
    "bash-language-server": "^5.4.0",
  },
};

writeFileSync(join(DIST, "package.json"), JSON.stringify(distPkg, null, 2) + "\n");

// ── Step 4: Copy supporting files ───────────────────────────

console.log("→ Copying supporting files...");
if (existsSync(join(ROOT, "README.md"))) {
  copyFileSync(join(ROOT, "README.md"), join(DIST, "README.md"));
}
if (existsSync(join(ROOT, "LICENSE"))) {
  copyFileSync(join(ROOT, "LICENSE"), join(DIST, "LICENSE"));
}

// ── Done ────────────────────────────────────────────────────

const stats = Bun.file(join(DIST, "devagent.js"));
const sizeKB = Math.round((await stats.size) / 1024);
console.log(`\n✓ Bundle ready: dist/devagent.js (${sizeKB} KB)`);
console.log(`  Package: ${distPkg.name}@${version}`);
console.log(`  Test: node dist/bootstrap.js --help`);
console.log(`  Publish: cd dist && npm publish --access public`);
