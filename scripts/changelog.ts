#!/usr/bin/env bun
/**
 * changelog.ts — Generate release notes from git tags.
 *
 * Usage:
 *   bun run scripts/changelog.ts              # Since last tag
 *   bun run scripts/changelog.ts v0.1.0       # Since specific tag
 *   bun run scripts/changelog.ts v0.1.0 v0.1.5  # Between two tags
 */

import { execSync } from "node:child_process";

const args = process.argv.slice(2);

// Get tag range
let from: string;
let to: string;

if (args.length === 2) {
  from = args[0]!;
  to = args[1]!;
} else if (args.length === 1) {
  from = args[0]!;
  to = "HEAD";
} else {
  // Find the last tag
  try {
    from = execSync("git describe --tags --abbrev=0 HEAD^", { encoding: "utf-8" }).trim();
  } catch {
    from = execSync("git rev-list --max-parents=0 HEAD", { encoding: "utf-8" }).trim().slice(0, 8);
  }
  to = "HEAD";
}

// Get current version
let version = to;
if (to === "HEAD") {
  try {
    const pkg = JSON.parse(execSync("cat package.json", { encoding: "utf-8" }));
    version = `v${pkg.version}`;
  } catch { /* ignore */ }
}

// Get commits
const log = execSync(
  `git log ${from}..${to} --pretty=format:"%s" --no-merges`,
  { encoding: "utf-8" },
).trim();

if (!log) {
  console.log("No changes since", from);
  process.exit(0);
}

const commits = log.split("\n").filter(Boolean);

// Categorize
const categories: Record<string, string[]> = {
  "Features": [],
  "Bug Fixes": [],
  "Other": [],
};

for (const msg of commits) {
  if (msg.startsWith("feat")) {
    const clean = msg.replace(/^feat\([^)]*\):\s*/, "").replace(/^feat:\s*/, "");
    categories["Features"]!.push(clean);
  } else if (msg.startsWith("fix")) {
    const clean = msg.replace(/^fix\([^)]*\):\s*/, "").replace(/^fix:\s*/, "");
    categories["Bug Fixes"]!.push(clean);
  } else {
    categories["Other"]!.push(msg);
  }
}

// Output
console.log(`# ${version}\n`);
console.log(`_${new Date().toISOString().split("T")[0]}_\n`);

for (const [category, items] of Object.entries(categories)) {
  if (items.length === 0) continue;
  console.log(`## ${category}\n`);
  for (const item of items) {
    console.log(`- ${item}`);
  }
  console.log("");
}

// Stats
const stats = execSync(
  `git diff --shortstat ${from}..${to}`,
  { encoding: "utf-8" },
).trim();
if (stats) {
  console.log(`---\n${stats}`);
}
