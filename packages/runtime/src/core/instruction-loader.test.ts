import {
  writeFileSync,
  mkdirSync,
  rmSync,
  mkdtempSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { it, expect, beforeEach, afterEach } from "vitest";

import { RepositoryInstructionLoader } from "./instruction-loader.js";
let root: string;

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), "instruction-loader-test-"));
});

afterEach(() => {
  rmSync(root, { recursive: true, force: true });
});

it("returns empty array when no instruction files exist", () => {
  const loader = new RepositoryInstructionLoader(root);
  const instructions = loader.load();
  expect(instructions).toEqual([]);
});

it("loads WORKFLOW.md with highest priority (0)", () => {
  writeFileSync(join(root, "WORKFLOW.md"), "# Workflow\nDo stuff.");
  const loader = new RepositoryInstructionLoader(root);
  const instructions = loader.load();

  expect(instructions).toHaveLength(1);
  expect(instructions[0].source).toBe("WORKFLOW.md");
  expect(instructions[0].scope).toBe("workflow");
  expect(instructions[0].priority).toBe(0);
  expect(instructions[0].content).toBe("# Workflow\nDo stuff.");
});

it("loads AGENTS.md with priority 1", () => {
  writeFileSync(join(root, "AGENTS.md"), "# Agents\nBe helpful.");
  const loader = new RepositoryInstructionLoader(root);
  const instructions = loader.load();

  expect(instructions).toHaveLength(1);
  expect(instructions[0].source).toBe("AGENTS.md");
  expect(instructions[0].scope).toBe("agent");
  expect(instructions[0].priority).toBe(1);
});

it("loads .github/copilot-instructions.md with priority 2", () => {
  mkdirSync(join(root, ".github"), { recursive: true });
  writeFileSync(
    join(root, ".github", "copilot-instructions.md"),
    "# Copilot\nGuidance.",
  );
  const loader = new RepositoryInstructionLoader(root);
  const instructions = loader.load();

  expect(instructions).toHaveLength(1);
  expect(instructions[0].source).toBe("copilot-instructions.md");
  expect(instructions[0].scope).toBe("repo");
  expect(instructions[0].priority).toBe(2);
});

it("loads .github/instructions/**/*.instructions.md with priority 3", () => {
  const instrDir = join(root, ".github", "instructions", "src");
  mkdirSync(instrDir, { recursive: true });
  writeFileSync(
    join(instrDir, "components.instructions.md"),
    "Use functional components.",
  );

  const loader = new RepositoryInstructionLoader(root);
  const instructions = loader.load();

  expect(instructions).toHaveLength(1);
  expect(instructions[0].scope).toBe("path-specific");
  expect(instructions[0].priority).toBe(3);
  expect(instructions[0].source).toBe(
    ".github/instructions/src/components.instructions.md",
  );
});

it("sorts all instructions by priority", () => {
  // Create files in reverse priority order
  mkdirSync(join(root, ".github", "instructions"), { recursive: true });
  writeFileSync(
    join(root, ".github", "instructions", "global.instructions.md"),
    "Global rules.",
  );
  writeFileSync(
    join(root, ".github", "copilot-instructions.md"),
    "Copilot guidance.",
  );
  writeFileSync(join(root, "AGENTS.md"), "Agent behavior.");
  writeFileSync(join(root, "WORKFLOW.md"), "Workflow contract.");

  const loader = new RepositoryInstructionLoader(root);
  const instructions = loader.load();

  expect(instructions).toHaveLength(4);
  expect(instructions[0].priority).toBe(0);
  expect(instructions[0].scope).toBe("workflow");
  expect(instructions[1].priority).toBe(1);
  expect(instructions[1].scope).toBe("agent");
  expect(instructions[2].priority).toBe(2);
  expect(instructions[2].scope).toBe("repo");
  expect(instructions[3].priority).toBe(3);
  expect(instructions[3].scope).toBe("path-specific");
});

it("loadForPath filters path-specific instructions correctly", () => {
  // Create a path-specific instruction for src/
  const instrDir = join(root, ".github", "instructions", "src");
  mkdirSync(instrDir, { recursive: true });
  writeFileSync(
    join(instrDir, "rules.instructions.md"),
    "Source rules.",
  );

  // Create a top-level instruction (applies to all)
  writeFileSync(
    join(root, ".github", "instructions", "global.instructions.md"),
    "Global rules.",
  );

  // Create a workflow instruction (always included)
  writeFileSync(join(root, "WORKFLOW.md"), "Workflow.");

  const loader = new RepositoryInstructionLoader(root);

  // File under src/ should match both path-specific instructions
  const srcInstructions = loader.loadForPath(join(root, "src", "index.ts"));
  expect(srcInstructions).toHaveLength(3); // workflow + global + src-specific

  // File outside src/ should not match src-specific instruction
  const otherInstructions = loader.loadForPath(
    join(root, "tests", "foo.test.ts"),
  );
  expect(otherInstructions).toHaveLength(2); // workflow + global only
});

it("discovers nested instruction files", () => {
  const deepDir = join(
    root,
    ".github",
    "instructions",
    "src",
    "components",
  );
  mkdirSync(deepDir, { recursive: true });
  writeFileSync(
    join(deepDir, "button.instructions.md"),
    "Button conventions.",
  );
  writeFileSync(
    join(
      root,
      ".github",
      "instructions",
      "src",
      "utils.instructions.md",
    ),
    "Util conventions.",
  );

  const loader = new RepositoryInstructionLoader(root);
  const instructions = loader.load();

  expect(instructions).toHaveLength(2);
  expect(instructions.every((i) => i.scope === "path-specific")).toBe(true);
});

it("findNearestAgentsMd walks up to find closest AGENTS.md", () => {
  // Create AGENTS.md in root and in a subdirectory
  writeFileSync(join(root, "AGENTS.md"), "Root agents.");
  const subDir = join(root, "packages", "core");
  mkdirSync(subDir, { recursive: true });
  writeFileSync(join(subDir, "AGENTS.md"), "Core agents.");

  const loader = new RepositoryInstructionLoader(root);

  // From packages/runtime/src/core, should find the nearest AGENTS.md
  const nearest = loader.findNearestAgentsMd(subDir);
  expect(nearest).not.toBeNull();
  expect(nearest!.path).toBe(join(subDir, "AGENTS.md"));
  expect(nearest!.content).toBe("Core agents.");

  // From packages/ (no AGENTS.md there), should find root AGENTS.md
  const fromPackages = loader.findNearestAgentsMd(join(root, "packages"));
  expect(fromPackages).not.toBeNull();
  expect(fromPackages!.path).toBe(join(root, "AGENTS.md"));
});

it("findNearestAgentsMd returns null when none exist", () => {
  const loader = new RepositoryInstructionLoader(root);
  const result = loader.findNearestAgentsMd(root);
  expect(result).toBeNull();
});

it("ignores files that do not match *.instructions.md pattern", () => {
  const instrDir = join(root, ".github", "instructions");
  mkdirSync(instrDir, { recursive: true });
  writeFileSync(join(instrDir, "README.md"), "Not an instruction.");
  writeFileSync(join(instrDir, "notes.txt"), "Also not.");
  writeFileSync(
    join(instrDir, "valid.instructions.md"),
    "This is valid.",
  );

  const loader = new RepositoryInstructionLoader(root);
  const instructions = loader.load();

  expect(instructions).toHaveLength(1);
  expect(instructions[0].source).toBe(
    ".github/instructions/valid.instructions.md",
  );
});
