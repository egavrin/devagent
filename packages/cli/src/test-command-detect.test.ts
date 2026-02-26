import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { describe, expect, it } from "vitest";
import { detectProjectTestCommand } from "./test-command-detect.js";

describe("detectProjectTestCommand", () => {
  it("prefers targeted package.json scripts", () => {
    const repo = mkdtempSync(join(tmpdir(), "devagent-test-detect-"));
    try {
      writeFileSync(
        join(repo, "package.json"),
        JSON.stringify(
          {
            packageManager: "pnpm@9.0.0",
            scripts: {
              test: "vitest run",
              "test:unit": "vitest run src/unit",
            },
          },
          null,
          2,
        ),
        "utf-8",
      );

      const cmd = detectProjectTestCommand(repo);
      expect(cmd).toBe("pnpm test:unit");
    } finally {
      rmSync(repo, { recursive: true, force: true });
    }
  });

  it("detects pytest projects when no package.json exists", () => {
    const repo = mkdtempSync(join(tmpdir(), "devagent-test-detect-"));
    try {
      writeFileSync(
        join(repo, "pyproject.toml"),
        "[tool.pytest.ini_options]\naddopts = \"-q\"\n",
        "utf-8",
      );
      mkdirSync(join(repo, "tests"), { recursive: true });
      writeFileSync(join(repo, "tests", "test_sample.py"), "def test_ok():\n    assert True\n", "utf-8");

      const cmd = detectProjectTestCommand(repo);
      expect(cmd).toBe("python -m pytest");
    } finally {
      rmSync(repo, { recursive: true, force: true });
    }
  });

  it("falls back to unittest discovery for python tests without pytest config", () => {
    const repo = mkdtempSync(join(tmpdir(), "devagent-test-detect-"));
    try {
      mkdirSync(join(repo, "tests"), { recursive: true });
      writeFileSync(join(repo, "tests", "test_math.py"), "import unittest\n", "utf-8");

      const cmd = detectProjectTestCommand(repo);
      expect(cmd).toBe("python -m unittest discover -s tests");
    } finally {
      rmSync(repo, { recursive: true, force: true });
    }
  });

  it("returns null when no test entrypoints are detected", () => {
    const repo = mkdtempSync(join(tmpdir(), "devagent-test-detect-"));
    try {
      writeFileSync(join(repo, "README.md"), "# no tests\n", "utf-8");
      const cmd = detectProjectTestCommand(repo);
      expect(cmd).toBeNull();
    } finally {
      rmSync(repo, { recursive: true, force: true });
    }
  });
});
