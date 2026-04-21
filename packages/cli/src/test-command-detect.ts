import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

const TEST_SCRIPT_CANDIDATES = ["test:implementation", "test:unit", "test"] as const;
const PYTHON_TEST_FILENAME = /^test_.*\.py$|.*_test\.py$/;
const IGNORED_DIRS = new Set([".git", "node_modules", ".venv", "venv", "__pycache__"]);

export function detectProjectTestCommand(repoRoot: string): string | null {
  const jsCommand = detectJsTestCommand(repoRoot);
  if (jsCommand) return jsCommand;

  if (!hasPythonTests(repoRoot)) return null;

  if (hasPytestConfig(repoRoot)) {
    return "python -m pytest";
  }

  return existsSync(join(repoRoot, "tests"))
    ? "python -m unittest discover -s tests"
    : "python -m unittest discover";
}

function detectJsTestCommand(repoRoot: string): string | null {
  const pkgPath = join(repoRoot, "package.json");
  if (!existsSync(pkgPath)) return null;

  try {
    const pkg = JSON.parse(readFileSync(pkgPath, "utf-8")) as {
      scripts?: Record<string, string>;
      packageManager?: string;
    };
    const scripts = pkg.scripts;
    if (!scripts) return null;

    const prefix = getPackageManagerPrefix(pkg.packageManager);
    for (const name of TEST_SCRIPT_CANDIDATES) {
      if (scripts[name]) {
        return `${prefix} ${name}`;
      }
    }
    return null;
  } catch {
    return null;
  }
}

function getPackageManagerPrefix(packageManager: string | undefined): string {
  if (packageManager?.startsWith("yarn")) return "corepack yarn";
  if (packageManager?.startsWith("pnpm")) return "pnpm";
  if (packageManager?.startsWith("bun")) return "bun run";
  return "npm run";
}

function hasPytestConfig(repoRoot: string): boolean {
  const pytestIni = join(repoRoot, "pytest.ini");
  if (existsSync(pytestIni)) return true;

  const pyproject = join(repoRoot, "pyproject.toml");
  if (fileContains(pyproject, "[tool.pytest")) return true;

  const toxIni = join(repoRoot, "tox.ini");
  if (fileContains(toxIni, "[pytest]")) return true;

  const setupCfg = join(repoRoot, "setup.cfg");
  if (fileContains(setupCfg, "[tool:pytest]")) return true;

  for (const req of ["requirements.txt", "requirements-dev.txt"]) {
    if (fileContains(join(repoRoot, req), "pytest")) return true;
  }

  return false;
}

function hasPythonTests(repoRoot: string): boolean {
  const testsDir = join(repoRoot, "tests");
  if (existsSync(testsDir) && directoryContainsPythonFile(testsDir, 5, true)) {
    return true;
  }

  return directoryContainsPythonFile(repoRoot, 4, false);
}
function directoryContainsPythonFile(
  dir: string,
  depth: number,
  anyPythonFile: boolean,
): boolean {
  if (depth < 0) return false;
  return readDirectoryEntries(dir).some((entry) => {
    if (entry.isDirectory()) {
      return !IGNORED_DIRS.has(entry.name) &&
        directoryContainsPythonFile(join(dir, entry.name), depth - 1, anyPythonFile);
    }
    return isMatchingPythonTestFile(entry, anyPythonFile);
  });
}

type DirectoryEntry = {
  name: string;
  isDirectory(): boolean;
  isFile(): boolean;
};

function readDirectoryEntries(dir: string): DirectoryEntry[] {
  try {
    return readdirSync(dir, { withFileTypes: true, encoding: "utf8" }) as DirectoryEntry[];
  } catch {
    return [];
  }
}

function isMatchingPythonTestFile(entry: DirectoryEntry, anyPythonFile: boolean): boolean {
  return entry.isFile() &&
    entry.name.endsWith(".py") &&
    (anyPythonFile || PYTHON_TEST_FILENAME.test(entry.name));
}

function fileContains(path: string, snippet: string): boolean {
  if (!existsSync(path)) return false;
  try {
    return readFileSync(path, "utf-8").includes(snippet);
  } catch {
    return false;
  }
}
