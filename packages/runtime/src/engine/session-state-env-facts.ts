export interface EnvFact {
  readonly key: string;
  readonly message: string;
}

type EnvFactExtractor = (combined: string) => EnvFact | null;

const ENV_FACT_EXTRACTORS: ReadonlyArray<EnvFactExtractor> = [
  extractCommandNotFoundFact,
  extractPermissionDeniedFact,
  extractMissingDependencyFact,
  extractCargoFailureFact,
  extractNetworkFailureFact,
  extractDiskFullFact,
  extractVersionMismatchFact,
  extractGitIssueFact,
  extractTimeoutFact,
];

export function extractEnvFact(
  toolName: string,
  error: string,
  output: string,
): EnvFact | null {
  if (toolName !== "run_command") return null;

  const combined = `${error}\n${output}`;
  for (const extractor of ENV_FACT_EXTRACTORS) {
    const fact = extractor(combined);
    if (fact) return fact;
  }
  return null;
}

function extractCommandNotFoundFact(combined: string): EnvFact | null {
  const cmdNotFound = combined.match(/(?:command not found|not found):\s*(\S+)/i)
    ?? combined.match(/(\S+):\s*(?:command not found|No such file)/i);
  if (!cmdNotFound?.[1]) return null;
  const cmd = cmdNotFound[1];
  return {
    key: `cmd-not-found:${cmd}`,
    message: `${cmd} is not installed on this system. Use an alternative command.`,
  };
}

function extractPermissionDeniedFact(combined: string): EnvFact | null {
  const permDenied = combined.match(/(\S+):\s*[Pp]ermission denied/);
  if (!permDenied?.[1]) return null;
  return {
    key: `permission-denied:${permDenied[1]}`,
    message: `Permission denied for ${permDenied[1]}. Check file permissions or use sudo.`,
  };
}

function extractMissingDependencyFact(combined: string): EnvFact | null {
  const missingDep = combined.match(/can't find crate for `([^`]+)`/)
    ?? combined.match(/ModuleNotFoundError:\s*No module named '([^']+)'/)
    ?? combined.match(/Cannot find module '([^']+)'/);
  if (!missingDep?.[1]) return null;
  return {
    key: `build-fail:missing-${missingDep[1]}`,
    message: `Build fails — missing dependency: ${missingDep[1]}. Install it or skip build verification.`,
  };
}

function extractCargoFailureFact(combined: string): EnvFact | null {
  if (!hasExitCode101(combined) || !hasRustBuildOutput(combined)) return null;
  return {
    key: "build-fail:cargo",
    message: "cargo check/build fails in this environment. Skip cargo verification or fix native dependencies first.",
  };
}

function hasExitCode101(combined: string): boolean {
  return combined.includes("exit code: 101") || combined.includes("Exit code: 101");
}

function hasRustBuildOutput(combined: string): boolean {
  return combined.includes("cargo") || combined.includes("rustc") || combined.includes("error[E");
}

function extractNetworkFailureFact(combined: string): EnvFact | null {
  if (!/Could not resolve host|Connection refused|ETIMEDOUT|ECONNREFUSED/i.test(combined)) {
    return null;
  }
  return {
    key: "network-failure",
    message: "Network access appears unavailable or restricted. Use offline alternatives.",
  };
}

function extractDiskFullFact(combined: string): EnvFact | null {
  if (!/No space left on device|Disk quota exceeded/i.test(combined)) return null;
  return {
    key: "disk-full",
    message: "Disk is full or quota exceeded. Free space before writing files.",
  };
}

function extractVersionMismatchFact(combined: string): EnvFact | null {
  const versionMismatch = combined.match(/requires Node\.js (\d+)/)
    ?? combined.match(/requires Python (\d+\.\d+)/);
  if (!versionMismatch) return null;
  return {
    key: "version-mismatch",
    message: `Runtime version mismatch detected: ${versionMismatch[0]}. Use compatible syntax or check version.`,
  };
}

function extractGitIssueFact(combined: string): EnvFact | null {
  const gitIssue = combined.match(/fatal: not a git repository/)
    ?? combined.match(/CONFLICT.*Merge conflict/i);
  if (!gitIssue) return null;
  return {
    key: "git-issue",
    message: `Git state issue: ${gitIssue[0]}. Resolve before proceeding with file operations.`,
  };
}

function extractTimeoutFact(combined: string): EnvFact | null {
  if (!combined.includes("timed out") && !combined.includes("SIGTERM")) return null;
  if (isSearchTimeout(combined)) {
    return {
      key: "search-timeout",
      message: "Recursive search command timed out on this repo. Use scoped searches: specific subdirectories, --include/--glob filters, -maxdepth for find, or builtin search_files/find_files tools.",
    };
  }
  return {
    key: "tool-timeout",
    message: "Command timed out. Use shorter-running commands or increase timeout.",
  };
}

function isSearchTimeout(combined: string): boolean {
  return /\bgrep\s+-[rRl]*R|\bfind\b|\brg\b/.test(combined);
}
