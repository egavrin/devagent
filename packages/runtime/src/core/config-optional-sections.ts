import { LANGUAGE_EXTENSIONS } from "./languages.js";
import type {
  DoubleCheckConfig,
  LoggingConfig,
  LSPConfig,
  SessionStateConfigCore,
} from "./types.js";

export function parseDoubleCheckConfig(
  rawDoubleCheck: Record<string, unknown> | undefined,
): DoubleCheckConfig | undefined {
  if (!rawDoubleCheck) return undefined;
  return {
    enabled: (rawDoubleCheck["enabled"] as boolean) ?? false,
    checkDiagnostics: rawDoubleCheck["check_diagnostics"] as boolean | undefined,
    runTests: rawDoubleCheck["run_tests"] as boolean | undefined,
    testCommand: rawDoubleCheck["test_command"] as string | null | undefined,
    diagnosticTimeout: rawDoubleCheck["diagnostic_timeout"] as number | undefined,
  };
}

export function parseLspConfig(rawLsp: Record<string, unknown> | undefined): LSPConfig | undefined {
  if (!rawLsp) return undefined;
  if (rawLsp["servers"]) return parseMultiServerLspConfig(rawLsp["servers"]);
  if (rawLsp["command"]) return parseLegacyLspConfig(rawLsp);
  return undefined;
}

export function parseLoggingConfig(
  rawLogging: Record<string, unknown> | undefined,
): LoggingConfig | undefined {
  if (!rawLogging) return undefined;
  return {
    enabled: (rawLogging["enabled"] as boolean) ?? true,
    logDir: rawLogging["log_dir"] as string | undefined,
    retentionDays: rawLogging["retention_days"] as number | undefined,
  };
}

export function parseSessionStateConfig(
  rawSessionState: Record<string, unknown> | undefined,
): SessionStateConfigCore | undefined {
  if (!rawSessionState) return undefined;
  return {
    persist: rawSessionState["persist"] as boolean | undefined,
    trackPlan: rawSessionState["track_plan"] as boolean | undefined,
    trackFiles: rawSessionState["track_files"] as boolean | undefined,
    trackEnv: rawSessionState["track_env"] as boolean | undefined,
    trackToolResults: rawSessionState["track_tool_results"] as boolean | undefined,
    trackFindings: rawSessionState["track_findings"] as boolean | undefined,
    maxModifiedFiles: rawSessionState["max_modified_files"] as number | undefined,
    maxEnvFacts: rawSessionState["max_env_facts"] as number | undefined,
    maxToolSummaries: rawSessionState["max_tool_summaries"] as number | undefined,
    maxFindings: rawSessionState["max_findings"] as number | undefined,
  };
}

function parseMultiServerLspConfig(rawServers: unknown): LSPConfig {
  return {
    servers: (rawServers as Array<Record<string, unknown>>).map((server) => ({
      command: server["command"] as string,
      args: (server["args"] as string[] | undefined) ?? ["--stdio"],
      languages: (server["languages"] as string[] | undefined) ?? ["typescript"],
      extensions: (server["extensions"] as string[] | undefined) ?? [".ts"],
      timeout: (server["timeout"] as number | undefined) ?? 10_000,
      diagnosticTimeout: server["diagnostic_timeout"] as number | undefined,
    })),
  };
}

function parseLegacyLspConfig(rawLsp: Record<string, unknown>): LSPConfig {
  const languageId = (rawLsp["language_id"] as string | undefined) ?? "typescript";
  const defaults = getLanguageDefaults(languageId);
  return {
    servers: [{
      command: rawLsp["command"] as string,
      args: (rawLsp["args"] as string[] | undefined) ?? ["--stdio"],
      languages: [languageId],
      extensions: defaults?.extensions ?? [".ts"],
      timeout: (rawLsp["timeout"] as number | undefined) ?? 10_000,
      diagnosticTimeout: rawLsp["diagnostic_timeout"] as number | undefined,
    }],
  };
}

function getLanguageDefaults(languageId: string): { extensions: string[] } | undefined {
  const exts = LANGUAGE_EXTENSIONS[languageId];
  return exts ? { extensions: [...exts] } : undefined;
}
