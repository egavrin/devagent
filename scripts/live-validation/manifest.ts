import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import type {
  CliScenarioInvocation,
  ExecuteScenarioInvocation,
  ScenarioPreSetupStep,
  ToolBatchRequirement,
  ToolCallRequirement,
  ValidationAssertion,
  ValidationScenario,
  ValidationSuite,
  VerificationCommand,
} from "./types";

const VALID_SUITES = new Set<ValidationSuite>(["smoke", "full"]);
const VALID_TARGET_REPOS = new Set([
  "arkcompiler_ets_frontend",
  "arkcompiler_runtime_core",
  "arkcompiler_runtime_core_docs",
]);
const VALID_SURFACES = new Set(["execute", "cli"]);
const VALID_TASK_SHAPES = new Set(["readonly", "review", "implement", "repair"]);
const VALID_ISOLATION_MODES = new Set(["temp-copy", "worktree"]);
const VALID_EXECUTE_TASK_TYPES = new Set(["triage", "plan", "implement", "review", "repair"]);
const VALID_ASSERTION_SOURCES = new Set(["stdout", "stderr", "repoDiff", "repoStatus", "events", "artifact"]);
const VALID_ASSERTION_TYPES = new Set(["contains", "matches"]);
const VALID_COMMAND_CWDS = new Set(["repo", "linter"]);

function expectString(value: unknown, label: string, filePath: string): string {
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new Error(`Invalid ${label} in ${filePath}: expected non-empty string.`);
  }
  return value;
}

function expectStringArray(value: unknown, label: string, filePath: string): string[] {
  if (!Array.isArray(value) || value.some((item) => typeof item !== "string" || item.trim().length === 0)) {
    throw new Error(`Invalid ${label} in ${filePath}: expected array of non-empty strings.`);
  }
  return value;
}

function parseVerificationCommands(raw: unknown, filePath: string): VerificationCommand[] {
  if (!Array.isArray(raw)) {
    throw new Error(`Invalid verificationCommands in ${filePath}: expected array.`);
  }
  return raw.map((entry, index) => {
    if (!entry || typeof entry !== "object") {
      throw new Error(`Invalid verificationCommands[${index}] in ${filePath}: expected object.`);
    }
    const record = entry as Record<string, unknown>;
    const cwd = record["cwd"];
    if (cwd !== undefined && (!VALID_COMMAND_CWDS.has(String(cwd)))) {
      throw new Error(`Invalid verificationCommands[${index}].cwd in ${filePath}: ${String(cwd)}`);
    }
    return {
      command: expectString(record["command"], `verificationCommands[${index}].command`, filePath),
      ...(cwd ? { cwd: cwd as "repo" | "linter" } : {}),
    };
  });
}

function parseRequiredToolCalls(raw: unknown, filePath: string): ToolCallRequirement[] | undefined {
  if (raw === undefined) return undefined;
  if (!Array.isArray(raw)) {
    throw new Error(`Invalid requiredToolCalls in ${filePath}: expected array.`);
  }
  return raw.map((entry, index) => {
    if (!entry || typeof entry !== "object") {
      throw new Error(`Invalid requiredToolCalls[${index}] in ${filePath}: expected object.`);
    }
    const record = entry as Record<string, unknown>;
    const minCalls = record["minCalls"];
    if (!Number.isInteger(minCalls) || Number(minCalls) < 1) {
      throw new Error(`Invalid requiredToolCalls[${index}].minCalls in ${filePath}: expected integer >= 1.`);
    }
    return {
      tool: expectString(record["tool"], `requiredToolCalls[${index}].tool`, filePath),
      minCalls: Number(minCalls),
    };
  });
}

function parseRequiredToolBatches(raw: unknown, filePath: string): ToolBatchRequirement[] | undefined {
  if (raw === undefined) return undefined;
  if (!Array.isArray(raw)) {
    throw new Error(`Invalid requiredToolBatches in ${filePath}: expected array.`);
  }
  return raw.map((entry, index) => {
    if (!entry || typeof entry !== "object") {
      throw new Error(`Invalid requiredToolBatches[${index}] in ${filePath}: expected object.`);
    }
    const record = entry as Record<string, unknown>;
    const minBatches = record["minBatches"];
    const minBatchSize = record["minBatchSize"];
    if (!Number.isInteger(minBatches) || Number(minBatches) < 1) {
      throw new Error(`Invalid requiredToolBatches[${index}].minBatches in ${filePath}: expected integer >= 1.`);
    }
    if (!Number.isInteger(minBatchSize) || Number(minBatchSize) < 2) {
      throw new Error(`Invalid requiredToolBatches[${index}].minBatchSize in ${filePath}: expected integer >= 2.`);
    }
    return {
      tool: expectString(record["tool"], `requiredToolBatches[${index}].tool`, filePath),
      minBatches: Number(minBatches),
      minBatchSize: Number(minBatchSize),
    };
  });
}

function parseAssertions(raw: unknown, filePath: string): ValidationAssertion[] {
  if (!Array.isArray(raw)) {
    throw new Error(`Invalid assertions in ${filePath}: expected array.`);
  }
  return raw.map((entry, index) => {
    if (!entry || typeof entry !== "object") {
      throw new Error(`Invalid assertions[${index}] in ${filePath}: expected object.`);
    }
    const record = entry as Record<string, unknown>;
    const type = expectString(record["type"], `assertions[${index}].type`, filePath);
    const source = expectString(record["source"], `assertions[${index}].source`, filePath);
    if (!VALID_ASSERTION_TYPES.has(type)) {
      throw new Error(`Invalid assertions[${index}].type in ${filePath}: ${type}`);
    }
    if (!VALID_ASSERTION_SOURCES.has(source)) {
      throw new Error(`Invalid assertions[${index}].source in ${filePath}: ${source}`);
    }
    const path = record["path"];
    if (path !== undefined && typeof path !== "string") {
      throw new Error(`Invalid assertions[${index}].path in ${filePath}: expected string.`);
    }
    if (type === "contains") {
      return {
        type,
        source: source as ValidationAssertion["source"],
        ...(path ? { path } : {}),
        value: expectString(record["value"], `assertions[${index}].value`, filePath),
      };
    }
    return {
      type,
      source: source as ValidationAssertion["source"],
      ...(path ? { path } : {}),
      pattern: expectString(record["pattern"], `assertions[${index}].pattern`, filePath),
    };
  });
}

function parsePreSetup(raw: unknown, filePath: string): ScenarioPreSetupStep[] | undefined {
  if (raw === undefined) return undefined;
  if (!Array.isArray(raw)) {
    throw new Error(`Invalid preSetup in ${filePath}: expected array.`);
  }
  return raw.map((entry, index) => {
    if (!entry || typeof entry !== "object") {
      throw new Error(`Invalid preSetup[${index}] in ${filePath}: expected object.`);
    }
    const record = entry as Record<string, unknown>;
    const kind = expectString(record["kind"], `preSetup[${index}].kind`, filePath);
    if (kind === "write-file") {
      const hasContent = typeof record["content"] === "string";
      const hasTemplate = typeof record["templateFile"] === "string";
      if (!hasContent && !hasTemplate) {
        throw new Error(`Invalid preSetup[${index}] in ${filePath}: write-file requires content or templateFile.`);
      }
      return {
        kind,
        path: expectString(record["path"], `preSetup[${index}].path`, filePath),
        ...(hasContent ? { content: record["content"] as string } : {}),
        ...(hasTemplate ? { templateFile: record["templateFile"] as string } : {}),
        ...(record["executable"] === true ? { executable: true } : {}),
      };
    }
    if (kind === "run-command") {
      const cwd = record["cwd"];
      if (cwd !== undefined && !VALID_COMMAND_CWDS.has(String(cwd))) {
        throw new Error(`Invalid preSetup[${index}].cwd in ${filePath}: ${String(cwd)}`);
      }
      return {
        kind,
        command: expectString(record["command"], `preSetup[${index}].command`, filePath),
        ...(cwd ? { cwd: cwd as "repo" | "linter" } : {}),
      };
    }
    throw new Error(`Invalid preSetup[${index}].kind in ${filePath}: ${kind}`);
  });
}

function parseExecuteInvocation(raw: Record<string, unknown>, filePath: string): ExecuteScenarioInvocation {
  const taskType = expectString(raw["taskType"], "invocation.taskType", filePath);
  if (!VALID_EXECUTE_TASK_TYPES.has(taskType)) {
    throw new Error(`Invalid invocation.taskType in ${filePath}: ${taskType}`);
  }
  const extraInstructions = raw["extraInstructions"];
  return {
    type: "execute",
    taskType: taskType as ExecuteScenarioInvocation["taskType"],
    workItemTitle: expectString(raw["workItemTitle"], "invocation.workItemTitle", filePath),
    summary: expectString(raw["summary"], "invocation.summary", filePath),
    ...(typeof raw["issueBody"] === "string" ? { issueBody: raw["issueBody"] } : {}),
    ...(Array.isArray(extraInstructions)
      ? { extraInstructions: expectStringArray(extraInstructions, "invocation.extraInstructions", filePath) }
      : {}),
    ...(typeof raw["maxIterations"] === "number" ? { maxIterations: raw["maxIterations"] } : {}),
  };
}

function parseCliInvocation(raw: Record<string, unknown>, filePath: string): CliScenarioInvocation {
  const approvalMode = raw["approvalMode"];
  const reasoning = raw["reasoning"];
  return {
    type: "cli",
    query: expectString(raw["query"], "invocation.query", filePath),
    ...(typeof raw["maxIterations"] === "number" ? { maxIterations: raw["maxIterations"] } : {}),
    ...(typeof approvalMode === "string" ? { approvalMode: approvalMode as CliScenarioInvocation["approvalMode"] } : {}),
    ...(typeof reasoning === "string" ? { reasoning: reasoning as CliScenarioInvocation["reasoning"] } : {}),
    ...(Array.isArray(raw["extraArgs"])
      ? { extraArgs: expectStringArray(raw["extraArgs"], "invocation.extraArgs", filePath) }
      : {}),
  };
}

export function validateScenarioManifest(raw: unknown, filePath: string): ValidationScenario {
  if (!raw || typeof raw !== "object") {
    throw new Error(`Invalid manifest ${filePath}: expected object.`);
  }
  const record = raw as Record<string, unknown>;
  const suites = expectStringArray(record["suites"], "suites", filePath) as ValidationSuite[];
  if (suites.some((entry) => !VALID_SUITES.has(entry))) {
    throw new Error(`Invalid suites in ${filePath}: expected smoke/full only.`);
  }

  const targetRepo = expectString(record["targetRepo"], "targetRepo", filePath);
  if (!VALID_TARGET_REPOS.has(targetRepo)) {
    throw new Error(`Invalid targetRepo in ${filePath}: ${targetRepo}`);
  }
  const surface = expectString(record["surface"], "surface", filePath);
  if (!VALID_SURFACES.has(surface)) {
    throw new Error(`Invalid surface in ${filePath}: ${surface}`);
  }
  const taskShape = expectString(record["taskShape"], "taskShape", filePath);
  if (!VALID_TASK_SHAPES.has(taskShape)) {
    throw new Error(`Invalid taskShape in ${filePath}: ${taskShape}`);
  }
  const isolationMode = expectString(record["isolationMode"], "isolationMode", filePath);
  if (!VALID_ISOLATION_MODES.has(isolationMode)) {
    throw new Error(`Invalid isolationMode in ${filePath}: ${isolationMode}`);
  }

  const invocationRaw = record["invocation"];
  if (!invocationRaw || typeof invocationRaw !== "object") {
    throw new Error(`Invalid invocation in ${filePath}: expected object.`);
  }
  const invocationRecord = invocationRaw as Record<string, unknown>;
  const invocationType = expectString(invocationRecord["type"], "invocation.type", filePath);
  const invocation = invocationType === "execute"
    ? parseExecuteInvocation(invocationRecord, filePath)
    : invocationType === "cli"
      ? parseCliInvocation(invocationRecord, filePath)
      : (() => {
          throw new Error(`Invalid invocation.type in ${filePath}: ${invocationType}`);
        })();

  const variablesRaw = record["variables"];
  const variables = variablesRaw && typeof variablesRaw === "object"
    ? Object.fromEntries(
        Object.entries(variablesRaw as Record<string, unknown>).map(([key, value]) => [
          key,
          expectString(value, `variables.${key}`, filePath),
        ]),
      )
    : undefined;

  const timeoutMs = record["timeoutMs"];
  if (timeoutMs !== undefined && (typeof timeoutMs !== "number" || timeoutMs <= 0)) {
    throw new Error(`Invalid timeoutMs in ${filePath}: expected positive number.`);
  }

  return {
    id: expectString(record["id"], "id", filePath),
    description: expectString(record["description"], "description", filePath),
    suites,
    targetRepo: targetRepo as ValidationScenario["targetRepo"],
    surface: surface as ValidationScenario["surface"],
    taskShape: taskShape as ValidationScenario["taskShape"],
    isolationMode: isolationMode as ValidationScenario["isolationMode"],
    ...(parsePreSetup(record["preSetup"], filePath) ? { preSetup: parsePreSetup(record["preSetup"], filePath)! } : {}),
    invocation,
    expectedArtifacts: expectStringArray(record["expectedArtifacts"], "expectedArtifacts", filePath),
    assertions: parseAssertions(record["assertions"], filePath),
    verificationCommands: parseVerificationCommands(record["verificationCommands"], filePath),
    cleanupPolicy: record["cleanupPolicy"] === "destroy"
      ? "destroy"
      : (() => {
          throw new Error(`Invalid cleanupPolicy in ${filePath}: expected "destroy".`);
        })(),
    ...(variables ? { variables } : {}),
    ...(record["baselineAfterSetup"] === true ? { baselineAfterSetup: true } : {}),
    ...(record["requiresChatgptAuth"] === true ? { requiresChatgptAuth: true } : {}),
    ...(record["requiresArktsLinter"] === true ? { requiresArktsLinter: true } : {}),
    ...(typeof timeoutMs === "number" ? { timeoutMs } : {}),
    ...(parseRequiredToolCalls(record["requiredToolCalls"], filePath)
      ? { requiredToolCalls: parseRequiredToolCalls(record["requiredToolCalls"], filePath)! }
      : {}),
    ...(parseRequiredToolBatches(record["requiredToolBatches"], filePath)
      ? { requiredToolBatches: parseRequiredToolBatches(record["requiredToolBatches"], filePath)! }
      : {}),
  };
}

export function loadValidationScenarios(
  scenariosDir: string,
): ValidationScenario[] {
  const files = readdirSync(scenariosDir)
    .filter((entry) => entry.endsWith(".json"))
    .sort();
  return files.map((fileName) => {
    const filePath = join(scenariosDir, fileName);
    const raw = JSON.parse(readFileSync(filePath, "utf-8")) as unknown;
    return validateScenarioManifest(raw, filePath);
  });
}
