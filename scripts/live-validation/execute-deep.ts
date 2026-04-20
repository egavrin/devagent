#!/usr/bin/env bun

import { spawn } from "node:child_process";
import { mkdir, mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { loadValidationScenarios } from "./manifest";
import { renderSummaryMarkdown, summarizeScenarioReports } from "./reporting";
import { runValidationScenario } from "./runner";
import type { AggregateValidationSummary, ValidationScenario, ValidationScenarioReport } from "./types";

export type ExecuteDeepGroupId = "canonical" | "continuity" | "remainder";
export type ExecuteDeepCoverageMode = "full" | "partial";
export type ExecuteDeepRunStatus = "passed" | "failed" | "blocked";
export type ExecuteDeepRecommendation = "ship" | "fix-before-ship" | "blocked" | "partial";

interface CliOptions {
  readonly outputRoot?: string;
  readonly provider: string;
  readonly model: string;
  readonly skipPrereqs: boolean;
  readonly includeProviderSmoke: boolean;
  readonly selectedGroups: ReadonlyArray<ExecuteDeepGroupId>;
}

interface CommandResult {
  readonly label: string;
  readonly command: string;
  readonly exitCode: number;
  readonly stdout: string;
  readonly stderr: string;
  readonly durationMs: number;
}

interface ExecuteDeepTimingSummary {
  readonly totalDurationMs: number;
  readonly prerequisiteDurationMs: number;
  readonly groupDurationMs: Readonly<Record<ExecuteDeepGroupId, number>>;
  readonly providerSmokeDurationMs?: number;
}

interface ExecuteDeepSummaryBase {
  readonly provider: string;
  readonly model: string;
  readonly outputRoot: string;
  readonly selectedGroups: ReadonlyArray<ExecuteDeepGroupId>;
  readonly coverageMode: ExecuteDeepCoverageMode;
  readonly prerequisiteResults: ReadonlyArray<CommandResult>;
  readonly canonicalFlow: AggregateValidationSummary;
  readonly continuityChecks: AggregateValidationSummary;
  readonly fullSuiteRemainder: AggregateValidationSummary;
  readonly timing: ExecuteDeepTimingSummary;
  readonly runStatus: ExecuteDeepRunStatus;
  readonly providerSmoke?: CommandResult;
}

export interface ExecuteDeepSummary extends ExecuteDeepSummaryBase {
  readonly releaseRecommendation: ExecuteDeepRecommendation;
}

type ExecuteDeepSummaryInput = Omit<ExecuteDeepSummary, "releaseRecommendation">;

interface ScenarioGroupDefinition {
  readonly id: ExecuteDeepGroupId;
  readonly label: string;
  readonly outputDirName: string;
  readonly suiteName: string;
}

export const EXECUTE_DEEP_GROUP_ORDER = [
  "canonical",
  "continuity",
  "remainder",
] as const satisfies ReadonlyArray<ExecuteDeepGroupId>;

const GROUP_DEFINITIONS: ReadonlyArray<ScenarioGroupDefinition> = [
  {
    id: "canonical",
    label: "Canonical Staged Flow",
    outputDirName: "canonical-staged-flow",
    suiteName: "canonical flow",
  },
  {
    id: "continuity",
    label: "Continuity Checks",
    outputDirName: "continuity-checks",
    suiteName: "continuity",
  },
  {
    id: "remainder",
    label: "Full Suite Remainder",
    outputDirName: "full-suite",
    suiteName: "full suite",
  },
] as const;

const CANONICAL_FLOW_SCENARIOS = [
  "runtime-core-docs-execute-design",
  "runtime-core-docs-execute-breakdown",
  "runtime-core-docs-execute-issue-generation",
] as const;

const CONTINUITY_SCENARIOS = [
  "runtime-core-execute-plan",
  "runtime-core-execute-triage",
  "ets-frontend-execute-repair",
] as const;

function isExecuteDeepGroupId(value: string): value is ExecuteDeepGroupId {
  return EXECUTE_DEEP_GROUP_ORDER.includes(value as ExecuteDeepGroupId);
}

export function normalizeSelectedGroups(requestedGroups: ReadonlyArray<ExecuteDeepGroupId>): ExecuteDeepGroupId[] {
  if (requestedGroups.length === 0) {
    return [...EXECUTE_DEEP_GROUP_ORDER];
  }

  const selected = new Set(requestedGroups);
  return EXECUTE_DEEP_GROUP_ORDER.filter((groupId) => selected.has(groupId));
}

export function parseArgs(argv: string[]): CliOptions {
  let outputRoot: string | undefined;
  let provider = "chatgpt";
  let model = "gpt-5.4";
  let skipPrereqs = false;
  let includeProviderSmoke = false;
  const requestedGroups: ExecuteDeepGroupId[] = [];

  for (let index = 2; index < argv.length; index++) {
    const arg = argv[index]!;
    if (arg === "--output-dir" && argv[index + 1]) {
      outputRoot = argv[++index]!;
    } else if (arg === "--provider" && argv[index + 1]) {
      provider = argv[++index]!;
    } else if (arg === "--model" && argv[index + 1]) {
      model = argv[++index]!;
    } else if (arg === "--only" && argv[index + 1]) {
      const value = argv[++index]!;
      if (!isExecuteDeepGroupId(value)) {
        throw new Error(`Invalid --only group: ${value}`);
      }
      requestedGroups.push(value);
    } else if (arg === "--skip-prereqs") {
      skipPrereqs = true;
    } else if (arg === "--include-provider-smoke") {
      includeProviderSmoke = true;
    } else if (arg === "--help" || arg === "-h") {
      process.stdout.write([
        "Deep execute live validation",
        "",
        "Usage:",
        "  bun run scripts/live-validation/execute-deep.ts",
        "  bun run scripts/live-validation/execute-deep.ts --output-dir <path>",
        "  bun run scripts/live-validation/execute-deep.ts --provider chatgpt --model gpt-5.4",
        "  bun run scripts/live-validation/execute-deep.ts --only canonical --skip-prereqs",
        "",
        "Options:",
        "  --output-dir <path>       Write the validation packet to a specific directory",
        "  --provider <provider>     Provider to use for execute scenarios (default: chatgpt)",
        "  --model <model>           Model to use for execute scenarios (default: gpt-5.4)",
        "  --only <group>            Run only canonical, continuity, or remainder (repeatable)",
        "  --skip-prereqs            Skip build and smoke prerequisites",
        "  --include-provider-smoke  Run provider smoke after the execute sweep",
        "",
      ].join("\n"));
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }

  return {
    outputRoot,
    provider,
    model,
    skipPrereqs,
    includeProviderSmoke,
    selectedGroups: normalizeSelectedGroups(requestedGroups),
  };
}

async function runCommand(
  label: string,
  command: string,
  cwd: string,
  outputDir: string,
): Promise<CommandResult> {
  await mkdir(outputDir, { recursive: true });
  const startedAt = Date.now();
  const result = await new Promise<CommandResult>((resolvePromise) => {
    const child = spawn(process.env["SHELL"] ?? "/bin/sh", ["-lc", command], {
      cwd,
      env: {
        ...process.env,
        DEVAGENT_DISABLE_UPDATE_CHECK: "1",
      },
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk: Buffer) => {
      stdout += chunk.toString();
    });
    child.stderr.on("data", (chunk: Buffer) => {
      stderr += chunk.toString();
    });
    child.once("close", (code) => {
      resolvePromise({
        label,
        command,
        exitCode: code ?? 1,
        stdout,
        stderr,
        durationMs: Date.now() - startedAt,
      });
    });
  });

  await Promise.all([
    writeFile(join(outputDir, "stdout.txt"), result.stdout),
    writeFile(join(outputDir, "stderr.txt"), result.stderr),
    writeFile(join(outputDir, "result.json"), JSON.stringify(result, null, 2)),
  ]);
  return result;
}

function selectScenarioById(
  scenarios: ReadonlyArray<ValidationScenario>,
  id: string,
): ValidationScenario {
  const scenario = scenarios.find((entry) => entry.id === id);
  if (!scenario) {
    throw new Error(`Missing validation scenario: ${id}`);
  }
  return scenario;
}

export function buildScenarioGroups(
  scenarios: ReadonlyArray<ValidationScenario>,
): Record<ExecuteDeepGroupId, ReadonlyArray<string>> {
  const alreadyRun = new Set<string>([...CANONICAL_FLOW_SCENARIOS, ...CONTINUITY_SCENARIOS]);
  const remainder = scenarios
    .filter((scenario) => scenario.suites.includes("full") && !alreadyRun.has(scenario.id))
    .map((scenario) => scenario.id);

  return {
    canonical: [...CANONICAL_FLOW_SCENARIOS],
    continuity: [...CONTINUITY_SCENARIOS],
    remainder,
  };
}

function createEmptySummary(provider: string, model: string): AggregateValidationSummary {
  return summarizeScenarioReports([], { provider, model, suite: "scenario" });
}

async function runScenarioGroup(
  scenarios: ReadonlyArray<ValidationScenario>,
  scenarioIds: ReadonlyArray<string>,
  outputRoot: string,
  context: {
    readonly devagentRoot: string;
    readonly provider: string;
    readonly model: string;
  },
  suiteName: string,
): Promise<AggregateValidationSummary> {
  await mkdir(outputRoot, { recursive: true });
  const reports: ValidationScenarioReport[] = [];
  for (const id of scenarioIds) {
    const scenario = selectScenarioById(scenarios, id);
    process.stdout.write(`Running ${suiteName}: ${scenario.id}\n`);
    const report = await runValidationScenario(scenario, {
      devagentRoot: context.devagentRoot,
      provider: context.provider,
      model: context.model,
      outputRoot,
    });
    reports.push(report);
    process.stdout.write(`  -> ${report.status} (${report.durationMs} ms)\n`);
  }

  const summary = summarizeScenarioReports(reports, {
    provider: context.provider,
    model: context.model,
    suite: "scenario",
  });
  await writeFile(join(outputRoot, "summary.json"), JSON.stringify(summary, null, 2));
  await writeFile(join(outputRoot, "summary.md"), renderSummaryMarkdown(summary));
  process.stdout.write(
    `Completed ${suiteName}: passed=${summary.passed} failed=${summary.failed} blocked=${summary.blocked} duration=${summary.durationMs} ms\n`,
  );
  return summary;
}

function renderPrerequisiteMarkdown(results: ReadonlyArray<CommandResult>): string[] {
  if (results.length === 0) {
    return ["No prerequisites were run."];
  }
  return results.flatMap((result) => [
    `- ${result.label}: ${result.exitCode === 0 ? "passed" : "failed"} (${result.durationMs} ms)`,
    `  - Command: \`${result.command}\``,
  ]);
}

function selectedGroupSummary(
  summary: ExecuteDeepSummaryBase,
  groupId: ExecuteDeepGroupId,
): AggregateValidationSummary {
  if (groupId === "canonical") return summary.canonicalFlow;
  if (groupId === "continuity") return summary.continuityChecks;
  return summary.fullSuiteRemainder;
}

export function computeRunStatus(summary: ExecuteDeepSummaryBase): ExecuteDeepRunStatus {
  if (summary.prerequisiteResults.some((result) => result.exitCode !== 0)) {
    return "blocked";
  }

  const selectedSummaries = summary.selectedGroups.map((groupId) => selectedGroupSummary(summary, groupId));
  if (selectedSummaries.some((entry) => entry.blocked > 0)) {
    return "blocked";
  }
  if (selectedSummaries.some((entry) => entry.failed > 0)) {
    return "failed";
  }
  if (summary.providerSmoke && summary.providerSmoke.exitCode !== 0) {
    return "failed";
  }
  return "passed";
}

export function computeReleaseRecommendation(summary: ExecuteDeepSummaryInput): ExecuteDeepRecommendation {
  if (summary.coverageMode === "partial") {
    return "partial";
  }
  if (summary.runStatus === "blocked") {
    return "blocked";
  }
  if (summary.runStatus === "failed") {
    return "fix-before-ship";
  }
  return "ship";
}

function renderGroupSection(
  summary: ExecuteDeepSummary,
  group: ScenarioGroupDefinition,
): string {
  const lines = [
    `## ${group.label}`,
    "",
    `- Selected: ${summary.selectedGroups.includes(group.id) ? "yes" : "no"}`,
    `- Duration: ${summary.timing.groupDurationMs[group.id]} ms`,
  ];

  if (!summary.selectedGroups.includes(group.id)) {
    lines.push("- Not selected in this run.");
    return `${lines.join("\n")}\n`;
  }

  lines.push("");
  lines.push(renderSummaryMarkdown(selectedGroupSummary(summary, group.id)).trim());
  lines.push("");
  return `${lines.join("\n").trimEnd()}\n`;
}

export function renderExecuteDeepMarkdown(summary: ExecuteDeepSummary): string {
  return [
    "# Execute Deep Validation",
    "",
    `- Provider: ${summary.provider}`,
    `- Model: ${summary.model}`,
    `- Output root: ${summary.outputRoot}`,
    `- Coverage mode: ${summary.coverageMode}`,
    `- Selected groups: ${summary.selectedGroups.join(", ")}`,
    `- Run status: ${summary.runStatus}`,
    `- Release recommendation: ${summary.releaseRecommendation}`,
    `- Total duration: ${summary.timing.totalDurationMs} ms`,
    `- Prerequisite duration: ${summary.timing.prerequisiteDurationMs} ms`,
    ...(summary.coverageMode === "partial"
      ? ["", "This packet covers a focused subset of execute-deep groups and is not release-complete."]
      : []),
    "",
    "## Prerequisites",
    "",
    ...renderPrerequisiteMarkdown(summary.prerequisiteResults),
    "",
    ...GROUP_DEFINITIONS.flatMap((group) => [renderGroupSection(summary, group)]),
    ...(summary.providerSmoke
      ? [
          "## Provider Smoke",
          "",
          `- Status: ${summary.providerSmoke.exitCode === 0 ? "passed" : "failed"}`,
          `- Command: \`${summary.providerSmoke.command}\``,
          `- Duration: ${summary.providerSmoke.durationMs} ms`,
          "",
        ]
      : []),
  ].join("\n").trim() + "\n";
}

async function main(): Promise<void> {
  const startedAt = Date.now();
  const options = parseArgs(process.argv);
  const scriptDir = dirname(fileURLToPath(import.meta.url));
  const devagentRoot = dirname(dirname(scriptDir));
  const scenarios = loadValidationScenarios(join(scriptDir, "scenarios"));
  const groupScenarioIds = buildScenarioGroups(scenarios);
  const outputRoot = options.outputRoot
    ? join(options.outputRoot)
    : await mkdtemp(join(tmpdir(), "devagent-execute-deep-"));
  await mkdir(outputRoot, { recursive: true });

  const prerequisiteResults: CommandResult[] = [];
  if (!options.skipPrereqs) {
    const prereqRoot = join(outputRoot, "prereqs");
    const prereqCommands = [
      { label: "build:publish", command: "bun run build:publish" },
      { label: "test:bundle-smoke", command: "bun run test:bundle-smoke" },
      { label: "test:live-validation", command: "bun run test:live-validation" },
    ];
    for (const prereq of prereqCommands) {
      process.stdout.write(`Running prerequisite: ${prereq.label}\n`);
      const result = await runCommand(prereq.label, prereq.command, devagentRoot, join(prereqRoot, prereq.label));
      prerequisiteResults.push(result);
      process.stdout.write(`  -> ${result.exitCode === 0 ? "passed" : "failed"} (${result.durationMs} ms)\n`);
      if (result.exitCode !== 0) {
        break;
      }
    }
  }

  const coverageMode: ExecuteDeepCoverageMode = options.selectedGroups.length === EXECUTE_DEEP_GROUP_ORDER.length
    ? "full"
    : "partial";
  const canRunScenarios = options.skipPrereqs || prerequisiteResults.every((result) => result.exitCode === 0);

  let canonicalFlow = createEmptySummary(options.provider, options.model);
  let continuityChecks = createEmptySummary(options.provider, options.model);
  let fullSuiteRemainder = createEmptySummary(options.provider, options.model);

  if (canRunScenarios) {
    for (const group of GROUP_DEFINITIONS) {
      if (!options.selectedGroups.includes(group.id)) {
        continue;
      }

      const summary = await runScenarioGroup(
        scenarios,
        groupScenarioIds[group.id],
        join(outputRoot, group.outputDirName),
        {
          devagentRoot,
          provider: options.provider,
          model: options.model,
        },
        group.suiteName,
      );

      if (group.id === "canonical") {
        canonicalFlow = summary;
      } else if (group.id === "continuity") {
        continuityChecks = summary;
      } else {
        fullSuiteRemainder = summary;
      }
    }
  }

  const providerSmoke = options.includeProviderSmoke
    ? await runCommand(
        "provider-smoke",
        `bun run validate:live:provider-smoke --output-dir ${JSON.stringify(join(outputRoot, "provider-smoke"))}`,
        devagentRoot,
        join(outputRoot, "provider-smoke-run"),
      )
    : undefined;

  const timing: ExecuteDeepTimingSummary = {
    totalDurationMs: Date.now() - startedAt,
    prerequisiteDurationMs: prerequisiteResults.reduce((totalDuration, result) => totalDuration + result.durationMs, 0),
    groupDurationMs: {
      canonical: canonicalFlow.durationMs,
      continuity: continuityChecks.durationMs,
      remainder: fullSuiteRemainder.durationMs,
    },
    ...(providerSmoke ? { providerSmokeDurationMs: providerSmoke.durationMs } : {}),
  };

  const summaryWithoutRecommendation = {
    provider: options.provider,
    model: options.model,
    outputRoot,
    selectedGroups: options.selectedGroups,
    coverageMode,
    prerequisiteResults,
    canonicalFlow,
    continuityChecks,
    fullSuiteRemainder,
    timing,
    runStatus: "passed",
    ...(providerSmoke ? { providerSmoke } : {}),
  } satisfies ExecuteDeepSummaryInput;

  const finalizedSummaryWithoutRecommendation: ExecuteDeepSummaryInput = {
    ...summaryWithoutRecommendation,
    runStatus: computeRunStatus(summaryWithoutRecommendation),
  };

  const summary: ExecuteDeepSummary = {
    ...finalizedSummaryWithoutRecommendation,
    releaseRecommendation: computeReleaseRecommendation(finalizedSummaryWithoutRecommendation),
  };

  await writeFile(join(outputRoot, "execute-deep-summary.json"), JSON.stringify(summary, null, 2));
  await writeFile(join(outputRoot, "execute-deep-summary.md"), renderExecuteDeepMarkdown(summary));

  process.stdout.write(`Execute deep validation packet written to ${outputRoot}\n`);
  if (summary.runStatus !== "passed") {
    process.exit(1);
  }
}

if (import.meta.main) {
  main().catch((error) => {
    process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
    process.exit(1);
  });
}
