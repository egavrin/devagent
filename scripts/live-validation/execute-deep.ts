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

interface CliOptions {
  readonly outputRoot?: string;
  readonly provider: string;
  readonly model: string;
  readonly skipPrereqs: boolean;
  readonly includeProviderSmoke: boolean;
}

interface CommandResult {
  readonly label: string;
  readonly command: string;
  readonly exitCode: number;
  readonly stdout: string;
  readonly stderr: string;
  readonly durationMs: number;
}

interface ExecuteDeepSummary {
  readonly provider: string;
  readonly model: string;
  readonly outputRoot: string;
  readonly prerequisiteResults: ReadonlyArray<CommandResult>;
  readonly canonicalFlow: AggregateValidationSummary;
  readonly continuityChecks: AggregateValidationSummary;
  readonly fullSuiteRemainder: AggregateValidationSummary;
  readonly providerSmoke?: CommandResult;
  readonly releaseRecommendation: "ship" | "fix-before-ship" | "blocked";
}

type ExecuteDeepSummaryInput = Omit<ExecuteDeepSummary, "releaseRecommendation">;

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

function parseArgs(argv: string[]): CliOptions {
  let outputRoot: string | undefined;
  let provider = "chatgpt";
  let model = "gpt-5.4";
  let skipPrereqs = false;
  let includeProviderSmoke = false;

  for (let index = 2; index < argv.length; index++) {
    const arg = argv[index]!;
    if (arg === "--output-dir" && argv[index + 1]) {
      outputRoot = argv[++index]!;
    } else if (arg === "--provider" && argv[index + 1]) {
      provider = argv[++index]!;
    } else if (arg === "--model" && argv[index + 1]) {
      model = argv[++index]!;
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
        "",
        "Options:",
        "  --output-dir <path>       Write the validation packet to a specific directory",
        "  --provider <provider>     Provider to use for execute scenarios (default: chatgpt)",
        "  --model <model>           Model to use for execute scenarios (default: gpt-5.4)",
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
    process.stdout.write(`  -> ${report.status}\n`);
  }

  const summary = summarizeScenarioReports(reports, {
    provider: context.provider,
    model: context.model,
    suite: "scenario",
  });
  await writeFile(join(outputRoot, "summary.json"), JSON.stringify(summary, null, 2));
  await writeFile(join(outputRoot, "summary.md"), renderSummaryMarkdown(summary));
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

function computeReleaseRecommendation(summary: ExecuteDeepSummaryInput): ExecuteDeepSummary["releaseRecommendation"] {
  const prereqFailed = summary.prerequisiteResults.some((result) => result.exitCode !== 0);
  if (prereqFailed) {
    return "blocked";
  }

  const scenarioSummaries = [summary.canonicalFlow, summary.continuityChecks, summary.fullSuiteRemainder];
  if (scenarioSummaries.some((entry) => entry.failed > 0)) {
    return "fix-before-ship";
  }
  if (scenarioSummaries.some((entry) => entry.blocked > 0)) {
    return "blocked";
  }
  if (summary.providerSmoke && summary.providerSmoke.exitCode !== 0) {
    return "fix-before-ship";
  }
  return "ship";
}

function renderExecuteDeepMarkdown(summary: ExecuteDeepSummary): string {
  return [
    "# Execute Deep Validation",
    "",
    `- Provider: ${summary.provider}`,
    `- Model: ${summary.model}`,
    `- Output root: ${summary.outputRoot}`,
    `- Release recommendation: ${summary.releaseRecommendation}`,
    "",
    "## Prerequisites",
    "",
    ...renderPrerequisiteMarkdown(summary.prerequisiteResults),
    "",
    "## Canonical Staged Flow",
    "",
    renderSummaryMarkdown(summary.canonicalFlow).trim(),
    "",
    "## Continuity Checks",
    "",
    renderSummaryMarkdown(summary.continuityChecks).trim(),
    "",
    "## Full Suite Remainder",
    "",
    renderSummaryMarkdown(summary.fullSuiteRemainder).trim(),
    "",
    ...(summary.providerSmoke
      ? [
          "## Provider Smoke",
          "",
          `- Status: ${summary.providerSmoke.exitCode === 0 ? "passed" : "failed"}`,
          `- Command: \`${summary.providerSmoke.command}\``,
          "",
        ]
      : []),
  ].join("\n").trim() + "\n";
}

async function main(): Promise<void> {
  const options = parseArgs(process.argv);
  const scriptDir = dirname(fileURLToPath(import.meta.url));
  const devagentRoot = dirname(dirname(scriptDir));
  const scenarios = loadValidationScenarios(join(scriptDir, "scenarios"));
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
      process.stdout.write(`  -> ${result.exitCode === 0 ? "passed" : "failed"}\n`);
      if (result.exitCode !== 0) {
        break;
      }
    }
  }

  const prereqsPassed = prerequisiteResults.every((result) => result.exitCode === 0);
  const canonicalFlow = prereqsPassed || options.skipPrereqs
    ? await runScenarioGroup(
        scenarios,
        [...CANONICAL_FLOW_SCENARIOS],
        join(outputRoot, "canonical-staged-flow"),
        {
          devagentRoot,
          provider: options.provider,
          model: options.model,
        },
        "canonical flow",
      )
    : summarizeScenarioReports([], { provider: options.provider, model: options.model, suite: "scenario" });

  const continuityChecks = prereqsPassed || options.skipPrereqs
    ? await runScenarioGroup(
        scenarios,
        [...CONTINUITY_SCENARIOS],
        join(outputRoot, "continuity-checks"),
        {
          devagentRoot,
          provider: options.provider,
          model: options.model,
        },
        "continuity",
      )
    : summarizeScenarioReports([], { provider: options.provider, model: options.model, suite: "scenario" });

  const alreadyRun = new Set<string>([...CANONICAL_FLOW_SCENARIOS, ...CONTINUITY_SCENARIOS]);
  const fullSuiteRemainderIds = scenarios
    .filter((scenario) => scenario.suites.includes("full") && !alreadyRun.has(scenario.id))
    .map((scenario) => scenario.id);
  const fullSuiteRemainder = prereqsPassed || options.skipPrereqs
    ? await runScenarioGroup(
        scenarios,
        fullSuiteRemainderIds,
        join(outputRoot, "full-suite"),
        {
          devagentRoot,
          provider: options.provider,
          model: options.model,
        },
        "full suite",
      )
    : summarizeScenarioReports([], { provider: options.provider, model: options.model, suite: "scenario" });

  const providerSmoke = options.includeProviderSmoke
    ? await runCommand(
        "provider-smoke",
        `bun run validate:live:provider-smoke --output-dir ${JSON.stringify(join(outputRoot, "provider-smoke"))}`,
        devagentRoot,
        join(outputRoot, "provider-smoke-run"),
      )
    : undefined;

  const summaryWithoutRecommendation: ExecuteDeepSummaryInput = {
    provider: options.provider,
    model: options.model,
    outputRoot,
    prerequisiteResults,
    canonicalFlow,
    continuityChecks,
    fullSuiteRemainder,
    ...(providerSmoke ? { providerSmoke } : {}),
  };
  const summary: ExecuteDeepSummary = {
    ...summaryWithoutRecommendation,
    releaseRecommendation: computeReleaseRecommendation(summaryWithoutRecommendation),
  };

  await writeFile(join(outputRoot, "execute-deep-summary.json"), JSON.stringify(summary, null, 2));
  await writeFile(join(outputRoot, "execute-deep-summary.md"), renderExecuteDeepMarkdown(summary));

  process.stdout.write(`Execute deep validation packet written to ${outputRoot}\n`);
  if (summary.releaseRecommendation !== "ship") {
    process.exit(1);
  }
}

main().catch((error) => {
  process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
  process.exit(1);
});
