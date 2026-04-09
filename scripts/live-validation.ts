#!/usr/bin/env bun

import { mkdtemp, mkdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { loadValidationScenarios } from "./live-validation/manifest";
import { renderSummaryMarkdown, summarizeScenarioReports } from "./live-validation/reporting";
import { runValidationScenario } from "./live-validation/runner";
import type { ValidationScenario, ValidationSuite } from "./live-validation/types";

interface CliOptions {
  readonly suite: ValidationSuite;
  readonly scenarioId?: string;
  readonly outputRoot?: string;
  readonly listOnly: boolean;
  readonly provider: string;
  readonly model: string;
}

function parseArgs(argv: string[]): CliOptions {
  let suite: ValidationSuite = "smoke";
  let scenarioId: string | undefined;
  let outputRoot: string | undefined;
  let listOnly = false;
  let provider = "chatgpt";
  let model = "gpt-5.4";

  for (let i = 2; i < argv.length; i++) {
    const arg = argv[i]!;
    if (arg === "--suite" && argv[i + 1]) {
      const value = argv[++i]!;
      if (value === "smoke" || value === "full") {
        suite = value;
      } else {
        throw new Error(`Invalid suite: ${value}`);
      }
    } else if (arg === "--scenario" && argv[i + 1]) {
      scenarioId = argv[++i]!;
    } else if (arg === "--output-dir" && argv[i + 1]) {
      outputRoot = argv[++i]!;
    } else if (arg === "--provider" && argv[i + 1]) {
      provider = argv[++i]!;
    } else if (arg === "--model" && argv[i + 1]) {
      model = argv[++i]!;
    } else if (arg === "--list-scenarios") {
      listOnly = true;
    } else if (arg === "--help" || arg === "-h") {
      printHelp();
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }

  return { suite, scenarioId, outputRoot, listOnly, provider, model };
}

function printHelp(): void {
  process.stdout.write(
    [
      "Live runtime validation harness",
      "",
      "Usage:",
      "  bun run scripts/live-validation.ts --suite smoke",
      "  bun run scripts/live-validation.ts --suite full",
      "  bun run scripts/live-validation.ts --scenario <id>",
      "  bun run scripts/live-validation.ts --suite full --provider openai --model gpt-5.4-mini",
      "  bun run scripts/live-validation.ts --list-scenarios",
      "",
      "Options:",
      "  --suite <smoke|full>   Run a predefined validation suite (default: smoke)",
      "  --scenario <id>        Run a single scenario by id",
      "  --output-dir <path>    Write reports to a specific directory",
      "  --provider <name>      Override the provider used for scenario execution",
      "  --model <id>           Override the model used for scenario execution",
      "  --list-scenarios       Print available scenarios and exit",
      "  -h, --help             Show this help text",
      "",
    ].join("\n"),
  );
}

function filterScenarios(
  scenarios: ReadonlyArray<ValidationScenario>,
  options: CliOptions,
): ValidationScenario[] {
  if (options.scenarioId) {
    const scenario = scenarios.find((entry) => entry.id === options.scenarioId);
    if (!scenario) {
      throw new Error(`Unknown scenario: ${options.scenarioId}`);
    }
    return [scenario];
  }
  return scenarios.filter((scenario) => scenario.suites.includes(options.suite));
}

async function main(): Promise<void> {
  const options = parseArgs(process.argv);
  const scriptDir = dirname(fileURLToPath(import.meta.url));
  const devagentRoot = dirname(scriptDir);
  const scenariosDir = join(scriptDir, "live-validation", "scenarios");
  const scenarios = loadValidationScenarios(scenariosDir);

  if (options.listOnly) {
    for (const scenario of scenarios) {
      process.stdout.write(`${scenario.id}\t${scenario.suites.join(",")}\t${scenario.description}\n`);
    }
    return;
  }

  const selectedScenarios = filterScenarios(scenarios, options);
  if (selectedScenarios.length === 0) {
    throw new Error(`No scenarios selected for suite ${options.suite}.`);
  }

  const outputRoot = options.outputRoot
    ? join(options.outputRoot)
    : await mkdtemp(join(tmpdir(), "devagent-live-validation-"));
  await mkdir(outputRoot, { recursive: true });

  const reports = [];
  for (const scenario of selectedScenarios) {
    process.stdout.write(`Running ${scenario.id}...\n`);
    const report = await runValidationScenario(scenario, {
      devagentRoot,
      provider: options.provider,
      model: options.model,
      outputRoot,
    });
    reports.push(report);
    process.stdout.write(`  -> ${report.status}\n`);
    if (report.status !== "passed") {
      break;
    }
  }

  const summary = summarizeScenarioReports(reports, {
    provider: options.provider,
    model: options.model,
    suite: options.scenarioId ? "scenario" : options.suite,
  });
  await writeFile(join(outputRoot, "summary.json"), JSON.stringify(summary, null, 2));
  await writeFile(join(outputRoot, "summary.md"), renderSummaryMarkdown(summary));

  process.stdout.write(`Summary written to ${outputRoot}\n`);
  if (summary.failed > 0 || summary.blocked > 0) {
    process.exit(1);
  }
}

main().catch((error) => {
  process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
  process.exit(1);
});
