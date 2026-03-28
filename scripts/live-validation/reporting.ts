import type {
  AggregateValidationSummary,
  AssertionResult,
  ValidationAssertion,
  ValidationScenarioReport,
} from "./types";

const ANSI_PATTERN = /\u001B\[[0-9;]*m/g;

function stripAnsi(value: string): string {
  return value.replace(ANSI_PATTERN, "");
}

function lookupSourceText(
  assertion: ValidationAssertion,
  inputs: {
    stdout: string;
    stderr: string;
    repoDiff: string;
    repoStatus: string;
    events: string;
    artifacts: Map<string, string>;
  },
): string {
  switch (assertion.source) {
    case "stdout":
      return stripAnsi(inputs.stdout);
    case "stderr":
      return stripAnsi(inputs.stderr);
    case "repoDiff":
      return inputs.repoDiff;
    case "repoStatus":
      return inputs.repoStatus;
    case "events":
      return inputs.events;
    case "artifact":
      return inputs.artifacts.get(assertion.path ?? "") ?? "";
  }
}

export function evaluateAssertions(
  assertions: ReadonlyArray<ValidationAssertion>,
  inputs: {
    stdout: string;
    stderr: string;
    repoDiff: string;
    repoStatus: string;
    events: string;
    artifacts: Map<string, string>;
  },
): { passed: boolean; results: AssertionResult[] } {
  const results = assertions.map((assertion) => {
    const text = lookupSourceText(assertion, inputs);
    if (assertion.type === "contains") {
      const passed = text.includes(assertion.value);
      return {
        type: assertion.type,
        source: assertion.source,
        passed,
        message: passed
          ? `Found expected text in ${assertion.source}.`
          : `Missing expected text "${assertion.value}" in ${assertion.source}.`,
      };
    }

    const regex = new RegExp(assertion.pattern, "i");
    const passed = regex.test(text);
    return {
      type: assertion.type,
      source: assertion.source,
      passed,
      message: passed
        ? `Pattern matched in ${assertion.source}.`
        : `Pattern "${assertion.pattern}" did not match ${assertion.source}.`,
    };
  });

  return {
    passed: results.every((result) => result.passed),
    results,
  };
}

export function summarizeScenarioReports(
  reports: ReadonlyArray<ValidationScenarioReport>,
  context: {
    provider: string;
    model: string;
    suite: AggregateValidationSummary["suite"];
  },
): AggregateValidationSummary {
  const passed = reports.filter((report) => report.status === "passed").length;
  return {
    provider: context.provider,
    model: context.model,
    suite: context.suite,
    total: reports.length,
    passed,
    failed: reports.length - passed,
    reports: [...reports],
  };
}

export function renderSummaryMarkdown(
  summary: AggregateValidationSummary,
): string {
  const lines = [
    "# Live Validation Summary",
    "",
    `- Suite: ${summary.suite}`,
    `- Provider: ${summary.provider}`,
    `- Model: ${summary.model}`,
    `- Total: ${summary.total}`,
    `- Passed: ${summary.passed}`,
    `- Failed: ${summary.failed}`,
    "",
  ];
  for (const report of summary.reports) {
    lines.push(`## ${report.scenarioId}`);
    lines.push(`- Status: ${report.status}`);
    lines.push(`- Repo: ${report.targetRepo}`);
    lines.push(`- Surface: ${report.surface}`);
    if (report.failureClass) lines.push(`- Failure class: ${report.failureClass}`);
    if (report.failureMessage) lines.push(`- Failure: ${report.failureMessage}`);
    lines.push("");
  }
  return lines.join("\n").trim() + "\n";
}
