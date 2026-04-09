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
  const blocked = reports.filter((report) => report.status === "blocked").length;
  return {
    provider: context.provider,
    model: context.model,
    suite: context.suite,
    total: reports.length,
    passed,
    failed: reports.length - passed - blocked,
    blocked,
    reports: [...reports],
  };
}

function renderArtifactInventory(report: ValidationScenarioReport): string[] {
  if (report.artifactInventory.length === 0) {
    return ["- Artifact inventory: none captured"];
  }

  return [
    "- Artifact inventory:",
    ...report.artifactInventory.map((entry) => `  - ${entry.category}: ${entry.path}${entry.exists
      ? entry.sizeBytes !== undefined
        ? ` (${entry.sizeBytes} bytes)`
        : ""
      : " (missing)"}`
    ),
  ];
}

function renderStageReview(report: ValidationScenarioReport): string[] {
  if (!report.stageReview) {
    return [];
  }

  const lines = [
    `- Artifact-quality verdict: ${report.stageReview.verdict}`,
    `- Handoff ready: ${report.stageReview.handoffReady ? "yes" : "no"}`,
    `- Review summary: ${report.stageReview.summary}`,
    `- Human judgment: ${report.stageReview.humanJudgment}`,
  ];

  if (report.stageReview.checks.length > 0) {
    lines.push("- Review checks:");
    for (const check of report.stageReview.checks) {
      lines.push(`  - [${check.passed ? "x" : " "}] ${check.severity}: ${check.name} - ${check.message}`);
    }
  }

  if (report.stageReview.followUpIssues.length > 0) {
    lines.push("- Follow-up issues:");
    for (const issue of report.stageReview.followUpIssues) {
      lines.push(`  - ${issue}`);
    }
  }

  return lines;
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
    `- Blocked: ${summary.blocked}`,
    "",
  ];
  for (const report of summary.reports) {
    lines.push(`## ${report.scenarioId}`);
    lines.push(`- Status: ${report.status}`);
    lines.push(`- Repo: ${report.targetRepo}`);
    lines.push(`- Surface: ${report.surface}`);
    if (report.taskType) lines.push(`- Task type: ${report.taskType}`);
    lines.push(`- Workspace effect: expected ${report.repoMutation.expectedWorkspaceEffect}, observed ${report.repoMutation.observedChanges ? "changes" : "clean workspace"}`);
    lines.push(`- Workspace review: ${report.repoMutation.summary}`);
    if (report.failureClass) lines.push(`- Failure class: ${report.failureClass}`);
    if (report.failureMessage) lines.push(`- Failure: ${report.failureMessage}`);
    lines.push(...renderArtifactInventory(report));
    lines.push(...renderStageReview(report));
    lines.push("");
  }
  return lines.join("\n").trim() + "\n";
}

export function renderScenarioReviewMarkdown(
  report: ValidationScenarioReport,
): string {
  const lines = [
    `# Scenario Review: ${report.scenarioId}`,
    "",
    `- Status: ${report.status}`,
    `- Provider: ${report.provider}`,
    `- Model: ${report.model}`,
    `- Repo: ${report.targetRepo}`,
    `- Surface: ${report.surface}`,
    `- Task shape: ${report.taskShape}`,
    ...(report.taskType ? [`- Task type: ${report.taskType}`] : []),
    "",
    "## Evidence",
    "",
    ...(report.rawOutputs.requestPath ? [`- Request: ${report.rawOutputs.requestPath}`] : []),
    ...(report.rawOutputs.stdoutPath ? [`- Stdout: ${report.rawOutputs.stdoutPath}`] : []),
    ...(report.rawOutputs.stderrPath ? [`- Stderr: ${report.rawOutputs.stderrPath}`] : []),
    ...(report.rawOutputs.repoStatusPath ? [`- Repo status: ${report.rawOutputs.repoStatusPath}`] : []),
    ...(report.rawOutputs.repoDiffPath ? [`- Repo diff: ${report.rawOutputs.repoDiffPath}`] : []),
    ...(report.rawOutputs.eventsPath ? [`- Events: ${report.rawOutputs.eventsPath}`] : []),
    "",
    "## Artifact Inventory",
    "",
    ...renderArtifactInventory(report),
    "",
    "## Workspace Review",
    "",
    `- Expected workspace effect: ${report.repoMutation.expectedWorkspaceEffect}`,
    `- Observed changes: ${report.repoMutation.observedChanges ? "yes" : "no"}`,
    `- Verdict: ${report.repoMutation.passed ? "pass" : "fail"}`,
    `- Notes: ${report.repoMutation.summary}`,
  ];

  if (report.repoMutation.repoStatusExcerpt) {
    lines.push("");
    lines.push("```text");
    lines.push(report.repoMutation.repoStatusExcerpt);
    lines.push("```");
  }

  if (report.stageReview) {
    lines.push("");
    lines.push("## Artifact Review");
    lines.push("");
    lines.push(`- Verdict: ${report.stageReview.verdict}`);
    lines.push(`- Handoff ready: ${report.stageReview.handoffReady ? "yes" : "no"}`);
    lines.push(`- Summary: ${report.stageReview.summary}`);
    lines.push("");
    lines.push("### Human Judgment");
    lines.push("");
    lines.push(report.stageReview.humanJudgment);
    lines.push("");
    if (report.stageReview.checks.length > 0) {
      lines.push("### Review Checks");
      lines.push("");
      for (const check of report.stageReview.checks) {
        lines.push(`- [${check.passed ? "x" : " "}] ${check.name} (${check.severity}): ${check.message}`);
      }
      lines.push("");
    }
    if (report.stageReview.followUpIssues.length > 0) {
      lines.push("### Follow-up Issues");
      lines.push("");
      for (const issue of report.stageReview.followUpIssues) {
        lines.push(`- ${issue}`);
      }
      lines.push("");
    }
  }

  return `${lines.join("\n").trim()}\n`;
}
