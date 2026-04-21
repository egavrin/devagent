import type { DoctorCheck, DoctorCheckStatus, DoctorReport } from "./doctor-types.js";

function statusIcon(status: DoctorCheckStatus): string {
  switch (status) {
    case "pass":
      return "✓";
    case "advisory":
      return "!";
    case "blocking":
      return "✗";
  }
}

function formatCheck(check: DoctorCheck): string {
  return check.detail
    ? `  ${statusIcon(check.status)} ${check.label}: ${check.detail}`
    : `  ${statusIcon(check.status)} ${check.label}`;
}

function appendBlockingIssues(lines: string[], report: DoctorReport): void {
  if (report.blockingIssues.length === 0) {
    return;
  }

  lines.push("Blocking issues:", "");
  for (const issue of report.blockingIssues) {
    lines.push(`  - ${issue.title}: ${issue.detail}`);
  }
  lines.push("", "What to do next:", "");
  for (const issue of report.blockingIssues) {
    lines.push(`  ${issue.title}:`);
    for (const step of issue.nextSteps) {
      lines.push(`    ${step}`);
    }
  }
  lines.push("");
}

function appendEffectiveConfig(lines: string[], report: DoctorReport): void {
  lines.push("Effective config:", "");
  lines.push(`  Provider: ${report.effectiveConfig.provider} (${report.effectiveConfig.providerSource})`);
  lines.push(`  Model: ${report.effectiveConfig.model} (${report.effectiveConfig.modelSource})`);
  lines.push(`  Credential: ${report.effectiveConfig.credentialSource}`);
  if (report.effectiveConfig.modelProviders && report.effectiveConfig.modelProviders.length > 0) {
    lines.push(`  Registered providers: ${report.effectiveConfig.modelProviders.join(", ")}`);
  }
  lines.push("");
}

function appendProviderStatuses(lines: string[], report: DoctorReport): void {
  lines.push("  Available providers:");
  for (const provider of report.providerStatuses) {
    const status = provider.hasCredential ? "✓" : "·";
    const active = provider.active ? " (active)" : "";
    lines.push(`    ${status} ${provider.id}${active} — ${provider.hint}`);
  }
  lines.push("");
}

function appendLspStatuses(lines: string[], report: DoctorReport): void {
  lines.push("  LSP servers:");
  for (const lsp of report.lspStatuses) {
    const status = lsp.found ? "✓" : "·";
    const install = lsp.found ? "" : ` — install: ${lsp.install}`;
    lines.push(`    ${status} ${lsp.label}${install}`);
  }
  if (!report.lspStatuses.some((lsp) => lsp.found)) {
    lines.push("    (none found — run 'devagent install-lsp' to install)");
  }
}

function appendChecks(lines: string[], report: DoctorReport): void {
  lines.push("Checks:", "");
  lines.push(formatCheck(report.runtimeCheck));
  lines.push(formatCheck(report.gitCheck));
  lines.push(formatCheck(report.configCheck));
  lines.push(formatCheck(report.providerCheck));
  lines.push("");
  appendProviderStatuses(lines, report);
  lines.push(formatCheck(report.modelRegistryCheck));
  lines.push(formatCheck(report.modelCheck));
  lines.push(formatCheck(report.providerModelCheck));
  appendLspStatuses(lines, report);
  lines.push("");
  lines.push(formatCheck(report.platformCheck));
  lines.push("");
}

function formatSummary(status: DoctorReport["summaryStatus"]): string {
  switch (status) {
    case "blocking":
      return "Blocking issues found.";
    case "advisory":
      return "Checks passed with advisories.";
    case "pass":
      return "All checks passed.";
  }
}

export function renderDoctorReport(report: DoctorReport): string {
  const lines: string[] = [`devagent v${report.version}`, ""];
  appendBlockingIssues(lines, report);
  appendEffectiveConfig(lines, report);
  appendChecks(lines, report);
  lines.push(formatSummary(report.summaryStatus));

  return lines.join("\n");
}
