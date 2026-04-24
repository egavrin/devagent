import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";

import type { AdoptionPrompt } from "./tool-script-adoption.js";

function filler(prefix: string, count = 120): string[] {
  return Array.from({ length: count }, (_, index) =>
    `${prefix} background material ${index + 1}: neutral validation padding with no audit keywords.`
  );
}

export async function createFixture(repoDir: string): Promise<void> {
  await Promise.all([
    mkdir(join(repoDir, "docs"), { recursive: true }),
    mkdir(join(repoDir, "src"), { recursive: true }),
    mkdir(join(repoDir, "tests"), { recursive: true }),
    mkdir(join(repoDir, "schema"), { recursive: true }),
  ]);
  await writePromptFixtures(repoDir);
  await writeImplementationFixtures(repoDir);
  await writeSchemaAndTestFixtures(repoDir);
}

async function writePromptFixtures(repoDir: string): Promise<void> {
  const content = [
    "## Readonly Batching",
    "Default to execute_tool_script for narrowed 3+ readonly call audits.",
    "Print only synthesized findings.",
    "After a successful script, answer from stdout without serial re-reading.",
    ...filler("shared prompt"),
  ].join("\n");
  await Promise.all([
    writeFile(join(repoDir, "docs", "root-prompt.md"), content),
    writeFile(join(repoDir, "docs", "child-prompt.md"), content),
    writeFile(join(repoDir, "docs", "tool-prompt.md"), content),
  ]);
}

async function writeImplementationFixtures(repoDir: string): Promise<void> {
  await Promise.all([
    writeFile(join(repoDir, "src", "tool-script.ts"), [
      "export async function runScript() {",
      "  // Inner tool outputs stay inside the script and only final stdout is returned.",
      "  return { hidesInnerOutputs: true, sandbox: 'readonly-vm' };",
      "}",
      ...filler("implementation"),
    ].join("\n")),
    writeFile(join(repoDir, "src", "tool-script-tool.ts"), [
      "export const schema = {",
      "  required: ['script'],",
      "  properties: { script: { type: 'string' }, timeout_ms: { type: 'number' }, max_output_chars: { type: 'number' } },",
      "  additionalProperties: false,",
      "};",
      "export const description = 'Use result.output, call print, and return final synthesized stdout.';",
      ...filler("implementation"),
    ].join("\n")),
    writeFile(join(repoDir, "src", "stagnation-detector.ts"), [
      "export const recovery = {",
      "  execute_tool_script: 'Fix the TypeScript script, use tools.read_file, inspect result.output, and print only the final synthesized answer.',",
      "  oldDsl: 'steps array DSL is rejected; use script instead',",
      "};",
      ...filler("implementation"),
    ].join("\n")),
  ]);
}

async function writeSchemaAndTestFixtures(repoDir: string): Promise<void> {
  await Promise.all([
    writeFile(join(repoDir, "schema", "tool-schema.json"), JSON.stringify({
      required: ["script"],
      properties: {
        script: { type: "string" },
        timeout_ms: { type: "number" },
        max_output_chars: { type: "number" },
      },
      additionalProperties: false,
    }, null, 2)),
    writeFile(join(repoDir, "tests", "tool-script.test.ts"), [
      "it('records only final stdout', () => expect('raw intermediate output').not.toContain('model history'));",
      "it('rejects steps', () => expect({ steps: [] }).toBeDefined());",
      "it('blocks shell and filesystem', () => expect('readonly-only').toBeTruthy());",
      ...filler("test padding"),
    ].join("\n")),
  ]);
}

export function buildPrompts(): AdoptionPrompt[] {
  const natural: AdoptionPrompt[] = [
    {
      label: "prompt-consistency",
      kind: "natural",
      prompt: "Compare docs/root-prompt.md, docs/child-prompt.md, and docs/tool-prompt.md for readonly batching guidance. Return only mismatches.",
    },
    {
      label: "leakage-verification",
      kind: "natural",
      prompt: "Verify or disprove whether the tool script runner leaks raw intermediate output by inspecting src/tool-script.ts, src/tool-script-tool.ts, and tests/tool-script.test.ts. Return only the evidence-backed conclusion.",
    },
    {
      label: "schema-test-audit",
      kind: "natural",
      prompt: "Audit src/tool-script-tool.ts, schema/tool-schema.json, and tests/tool-script.test.ts for execute_tool_script schema consistency. Return only inconsistencies. If reading JSON through read_file, account for line prefixes instead of blindly JSON.parse(result.output).",
    },
    {
      label: "recovery-hints",
      kind: "natural",
      prompt: "Review src/tool-script-tool.ts, src/stagnation-detector.ts, and tests/tool-script.test.ts for execute_tool_script recovery-hint consistency. Return only disagreements or stale old-DSL guidance.",
    },
  ];
  const explicit = natural.map((entry) => ({
    ...entry,
    label: `explicit-${entry.label}`,
    kind: "explicit" as const,
    prompt: `Use execute_tool_script as the first inspection tool. In the script, directly read the file paths named in the request and print only a compact synthesized answer. If the script succeeds, stop and answer from its stdout; do not call find_files, read_file, or search_files outside execute_tool_script. ${entry.prompt}`,
  }));
  return [...natural, ...explicit];
}
