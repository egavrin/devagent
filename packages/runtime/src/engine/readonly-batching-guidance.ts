export const READONLY_BATCHING_SECTION_TITLE = "Readonly Batching";

export const READONLY_BATCHING_SCRIPT_EXAMPLE = [
  'const paths = ["src/a.ts", "src/b.ts", "tests/a.test.ts"];',
  "const reads = await Promise.all(paths.map((path) => tools.read_file({ path })));",
  "const failures = reads.map((result, index) => result.success ? null : { path: paths[index], error: result.error }).filter(Boolean);",
  "const mismatches = reads.flatMap((result, index) => result.success && !result.output.includes('expected') ? [{ path: paths[index], issue: 'missing expected marker' }] : []);",
  "const conclusion = failures.length || mismatches.length ? 'issues found' : 'consistent';",
  "print(JSON.stringify({ conclusion, failures, mismatches }));",
].join(" ");

const CORE_BATCHING_LINES = [
  "Default to `execute_tool_script` as the first inspection tool when a narrowed task likely needs 3+ readonly calls and the file set is known. If files are only conceptually known, use at most one targeted discovery call, then batch the reads.",
  "- Prime fit: known-path multi-file audits where you can group `read_file` calls, check `result.success`, inspect `result.output`, and synthesize the answer in code.",
  "- Prime fit: verification prompts such as \"verify/disprove whether X leaks\", \"compare implementation, schema, and tests\", or \"check prompt consistency\" across a small known file set.",
  "- Good fit: implementation/schema/test/prompt-consistency/security-leakage checks where the files are named or easy to enumerate.",
  "- Also good: multiple related `search_files` calls, or `find_files` plus focused `read_file` follow-ups that can be filtered in code.",
  "- Do not spend separate turns on serial `read_file` calls when the file set is already known; batch them and print one compact conclusion.",
  "- Write one TypeScript script that calls readonly tools through `tools.*` and filters or aggregates results in code.",
  "- Example: `" + READONLY_BATCHING_SCRIPT_EXAMPLE + "`",
  "- Tool calls return `ToolResult` objects; inspect `result.output` for text content. There is no `result.content` field.",
  "- `read_file` output may include line numbers or section labels; do not blindly `JSON.parse(result.output)`. Extract the JSON body first, or compare schema text without parsing.",
  "- Print only synthesized findings, counts, paths, or summaries to final stdout. Do not print raw file contents, broad regex line-hit dumps, or raw diffs unless the user asked for them.",
  "- After a successful `execute_tool_script`, answer from its stdout. Do not follow with serial `read_file`/`search_files` on the same evidence unless the script failed or stdout explicitly says evidence is missing.",
  "- For DevAgent prompt/tool-script audits, common targets include `packages/cli/src/prompts/fragments.ts`, `packages/runtime/src/engine/agent-prompt.ts`, `packages/runtime/src/engine/prompts/*.md`, `packages/runtime/src/engine/tool-script-tool.ts`, `packages/runtime/src/engine/stagnation-detector.ts`, `packages/runtime/src/engine/tool-script.ts`, and `packages/runtime/src/engine/tool-script.test.ts`.",
  "- If a script fails, debug the failed script with direct readonly tool calls instead of retrying the same script.",
] as const;

export interface ReadonlyBatchingGuidanceOptions {
  readonly includeRootGuardrails?: boolean;
  readonly includeDelegationGuardrail?: boolean;
}

export function getReadonlyBatchingCoreLines(): readonly string[] {
  return CORE_BATCHING_LINES;
}

export function formatReadonlyBatchingGuidance(options: ReadonlyBatchingGuidanceOptions = {}): string {
  const lines = [
    `## ${READONLY_BATCHING_SECTION_TITLE}`,
    "",
    ...CORE_BATCHING_LINES,
  ];

  if (options.includeRootGuardrails) {
    lines.push(
      "- Lane selection applies to broad unknown-scope research. For already narrowed prompt/schema/test audits, batch after the file set is known or after one targeted discovery call.",
      "- Do not use broad reconnaissance batches as a substitute for evidence-lane decomposition.",
    );
  }

  if (options.includeDelegationGuardrail) {
    lines.push(
      "- For broad research tasks, lane decomposition via `explore` delegates takes priority over local batching.",
    );
  }

  return lines.join("\n");
}

export function formatToolScriptDescription(): string {
  return [
    "Execute a TypeScript program that calls multiple readonly tools locally and returns only final synthesized stdout.",
    "Default to this as the first inspection tool for narrowed tasks needing 3+ readonly calls when files are known; if files are only conceptually known, use one targeted discovery call, then batch.",
    "Good fits include known-path multi-file audits, grouped read_file checks, implementation/schema/test comparisons, prompt-consistency checks, and security-leakage verification.",
    "Use direct readonly tools instead for one-off lookups, broad unknown-scope reconnaissance, or debugging a failed script.",
    `Valid example: ${READONLY_BATCHING_SCRIPT_EXAMPLE}`,
    "Check result.success, inspect result.output, call print(...), and avoid raw file dumps or broad regex line-hit dumps.",
    "After a successful script, answer from its stdout instead of serial-reading the same evidence again.",
    "Only readonly tools are exposed through the tools object; imports, shell, filesystem, network, process, and recursive execute_tool_script calls are unavailable.",
  ].join(" ");
}

export function formatToolScriptSchemaDescription(): string {
  return [
    "TypeScript code to run in a restricted child process.",
    "Use this for narrowed audits that need 3+ readonly calls, especially known-path multi-file read_file batches. If the files are conceptually named but not exact paths, make one targeted discovery call first, then batch.",
    "Call readonly tools with await tools.read_file({ path }), tools.search_files({...}), tools.find_files({...}), tools.git_status({}), or tools.git_diff({...}).",
    "Tool calls return ToolResult objects; check result.success and inspect result.output, not result.content.",
    "read_file output may include line numbers or section labels; do not blindly JSON.parse(result.output). Extract a JSON body first, or compare schema text without parsing.",
    `Compact example: ${READONLY_BATCHING_SCRIPT_EXAMPLE}`,
    "Call print(...) with only the final synthesized answer; raw intermediate tool outputs are not returned to the model. Do not print raw file contents or broad line-hit dumps. After success, answer from stdout instead of re-reading the same files.",
  ].join(" ");
}
