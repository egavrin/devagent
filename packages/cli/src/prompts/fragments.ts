import type {
  AgentType,
  ReasoningEffort,
  TaskMode,
  ToolSpec,
} from "@devagent/runtime";

interface RootPromptCapabilities {
  readonly hasDelegate: boolean;
  readonly hasRunCommand: boolean;
  readonly hasExecuteToolScript: boolean;
  readonly hasReadonlyFileTools: boolean;
  readonly hasMutatingTools: boolean;
}

const DEFAULT_ROOT_CAPABILITIES: RootPromptCapabilities = {
  hasDelegate: true,
  hasRunCommand: true,
  hasExecuteToolScript: true,
  hasReadonlyFileTools: true,
  hasMutatingTools: true,
};

const ROLE_ORDER: readonly AgentType[] = [
  "explore" as AgentType,
  "reviewer" as AgentType,
  "architect" as AgentType,
  "general" as AgentType,
];

function buildConstraintResolutionFragment(capabilities: RootPromptCapabilities): string {
  const inspectionModes = ["read-only file tools"];
  if (capabilities.hasDelegate) inspectionModes.push("readonly delegates");
  if (capabilities.hasRunCommand) inspectionModes.push("readonly shell search");

  return [
    "## Constraint Resolution",
    "",
    "When prompt rules appear to pull in different directions, resolve them in this order:",
    "1. explicit user constraints",
    "2. actual available tools and permissions",
    "3. the playbook that matches the task shape",
    "4. optimization guidance such as batching or latency reduction",
    "",
    `\"Read-only\" means no repo mutation. It still allows ${inspectionModes.join(", ")} when those are the available ways to inspect the requested target.`,
  ].join("\n");
}

function buildInvestigationPlaybookFragment(capabilities: RootPromptCapabilities): string {
  const lines = [
    "## Investigation Playbook",
    "",
    "For research, comparison, contradiction analysis, or evidence gathering:",
    "- Create an evidence-lane plan before searching.",
    "- Default to 2-4 narrow lanes, not broad phase steps.",
    "- Use this default lane pattern unless the prompt suggests a better split:",
    "  docs/spec claims; frontend/compile-time behavior; runtime/tests behavior; synthesis only after evidence exists.",
  ];

  if (capabilities.hasDelegate) {
    lines.push(
      "- If lanes are independent, emit multiple `explore` delegates in the same turn instead of serializing the investigation yourself.",
    );
  } else {
    lines.push(
      "- If delegation is unavailable, keep the same lane plan but execute the lanes locally with focused readonly search.",
    );
  }

  lines.push(
    "- Avoid serial phase plans like `locate -> inspect -> compare -> summarize` for broad investigations.",
    "- Parent-agent synthesis happens after lane evidence exists; do not let one lane drift into the whole investigation.",
  );

  return lines.join("\n");
}

function buildImplementationPlaybookFragment(): string {
  return [
    "## Implementation Playbook",
    "",
    "For implementation-heavy tasks:",
    "- Localize the files and functions first before editing.",
    "- Use `architect` only when there is real multi-file design uncertainty.",
    "- Use `reviewer` after code changes for a clean correctness pass when review is available.",
    "- Prefer direct local progress over speculative planning once the implementation path is clear.",
  ].join("\n");
}

function buildEditingFragment(): string {
  return [
    "## Editing",
    "",
    "- Always `read_file` before `replace_in_file`.",
    "- The `search` block must match the current file contents exactly.",
    "- Re-read a file after each successful edit before attempting another edit in the same file.",
    "- After `write_file`, immediately verify the new file contents and run a relevant syntax, build, or test check.",
  ].join("\n");
}

function buildDelegationFragment(): string {
  return [
    "## Delegation and Decomposition",
    "",
    "If `delegate` is available, use it for specialized sub-work:",
    "- `explore` for codebase search, repo discovery, and evidence lanes.",
    "- `reviewer` for post-change correctness and regression review.",
    "- `architect` for multi-file design uncertainty.",
    "- `general` for isolated implementation subtasks.",
    "",
    "Execution rules:",
    "- Multiple codebase searches or repo discovery -> `explore`.",
    "- Post-change correctness or regression checking -> `reviewer`.",
    "- Multi-file design before implementation -> `architect`.",
    "- If one of these patterns applies and the answer is not already in context, delegate before proceeding.",
    "",
    "For broad investigations:",
    "- Decompose into narrow evidence lanes before local searching begins.",
    "- Do not send one umbrella `explore` delegate when the prompt already names distinct evidence areas.",
    "- If the first delegate returns `partial`, prefer narrower follow-up delegates over switching the parent into broad direct exploration.",
    "",
    "Examples:",
    "- Broad cross-repo contradiction analysis: emit one `explore` delegate for docs/spec, one for frontend/compile-time handling, and one for runtime/tests, then synthesize locally.",
    "- Narrow lookup like `where is X lowered?`: answer directly with one focused search, or at most one `explore` delegate if the location is not already obvious.",
  ].join("\n");
}

function buildCrossRepoFragment(): string {
  return [
    "## Cross-Repo Search",
    "",
    "- `find_files`, `search_files`, and `read_file` are bounded to the current repo root.",
    "- One path-guard failure on a `../...` target is enough to pivot; do not keep retrying repo-bounded tools on unreachable paths.",
    "- For sibling repos, use readonly shell search with targeted `run_command` calls (`rg`, `sed`, focused `git` or build commands).",
    "- If broad scans keep returning empty or noisy results, narrow the scope or delegate to `explore`; do not keep scanning blindly.",
    "- Example: for `../arkcompiler_ets_frontend` and neighboring `arkcompiler_*` repos, pivot to targeted shell search instead of repeating repo-bounded tool calls.",
  ].join("\n");
}

function buildShellOperationsFragment(): string {
  return [
    "## Shell Operations",
    "",
    "- Use `run_command` for builds, tests, linting, and other real shell operations.",
    "- Prefer targeted verification commands first, then broaden if needed.",
    "- Prefer `rg` and `rg --files` for shell-based search.",
    "- Use non-interactive commands only.",
    "- Inspect the earliest stderr failure first; later errors are often cascading.",
  ].join("\n");
}

function buildBatchingFragment(capabilities: RootPromptCapabilities): string {
  const lines = [
    "## Readonly Batching",
    "",
    "Use `execute_tool_script` when you can plan multiple readonly operations upfront.",
    "- Good fit: multiple related `search_files` calls, or `find_files` plus focused `read_file` follow-ups.",
    "- Use it after lane selection. Do not use broad reconnaissance batches as a substitute for evidence-lane decomposition.",
  ];

  if (capabilities.hasDelegate) {
    lines.push(
      "- For broad research tasks, lane decomposition via `explore` delegates takes priority over local batching.",
    );
  }

  lines.push(
    "- If a script fails, break the failed steps into direct tool calls instead of retrying the same script.",
  );

  return lines.join("\n");
}

function buildModelPolicyFragment(
  agentModelOverrides?: Partial<Record<AgentType, string>>,
  agentReasoningOverrides?: Partial<Record<AgentType, ReasoningEffort>>,
): string | null {
  const lines: string[] = [];

  for (const agentType of ROLE_ORDER) {
    const model = agentModelOverrides?.[agentType];
    const reasoning = agentReasoningOverrides?.[agentType];
    if (!model && !reasoning) continue;

    const parts = [agentType as string];
    if (model) parts.push(`model: ${model}`);
    if (reasoning) parts.push(`reasoning: ${reasoning}`);
    lines.push(`- ${parts.join(" | ")}`);
  }

  if (lines.length === 0) return null;

  return [
    "## Subagent Model Policy",
    "",
    "Use these configured defaults when choosing delegate types:",
    ...lines,
  ].join("\n");
}

export function deriveRootPromptCapabilities(
  availableTools?: ReadonlyArray<Pick<ToolSpec, "name" | "category">>,
): RootPromptCapabilities {
  if (!availableTools || availableTools.length === 0) {
    return DEFAULT_ROOT_CAPABILITIES;
  }

  return {
    hasDelegate: availableTools.some((tool) => tool.name === "delegate"),
    hasRunCommand: availableTools.some((tool) => tool.name === "run_command"),
    hasExecuteToolScript: availableTools.some((tool) => tool.name === "execute_tool_script"),
    hasReadonlyFileTools: availableTools.some((tool) =>
      tool.name === "read_file" || tool.name === "find_files" || tool.name === "search_files"
    ),
    hasMutatingTools: availableTools.some((tool) => tool.category === "mutating"),
  };
}

export function buildRootPromptFragments(opts: {
  readonly mode: TaskMode;
  readonly capabilities: RootPromptCapabilities;
  readonly agentModelOverrides?: Partial<Record<AgentType, string>>;
  readonly agentReasoningOverrides?: Partial<Record<AgentType, ReasoningEffort>>;
}): string[] {
  const sections = [buildConstraintResolutionFragment(opts.capabilities)];

  if (opts.capabilities.hasReadonlyFileTools) {
    sections.push(buildInvestigationPlaybookFragment(opts.capabilities));
  }
  if (opts.mode === "act" && opts.capabilities.hasMutatingTools) {
    sections.push(buildEditingFragment());
    sections.push(buildImplementationPlaybookFragment());
  }
  if (opts.capabilities.hasDelegate) {
    sections.push(buildDelegationFragment());
  }
  if (opts.capabilities.hasRunCommand) {
    sections.push(buildShellOperationsFragment());
    sections.push(buildCrossRepoFragment());
  }
  if (opts.capabilities.hasExecuteToolScript) {
    sections.push(buildBatchingFragment(opts.capabilities));
  }

  const modelPolicy = buildModelPolicyFragment(
    opts.agentModelOverrides,
    opts.agentReasoningOverrides,
  );
  if (modelPolicy) {
    sections.push(modelPolicy);
  }

  return sections;
}
