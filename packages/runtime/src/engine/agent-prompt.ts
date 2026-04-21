import * as fs from "node:fs";
import { join } from "node:path";

import { formatBriefing } from "./briefing.js";
import type { TurnBriefing } from "./briefing.js";
import type { AgentType, SkillRegistry, ToolSpec } from "../core/index.js";
import { AgentType as AgentTypeEnum, LRUCache } from "../core/index.js";
import {
  formatSkillMatchLine,
  formatSkillPromptGuidance,
} from "../core/skills/prompt-format.js";
import type { DeferredToolStub } from "../tools/index.js";
import { PROMPT_AGENT_COMMON } from "./prompts/embedded.js";

const TOTAL_MAX_CHARS = 32 * 1024;

/** Memoization cache for assembled prompts. Keyed by hash of inputs. */
const promptCache = new LRUCache<string, string>(10);

/** Clear the prompt assembly cache. Call after compaction or tool resolution. */
export function clearPromptCache(): void {
  promptCache.clear();
}

interface InstructionFileSpec {
  readonly filename: string;
  readonly scope: string;
  readonly priority: number;
}

interface FoundInstruction extends InstructionFileSpec {
  readonly content: string;
}

const INSTRUCTION_FILES: readonly InstructionFileSpec[] = [
  {
    filename: ".devagent/ai_agent_instructions.md",
    scope: "DevAgent project rules",
    priority: 1.0,
  },
  {
    filename: ".devagent/instructions.md",
    scope: "DevAgent project rules (legacy)",
    priority: 0.9,
  },
  {
    filename: "AGENTS.md",
    scope: "Agent instructions (Codex-compatible)",
    priority: 0.8,
  },
  {
    filename: "CLAUDE.md",
    scope: "Agent instructions (Claude Code-compatible)",
    priority: 0.6,
  },
];

export interface AssembleAgentSystemPromptOptions {
  readonly agentType?: AgentType;
  readonly repoRoot: string;
  readonly rolePrompt: string;
  readonly availableTools?: ReadonlyArray<Pick<ToolSpec, "name" | "category">>;
  /** Deferred tools to list as available-on-demand in the prompt. */
  readonly deferredTools?: ReadonlyArray<DeferredToolStub>;
  readonly approvalMode?: string;
  readonly providerLabel?: string;
  readonly skills?: SkillRegistry;
  readonly briefing?: TurnBriefing;
  readonly projectInstructions?: string | null;
}

interface AgentPromptCapabilities {
  readonly hasReadonlySearch: boolean;
  readonly hasMutatingTools: boolean;
  readonly hasRunCommand: boolean;
  readonly hasExecuteToolScript: boolean;
  readonly hasLspTools: boolean;
  readonly hasDelegate: boolean;
}

function loadCommonPrompt(): string {
  return PROMPT_AGENT_COMMON;
}

function deriveAgentPromptCapabilities(
  availableTools?: ReadonlyArray<Pick<ToolSpec, "name" | "category">>,
): AgentPromptCapabilities {
  if (!availableTools || availableTools.length === 0) {
    return {
      hasReadonlySearch: true,
      hasMutatingTools: false,
      hasRunCommand: false,
      hasExecuteToolScript: false,
      hasLspTools: false,
      hasDelegate: false,
    };
  }

  return {
    hasReadonlySearch: availableTools.some((tool) =>
      tool.name === "read_file" || tool.name === "find_files" || tool.name === "search_files"
    ),
    hasMutatingTools: availableTools.some((tool) => tool.category === "mutating"),
    hasRunCommand: availableTools.some((tool) => tool.name === "run_command"),
    hasExecuteToolScript: availableTools.some((tool) => tool.name === "execute_tool_script"),
    hasLspTools: availableTools.some((tool) =>
      tool.name === "diagnostics" ||
      tool.name === "symbols" ||
      tool.name === "definitions" ||
      tool.name === "references"
    ),
    hasDelegate: availableTools.some((tool) => tool.name === "delegate"),
  };
}

function buildSearchFragment(): string {
  return [
    "## Search Strategy",
    "",
    "When exploring an unfamiliar codebase:",
    "1. `find_files` with focused patterns to identify the relevant area.",
    "2. `search_files` with a scoped `file_pattern` to locate exact symbols or strings.",
    "3. `read_file` on only the relevant sections.",
    "",
    "Rules:",
    "- Prefer targeted `file_pattern` values over global scans.",
    "- Avoid speculative full-file reads when a focused read will answer the question.",
    "- Use exact canonical tool names.",
  ].join("\n");
}

function buildEditingFragment(): string {
  return [
    "## Editing",
    "",
    "- Always `read_file` before `replace_in_file`.",
    "- Copy the `search` block verbatim from the current file contents.",
    "- Re-read the file after each successful edit before making a second edit to the same file.",
    "- After `write_file`, immediately verify the file contents and run a targeted syntax, test, or build check.",
  ].join("\n");
}

function buildShellFragment(): string {
  return [
    "## Shell Commands",
    "",
    "- Use `run_command` for real shell operations such as tests, builds, or focused search that file tools cannot handle.",
    "- Prefer targeted commands first, then broaden only if needed.",
    "- Use non-interactive commands only.",
    "- Inspect the earliest stderr failure first; later errors are often cascading.",
  ].join("\n");
}

function buildLspFragment(): string {
  return [
    "## LSP Tools",
    "",
    "- `diagnostics` for compiler and language-server feedback after edits.",
    "- `symbols` for structural overview.",
    "- `definitions` and `references` to trace symbol ownership and usage.",
  ].join("\n");
}

function buildBatchingFragment(): string {
  return [
    "## Batched Readonly Calls",
    "",
    "- Use `execute_tool_script` only after you have already narrowed the search scope.",
    "- Batch independent readonly calls when it reduces round-trips.",
    "- If a script fails, break the failed steps into direct tool calls instead of retrying the same script.",
  ].join("\n");
}

function buildDelegationFragment(): string {
  return [
    "## Delegation",
    "",
    "- If `delegate` is available, use it only for independent subtasks that benefit from a separate context window.",
    "- Prefer multiple readonly `explore` or `reviewer` delegates in one turn when the subtasks are independent.",
    "- Do not assume mutating or planning delegates run in parallel unless the runtime explicitly allows it.",
  ].join("\n");
}

function buildAgentCapabilityFragments(
  agentType: AgentType | undefined,
  capabilities: AgentPromptCapabilities,
): string[] {
  const sections: string[] = [];

  if (capabilities.hasReadonlySearch && agentType !== AgentTypeEnum.EXPLORE) {
    sections.push(buildSearchFragment());
  }
  if (capabilities.hasMutatingTools) {
    sections.push(buildEditingFragment());
  }
  if (capabilities.hasRunCommand) {
    sections.push(buildShellFragment());
  }
  if (capabilities.hasLspTools) {
    sections.push(buildLspFragment());
  }
  if (capabilities.hasExecuteToolScript) {
    sections.push(buildBatchingFragment());
  }
  if (capabilities.hasDelegate) {
    sections.push(buildDelegationFragment());
  }

  return sections;
}

function allocateBudgets(
  files: ReadonlyArray<FoundInstruction>,
  totalBudget: number,
): number[] {
  if (files.length === 0 || totalBudget <= 0) return [];

  const minPerFile = Math.max(250, Math.floor(totalBudget / (files.length * 3)));
  const budgets = new Array<number>(files.length).fill(0);
  let remainingBudget = totalBudget;
  let remainingPriority = files.reduce((sum, file) => sum + file.priority, 0);

  for (let i = 0; i < files.length; i++) {
    const isLast = i === files.length - 1;
    if (isLast) {
      budgets[i] = Math.max(0, remainingBudget);
      break;
    }

    const filesLeft = files.length - i;
    const minForRest = minPerFile * (filesLeft - 1);
    const maxForCurrent = Math.max(minPerFile, remainingBudget - minForRest);
    const weightedShare = remainingPriority > 0
      ? Math.floor((remainingBudget * files[i]!.priority) / remainingPriority)
      : maxForCurrent;
    const budget = Math.max(minPerFile, Math.min(maxForCurrent, weightedShare));

    budgets[i] = budget;
    remainingBudget -= budget;
    remainingPriority -= files[i]!.priority;
  }

  return budgets;
}

function truncateContent(content: string, maxChars: number): string {
  if (content.length <= maxChars) return content;
  return `${content.slice(0, maxChars).trimEnd()}\n\n[...truncated]`;
}

export function loadAgentProjectInstructions(repoRoot: string): string | null {
  const found: FoundInstruction[] = [];

  for (const spec of INSTRUCTION_FILES) {
    const filePath = join(repoRoot, spec.filename);
    if (!fs.existsSync(filePath)) continue;

    const content = fs.readFileSync(filePath, "utf-8");
    if (content.trim().length === 0) continue;
    found.push({ ...spec, content });
  }

  if (found.length === 0) return null;

  const ordered = [...found].sort((a, b) => b.priority - a.priority);
  const budgets = allocateBudgets(ordered, TOTAL_MAX_CHARS);

  return ordered
    .map((entry, index) => {
      const text = truncateContent(entry.content, budgets[index] ?? TOTAL_MAX_CHARS);
      return `## Project Instructions (${entry.scope})\n\nSource: \`${entry.filename}\`\n\n${text}`;
    })
    .join("\n\n");
}

/** @deprecated Prompts are now embedded constants — cache reset is a no-op. */
export function __resetCommonPromptCacheForTesting(): void {
  // No-op: prompts are embedded constants, no filesystem cache to reset.
}

/** @deprecated Prompts are now embedded constants — always returns 0. */
export function __getCommonPromptReadCountForTesting(): number {
  return 0;
}

/** Format the deferred tools prompt section. Shared by CLI and engine prompt assemblers. */
export function formatDeferredToolsSection(
  tools: ReadonlyArray<{ name: string; description: string }>,
): string {
  const toolLines = tools.map((t) => `- ${t.name}: ${t.description}`);
  return (
    "## Additional Tools (available via tool_search)\n\n" +
    "The following tools are available but not loaded. Use `tool_search` with a keyword to activate them:\n" +
    toolLines.join("\n")
  );
}
export function assembleAgentSystemPrompt(
  options: AssembleAgentSystemPromptOptions,
): string {
  // Check memoization cache — skip date in key since it changes daily, not per-call
  const cacheKey = buildPromptCacheKey(options);
  const cached = promptCache.get(cacheKey);
  if (cached) return cached;

  const sections: string[] = [];
  const capabilities = deriveAgentPromptCapabilities(options.availableTools);
  const commonPrompt = loadCommonPrompt().replace(/\{\{repoRoot\}\}/g, options.repoRoot);

  sections.push(commonPrompt);
  sections.push(options.rolePrompt.replace(/\{\{repoRoot\}\}/g, options.repoRoot));
  sections.push(...buildAgentCapabilityFragments(options.agentType, capabilities));
  sections.push(formatEnvironmentSection(options));
  sections.push(...buildOptionalPromptSections(options));

  const result = sections.join("\n\n");
  promptCache.set(cacheKey, result);
  return result;
}

function formatEnvironmentSection(options: AssembleAgentSystemPromptOptions) {
  return `## Environment\n\n${buildEnvironmentLines(options).join("\n")}`;
}

function buildEnvironmentLines(options: AssembleAgentSystemPromptOptions) {
  return [
    `Working directory: ${options.repoRoot}`,
    `Date: ${new Date().toISOString().split("T")[0]}`,
    options.approvalMode ? `Safety mode: ${options.approvalMode}` : null,
    options.providerLabel ? `Provider: ${options.providerLabel}` : null,
  ].filter((line): line is string => line !== null);
}

function buildOptionalPromptSections(options: AssembleAgentSystemPromptOptions): string[] {
  return [
    options.projectInstructions ?? loadAgentProjectInstructions(options.repoRoot),
    formatAvailableSkillsSection(options),
    options.deferredTools?.length ? formatDeferredToolsSection(options.deferredTools) : null,
    options.briefing ? formatBriefingSection(options.briefing) : null,
  ].filter((section): section is string => Boolean(section));
}

function formatAvailableSkillsSection(options: AssembleAgentSystemPromptOptions): string | null {
  const skillList = options.skills?.list() ?? [];
  if (skillList.length === 0) return null;
  return (
    "## Available Skills\n\n" +
    skillList.map((skill) => formatSkillMatchLine(skill)).join("\n") +
    "\n\n" +
    formatSkillPromptGuidance()
  );
}

function formatBriefingSection(briefing: NonNullable<AssembleAgentSystemPromptOptions["briefing"]>) {
  return `## Session Context\n\nYou are continuing a conversation. Here is a summary of prior work:\n\n${formatBriefing(briefing)}`;
}

/** Build a cache key from prompt assembly inputs. */
function buildPromptCacheKey(options: AssembleAgentSystemPromptOptions): string {
  const toolNames = options.availableTools
    ? options.availableTools.map((t) => t.name).sort().join(",")
    : "";
  const deferredNames = options.deferredTools
    ? options.deferredTools.map((t) => t.name).sort().join(",")
    : "";
  const skillNames = options.skills
    ? options.skills
      .list()
      .map((s) =>
        JSON.stringify({
          name: s.name,
          description: s.description,
          triggers: s.triggers ?? [],
          paths: s.paths ?? [],
          examples: s.examples ?? [],
          source: s.source,
        }))
      .sort()
      .join(",")
    : "";
  // Hash project instructions content for stability
  const instrHash = options.projectInstructions
    ? simpleHash(options.projectInstructions)
    : "none";
  return [
    options.agentType ?? "default",
    options.repoRoot,
    toolNames,
    deferredNames,
    options.approvalMode ?? "",
    options.providerLabel ?? "",
    options.briefing?.turnNumber ?? 0,
    skillNames,
    instrHash,
  ].join("|");
}

/** Fast non-crypto hash for cache key generation. */
function simpleHash(str: string): string {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash |= 0; // Convert to 32-bit integer
  }
  return hash.toString(36);
}
