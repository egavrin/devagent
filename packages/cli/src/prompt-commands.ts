export type PromptCommandName = "review" | "simplify";
export type PromptCommandFocus = "correctness" | "performance" | "tests" | "types";
export type PromptCommandDelegatePreference = "auto" | "force" | "forbid";
export type PromptCommandVerificationPreference = "normal" | "skip";

export type PromptCommandTarget =
  | { readonly kind: "auto" }
  | { readonly kind: "staged" }
  | { readonly kind: "unstaged" }
  | { readonly kind: "last-commit" }
  | { readonly kind: "commit"; readonly ref: string };

export type ResolvedPromptCommandTarget = Exclude<PromptCommandTarget, { readonly kind: "auto" }>;

export interface PromptCommandSegment {
  readonly name: PromptCommandName;
  readonly body: string;
  readonly rawBody: string;
  readonly targetHint: PromptCommandTarget;
  readonly pathFilters: ReadonlyArray<string>;
  readonly focusAreas: ReadonlyArray<PromptCommandFocus>;
  readonly delegatePreference: PromptCommandDelegatePreference;
  readonly verificationPreference: PromptCommandVerificationPreference;
  readonly start: number;
  readonly end: number;
}

export interface ParsedPromptCommandQuery {
  readonly originalQuery: string;
  readonly leadingText: string;
  readonly segments: ReadonlyArray<PromptCommandSegment>;
}

export interface ResolvedPromptCommandSegment extends PromptCommandSegment {
  readonly target: ResolvedPromptCommandTarget;
}

export interface PreparedPromptCommandQuery {
  readonly parsed: ParsedPromptCommandQuery;
  readonly resolvedSegments: ReadonlyArray<ResolvedPromptCommandSegment>;
  readonly rewrittenQuery: string;
  readonly preloadedDiffs: ReadonlyArray<{
    readonly target: ResolvedPromptCommandTarget;
    readonly pathFilters: ReadonlyArray<string>;
    readonly content: string;
  }>;
  readonly finalTextValidator?: PromptCommandFinalTextValidator;
}

export interface PromptCommandFinalTextValidationResult {
  readonly valid: boolean;
  readonly retryMessage?: string;
}

export type PromptCommandFinalTextValidator = (
  candidate: string,
) => PromptCommandFinalTextValidationResult;

interface CommandMatch {
  readonly name: PromptCommandName;
  readonly start: number;
  readonly end: number;
}

interface DiffComplexity {
  readonly filesChanged: number;
  readonly changedLines: number;
}

const COMMANDS: readonly PromptCommandName[] = ["review", "simplify"];
const AUTO_TARGET: PromptCommandTarget = { kind: "auto" };
const SIMPLIFY_LOCAL_FILE_THRESHOLD = 1;
const SIMPLIFY_LOCAL_LINE_THRESHOLD = 40;
const REVIEW_BLOCKING_FINDING_PATTERN = /^(?:[-*]\s*)?\[(error|warning)\]\s+`?([^`\n]+?)`?(?::(\d+))?\s+[—-]\s+(.+)$/i;
const REVIEW_NON_BLOCKING_SUGGESTION_PATTERN = /^(?:[-*]\s+)(?:`?([^`\n]+?)`?(?::(\d+))?\s+[—-]\s+)?(.+)$/i;
const REVIEW_CODE_HEALTH_VERDICT_PATTERN = /\boverall:\s+.+code health\b/i;

function isBoundaryBefore(char: string | undefined): boolean {
  return !char || /\s|[([{'"`,;!?]/.test(char);
}

function isBoundaryAfter(char: string | undefined): boolean {
  return !char || /\s|[)\]}'"`,.;:!?]/.test(char);
}

function isAutoTarget(target: PromptCommandTarget): target is { readonly kind: "auto" } {
  return target.kind === "auto";
}

function looksLikePath(value: string): boolean {
  return (
    value.includes("/") ||
    value.startsWith(".") ||
    /\.[A-Za-z0-9]+$/.test(value) ||
    /^(packages|models|prompts|scripts|docs|README\.md|CLAUDE\.md|AGENTS\.md)\b/.test(value)
  );
}

function normalizePathCandidate(value: string): string | null {
  const trimmed = value.trim().replace(/^['"`]|['"`]$/g, "");
  if (!trimmed || !looksLikePath(trimmed)) {
    return null;
  }
  return trimmed.replace(/[),.;:!?]+$/g, "");
}

function extractPathFilters(text: string): string[] {
  const matches = new Set<string>();
  const capturePattern = /\b(?:only|in|under)\s+([A-Za-z0-9_./'"`-]+(?:\s*(?:,|and)\s+[A-Za-z0-9_./'"`-]+)*)/gi;
  for (const match of text.matchAll(capturePattern)) {
    const rawGroup = match[1];
    if (!rawGroup) continue;
    const candidates = rawGroup.split(/\s*(?:,|and)\s*/);
    for (const candidate of candidates) {
      const normalized = normalizePathCandidate(candidate);
      if (normalized) {
        matches.add(normalized);
      }
    }
  }
  return [...matches];
}

function extractFocusAreas(text: string): PromptCommandFocus[] {
  const focuses = new Set<PromptCommandFocus>();
  if (/\b(correctness|regression|bug|bugs)\b/i.test(text)) focuses.add("correctness");
  if (/\b(perf|performance|efficiency|hot path|concurrency)\b/i.test(text)) focuses.add("performance");
  if (/\b(test|tests|coverage|verification)\b/i.test(text)) focuses.add("tests");
  if (/\b(type|types|typing|type safety)\b/i.test(text)) focuses.add("types");
  return [...focuses];
}

function resolveDelegatePreference(text: string): PromptCommandDelegatePreference {
  if (/\b(no|without)\s+delegates?\b/i.test(text) || /\bstay local\b/i.test(text)) {
    return "forbid";
  }
  if (/\b(use|with|launch)\s+delegates?\b/i.test(text) || /\bparallel(?:ize)?\b/i.test(text)) {
    return "force";
  }
  return "auto";
}

function resolveVerificationPreference(text: string): PromptCommandVerificationPreference {
  return /\b(skip|without|no)\s+(tests?|verification)\b/i.test(text) ? "skip" : "normal";
}

function findCommandMatches(query: string): CommandMatch[] {
  const matches: CommandMatch[] = [];

  for (let index = 0; index < query.length; index++) {
    if (query[index] !== "/" || !isBoundaryBefore(query[index - 1])) {
      continue;
    }

    for (const command of COMMANDS) {
      if (!query.startsWith(command, index + 1)) {
        continue;
      }

      const end = index + 1 + command.length;
      if (!isBoundaryAfter(query[end])) {
        continue;
      }

      matches.push({ name: command, start: index, end });
      index = end - 1;
      break;
    }
  }

  return matches;
}

export function resolvePromptCommandTargetHint(text: string): PromptCommandTarget {
  if (/\bunstaged\b/i.test(text)) return { kind: "unstaged" };
  if (/\bstaged\b/i.test(text)) return { kind: "staged" };
  if (/\bfiles?\s+touched\s+by\s+last\s+commit\b/i.test(text)) return { kind: "last-commit" };
  if (/\blast(?:[-\s])commit\b/i.test(text)) return { kind: "last-commit" };
  const commitMatch = text.match(/\bcommit\s+([A-Za-z0-9._/-]+)\b/i);
  if (commitMatch?.[1]) {
    return { kind: "commit", ref: commitMatch[1] };
  }
  return AUTO_TARGET;
}

export function parsePromptCommandQuery(query: string): ParsedPromptCommandQuery | null {
  const matches = findCommandMatches(query);
  if (matches.length === 0) {
    return null;
  }

  const segments: PromptCommandSegment[] = [];
  for (let index = 0; index < matches.length; index++) {
    const match = matches[index]!;
    const next = matches[index + 1];
    const rawBody = query.slice(match.end, next?.start ?? query.length);
    const body = rawBody.trim();
    segments.push({
      name: match.name,
      rawBody,
      body,
      targetHint: resolvePromptCommandTargetHint(body),
      pathFilters: extractPathFilters(body),
      focusAreas: extractFocusAreas(body),
      delegatePreference: resolveDelegatePreference(body),
      verificationPreference: resolveVerificationPreference(body),
      start: match.start,
      end: next?.start ?? query.length,
    });
  }

  return {
    originalQuery: query,
    leadingText: query.slice(0, matches[0]!.start).trim(),
    segments,
  };
}

function humanizeTarget(target: ResolvedPromptCommandTarget): string {
  switch (target.kind) {
    case "unstaged":
      return "unstaged working tree diff";
    case "staged":
      return "staged diff";
    case "last-commit":
      return "last commit diff";
    case "commit":
      return `commit ${target.ref} diff`;
  }
}

function humanizePathFilters(pathFilters: ReadonlyArray<string>): string {
  return pathFilters.join(", ");
}

function describeSegmentFocus(segment: ResolvedPromptCommandSegment): string | null {
  const details: string[] = [];
  if (segment.pathFilters.length > 0) {
    details.push(`Path filters: ${humanizePathFilters(segment.pathFilters)}`);
  }
  if (segment.focusAreas.length > 0) {
    details.push(`Focus: ${segment.focusAreas.join(", ")}`);
  }
  if (segment.verificationPreference === "skip") {
    details.push("Verification preference: skip tests or verification commands");
  }
  return details.length > 0 ? details.join(" | ") : null;
}

function analyzeDiffComplexity(diff: string): DiffComplexity {
  const filesChanged = Math.max(1, (diff.match(/^diff --git /gm) ?? []).length);
  const changedLines = diff
    .split("\n")
    .filter((line) => (line.startsWith("+") || line.startsWith("-")) && !line.startsWith("+++") && !line.startsWith("---"))
    .length;
  return { filesChanged, changedLines };
}

function buildSimplifyDelegationInstruction(
  segment: ResolvedPromptCommandSegment,
  diff: string | null,
): string[] {
  if (segment.delegatePreference === "forbid") {
    return [
      "   Do not spawn reviewer delegates for this step unless you are blocked on missing evidence.",
      "   Keep the simplify pass local, skeptical, and deletion-first.",
    ];
  }

  if (segment.delegatePreference === "force") {
    return [
      "   Launch three readonly `reviewer` delegates for reuse, quality, and efficiency before editing.",
      "   Aggregate their findings, then use them to remove code aggressively in the main agent before refactoring what remains.",
    ];
  }

  if (diff) {
    const complexity = analyzeDiffComplexity(diff);
    if (
      complexity.filesChanged <= SIMPLIFY_LOCAL_FILE_THRESHOLD &&
      complexity.changedLines <= SIMPLIFY_LOCAL_LINE_THRESHOLD
    ) {
      return [
        `   Stay local for this step because the scope is small (${complexity.filesChanged} file, ${complexity.changedLines} changed lines).`,
        "   Only fan out if you uncover evidence that the touched code depends on a broader shared path or added unjustified surface nearby.",
      ];
    }
  }

  return [
    "   Launch three readonly `reviewer` delegates for reuse, quality, and efficiency before editing.",
    "   Aggregate their findings, then use them to remove code aggressively in the main agent before refactoring what remains.",
  ];
}

function buildVerificationInstruction(segment: ResolvedPromptCommandSegment): string {
  if (segment.verificationPreference === "skip") {
    return "   Skip verification commands unless the user later asks for them or the change becomes risky enough to require proof.";
  }
  return "   Run focused verification after meaningful edits and report what you did or did not run.";
}

function buildReviewDelegationInstruction(
  segment: ResolvedPromptCommandSegment,
): string[] {
  if (segment.delegatePreference === "forbid") {
    return [
      "   Do not spawn reviewer delegates for this step unless you need targeted independent evidence.",
    ];
  }

  return [
    "   Launch three readonly `reviewer` delegates before concluding.",
    "   Use distinct lanes for correctness/regressions, tests/contracts, and performance/fail-fast risks.",
    "   Aggregate the delegates' blocking issues, then produce one final review.",
  ];
}

function buildWorkflowStep(
  segment: ResolvedPromptCommandSegment,
  index: number,
  diff: string | null,
): string {
  const stepLines = [
    `${index + 1}. \`/${segment.name}\``,
    `   Scope: ${humanizeTarget(segment.target)}`,
    `   User focus: ${segment.body || "(none specified)"}`,
  ];

  const detailLine = describeSegmentFocus(segment);
  if (detailLine) {
    stepLines.push(`   ${detailLine}`);
  }

  if (segment.name === "review") {
    stepLines.push("   Requirements:");
    stepLines.push("   Invoke the `review` skill before starting this step.");
    stepLines.push(`   Review the pre-loaded ${humanizeTarget(segment.target)} first.`);
    stepLines.push(...buildReviewDelegationInstruction(segment));
    stepLines.push("   Classify issues into blocking defects vs non-blocking suggestions.");
    stepLines.push("   Reserve blocking findings for correctness, regressions, security, fail-fast violations, contract drift, serious missing coverage, or material performance risks.");
    stepLines.push("   Focus on blocking issues first. Keep non-blocking suggestions brief and only include them when they are genuinely worthwhile.");
    stepLines.push("   Put stylistic, readability, naming, and minor maintainability comments into Non-blocking Suggestions instead of blocking findings.");
    stepLines.push("   Do not block on optional polish or chase perfection.");
    stepLines.push("   Do not narrate your process, review steps, or delegate work.");
    stepLines.push("   Do not add any preamble, status update, or duplicate headings.");
    stepLines.push("   Start immediately with `Blocking Findings`.");
    stepLines.push("   Every blocking finding must include severity, file, and rationale.");
    stepLines.push("   Structure the final answer as: Blocking Findings, Non-blocking Suggestions, Open Questions / Assumptions, Short Summary.");
    stepLines.push("   In Short Summary, include an explicit code-health verdict such as `Overall: improves code health` or `Overall: does not improve code health`.");
  } else {
    stepLines.push("   Requirements:");
    stepLines.push("   Invoke the `simplify` skill before starting this step.");
    stepLines.push(`   Simplify the pre-loaded ${humanizeTarget(segment.target)} first.`);
    stepLines.push("   Cover reuse, code quality, efficiency or performance, and explicit minimization or deletion.");
    stepLines.push("   Challenge newly added surface area, not just implementation details.");
    stepLines.push("   Treat new files, exports, config keys, tests, and docs as removable candidates unless they are clearly justified.");
    stepLines.push("   Prefer collapsing feature scope over keeping a larger patch and only cleaning style or structure.");
    stepLines.push("   Aggregate delegate evidence, then perform a deletion-first synthesis pass before editing.");
    stepLines.push(...buildSimplifyDelegationInstruction(segment, diff));
    stepLines.push(buildVerificationInstruction(segment));
  }

  return stepLines.join("\n");
}

function makePreloadKey(
  target: ResolvedPromptCommandTarget,
  pathFilters: ReadonlyArray<string>,
): string {
  const targetKey = target.kind === "commit" ? `commit:${target.ref}` : target.kind;
  return `${targetKey}::${pathFilters.join("|")}`;
}

export function buildPromptCommandRewrite(
  parsed: ParsedPromptCommandQuery,
  resolvedSegments: ReadonlyArray<ResolvedPromptCommandSegment>,
  diffByScope: ReadonlyMap<string, string | null> = new Map(),
): string {
  const lines = [
    "The user used embedded prompt commands. Treat them as an ordered workflow and preserve their left-to-right order.",
    `Original query: ${parsed.originalQuery}`,
  ];

  if (parsed.leadingText) {
    lines.push(`Context before the first command: ${parsed.leadingText}`);
  }

  lines.push("If a referenced pre-loaded diff is unavailable, gather the same scope yourself with git tools.");
  lines.push("");
  lines.push("Workflow:");
  for (const [index, segment] of resolvedSegments.entries()) {
    lines.push(buildWorkflowStep(
      segment,
      index,
      diffByScope.get(makePreloadKey(segment.target, segment.pathFilters)) ?? null,
    ));
  }

  return lines.join("\n");
}

export function formatPreloadedDiffMessage(
  target: ResolvedPromptCommandTarget,
  diff: string,
  pathFilters: ReadonlyArray<string> = [],
): string {
  const scope = humanizeTarget(target);
  const suffix = pathFilters.length > 0 ? ` scoped to ${humanizePathFilters(pathFilters)}` : "";
  return `[Pre-loaded local ${scope}${suffix}]\n\n${diff}`;
}

function normalizeHeading(line: string): string {
  return line
    .trim()
    .replace(/^#+\s*/, "")
    .replace(/^\*\*|\*\*:?$/g, "")
    .replace(/:$/, "")
    .toLowerCase();
}

function collectSection(lines: ReadonlyArray<string>, startIndex: number): string[] {
  const section: string[] = [];
  for (let index = startIndex + 1; index < lines.length; index++) {
    const line = lines[index]!;
    const heading = normalizeHeading(line);
    if (
      heading === "blocking findings" ||
      heading === "non-blocking suggestions" ||
      heading === "findings" ||
      heading === "open questions / assumptions" ||
      heading === "short summary"
    ) {
      break;
    }
    section.push(line);
  }
  return section;
}

function extractReviewSections(candidate: string): {
  readonly blockingFindings: ReadonlyArray<string>;
  readonly nonBlockingSuggestions: ReadonlyArray<string>;
  readonly openQuestions: ReadonlyArray<string>;
  readonly summary: ReadonlyArray<string>;
} | null {
  const lines = candidate.split("\n");
  const headings: Array<{ readonly heading: string; readonly index: number }> = [];

  for (let index = 0; index < lines.length; index++) {
    const heading = normalizeHeading(lines[index]!);
    if (
      heading === "blocking findings" ||
      heading === "non-blocking suggestions" ||
      heading === "open questions / assumptions" ||
      heading === "short summary"
    ) {
      headings.push({ heading, index });
    }
  }

  const expected = [
    "blocking findings",
    "non-blocking suggestions",
    "open questions / assumptions",
    "short summary",
  ] as const;
  const firstNonEmptyLineIndex = lines.findIndex((line) => line.trim().length > 0);

  if (
    firstNonEmptyLineIndex !== headings[0]?.index ||
    headings.length !== expected.length ||
    headings.some((entry, index) => entry.heading !== expected[index])
  ) {
    return null;
  }

  return {
    blockingFindings: collectSection(lines, headings[0]!.index),
    nonBlockingSuggestions: collectSection(lines, headings[1]!.index),
    openQuestions: collectSection(lines, headings[2]!.index),
    summary: collectSection(lines, headings[3]!.index),
  };
}

function blockingFindingsSectionIsValid(lines: ReadonlyArray<string>): boolean {
  const contentLines = lines.map((line) => line.trim()).filter(Boolean);
  if (contentLines.length === 0) {
    return false;
  }
  if (contentLines.length === 1 && /\b(no issues found|no findings|none)\b/i.test(contentLines[0]!)) {
    return true;
  }

  return contentLines.every((line) => {
    const match = line.match(REVIEW_BLOCKING_FINDING_PATTERN);
    if (!match) return false;
    const file = match[2]?.trim() ?? "";
    const rationale = match[4]?.trim() ?? "";
    return looksLikePath(file) && rationale.length >= 8;
  });
}

function nonBlockingSuggestionsSectionIsValid(lines: ReadonlyArray<string>): boolean {
  const contentLines = lines.map((line) => line.trim()).filter(Boolean);
  if (contentLines.length === 0) {
    return false;
  }
  if (contentLines.length === 1 && /\b(no suggestions|no non-blocking suggestions|none)\b/i.test(contentLines[0]!)) {
    return true;
  }

  return contentLines.every((line) => {
    const match = line.match(REVIEW_NON_BLOCKING_SUGGESTION_PATTERN);
    if (!match) {
      return false;
    }
    const suggestionText = match[3]?.trim() ?? "";
    if (suggestionText.length < 8) {
      return false;
    }
    const file = match[1]?.trim() ?? "";
    return !file || looksLikePath(file);
  });
}

function buildReviewRetryMessage(): string {
  return [
    "Your previous /review response was not in the required final format.",
    "Do not add any preamble, process narration, or duplicated sections.",
    "Start immediately with `Blocking Findings`.",
    "Return exactly four sections in this order: Blocking Findings, Non-blocking Suggestions, Open Questions / Assumptions, Short Summary.",
    "Every blocking finding must include severity, file, and rationale using `[error|warning] path:line - rationale`.",
    "Non-blocking Suggestions must contain either bullet suggestions or explicit `None.`.",
    "Short Summary must include an `Overall: ... code health` verdict.",
  ].join(" ");
}

export function createPromptCommandFinalTextValidator(
  resolvedSegments: ReadonlyArray<ResolvedPromptCommandSegment>,
): PromptCommandFinalTextValidator | undefined {
  if (resolvedSegments.length === 0 || !resolvedSegments.every((segment) => segment.name === "review")) {
    return undefined;
  }

  return (candidate) => {
    const sections = extractReviewSections(candidate);
    if (!sections) {
      return { valid: false, retryMessage: buildReviewRetryMessage() };
    }
    if (!blockingFindingsSectionIsValid(sections.blockingFindings)) {
      return { valid: false, retryMessage: buildReviewRetryMessage() };
    }
    if (!nonBlockingSuggestionsSectionIsValid(sections.nonBlockingSuggestions)) {
      return { valid: false, retryMessage: buildReviewRetryMessage() };
    }
    const summaryText = sections.summary.join("\n").trim();
    if (!summaryText || !REVIEW_CODE_HEALTH_VERDICT_PATTERN.test(summaryText)) {
      return { valid: false, retryMessage: buildReviewRetryMessage() };
    }
    return { valid: true };
  };
}

export async function preparePromptCommandQuery(
  query: string,
  options: {
    readonly resolveAutoTarget: (pathFilters: ReadonlyArray<string>) => Promise<ResolvedPromptCommandTarget>;
    readonly loadDiff: (
      target: ResolvedPromptCommandTarget,
      pathFilters: ReadonlyArray<string>,
    ) => Promise<string | null>;
  },
): Promise<PreparedPromptCommandQuery | null> {
  const parsed = parsePromptCommandQuery(query);
  if (!parsed) {
    return null;
  }

  const autoTargetCache = new Map<string, ResolvedPromptCommandTarget>();
  const resolvedSegments: ResolvedPromptCommandSegment[] = [];
  for (const segment of parsed.segments) {
    const cacheKey = segment.pathFilters.join("|");
    const target = isAutoTarget(segment.targetHint)
      ? (autoTargetCache.get(cacheKey) ?? await options.resolveAutoTarget(segment.pathFilters))
      : segment.targetHint;
    if (isAutoTarget(segment.targetHint)) {
      autoTargetCache.set(cacheKey, target);
    }
    resolvedSegments.push({ ...segment, target });
  }

  const preloadedDiffs: Array<{
    readonly target: ResolvedPromptCommandTarget;
    readonly pathFilters: ReadonlyArray<string>;
    readonly content: string;
  }> = [];
  const diffByScope = new Map<string, string | null>();
  for (const segment of resolvedSegments) {
    const key = makePreloadKey(segment.target, segment.pathFilters);
    if (diffByScope.has(key)) {
      continue;
    }

    const diff = await options.loadDiff(segment.target, segment.pathFilters);
    diffByScope.set(key, diff);
    if (!diff) {
      continue;
    }

    preloadedDiffs.push({
      target: segment.target,
      pathFilters: segment.pathFilters,
      content: formatPreloadedDiffMessage(segment.target, diff, segment.pathFilters),
    });
  }

  return {
    parsed,
    resolvedSegments,
    rewrittenQuery: buildPromptCommandRewrite(parsed, resolvedSegments, diffByScope),
    preloadedDiffs,
    finalTextValidator: createPromptCommandFinalTextValidator(resolvedSegments),
  };
}
