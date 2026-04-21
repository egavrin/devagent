/**
 * Review pipeline orchestrator — wires together patch parsing, chunking,
 * context enrichment, LLM review, validation, and deduplication.
 *
 * This is the main entry point for rule-based patch review.
 */

import { readFileSync } from "node:fs";
import { resolve, basename } from "node:path";

import {
  splitLargeFileEntries,
  chunkPatchFiles,
  computeDynamicLineLimit,
  computeDynamicFileLimit,
  refineChunksForTokenBudget,
  formatPatchDataset,
 DEFAULT_REVIEW_CONFIG } from "./chunker.js";
import type { ReviewConfig } from "./chunker.js";
import {
  SourceContextProvider,
  ContextOrchestrator,
} from "./context.js";
import { extractAppliesToPattern } from "./rules.js";
import { VIOLATION_SCHEMA } from "./schema.js";
import type { Violation, ReviewResult } from "./schema.js";
import {
  collectPatchReviewData,
  validateReviewResponse,
  deduplicateViolations,
} from "./validator.js";
import type { LLMProvider, Message } from "../../core/index.js";
import { MessageRole , extractErrorMessage } from "../../core/index.js";
import { PatchParser } from "../../tools/builtins/patch-parser.js";
import type { FileEntry, ParsedPatch } from "../../tools/builtins/patch-parser.js";


// ── Pipeline options ────────────────────────────────────────────────────────

export interface ReviewPipelineOptions {
  provider: LLMProvider;
  workspaceRoot: string;
  config?: Partial<ReviewConfig>;
  contextPadLines?: number;
  contextMaxLinesPerItem?: number;
  contextMaxTotalLines?: number;
}

export interface ReviewPipelineInput {
  patchFile: string;
  ruleFile: string;
}

// ── Review prompt builder ───────────────────────────────────────────────────

const REVIEW_WORKFLOW_TEXT = `## Critical Instructions

1. **USE ONLY THE PRE-PARSED PATCH DATASET BELOW** as your source of truth
   - Do NOT attempt to read files using tools
   - Do NOT guess at line numbers or content
   - All line numbers and code snippets MUST come from the "ADDED LINES" sections

2. **EXACT MATCHING REQUIRED**
   - The \`file\` field must exactly match the file path shown in the dataset
   - The \`line\` field must exactly match a line number from "ADDED LINES"
   - The \`codeSnippet\` field must exactly match the content from that line (strip whitespace for comparison)

3. **OUTPUT FORMAT**
   - Return valid JSON conforming to the schema provided
   - Each violation MUST reference an actual added line from the dataset
   - If no violations found, return empty violations array

## Review Workflow

### Step 1: Understand the Rule
- Read the rule's title, description, and scope
- Study the "Detect" section to understand what constitutes a violation
- Review examples of BAD and GOOD code patterns

### Step 2: Analyze the Patch Dataset
- The dataset shows files with their added/removed lines
- Focus ONLY on "ADDED LINES" sections (marked with +)
- Note the exact line numbers next to each added line

### Step 3: Identify Violations
For each added line:
- Check if it matches the violation pattern described in the rule
- Verify the file matches the rule's "Applies To" pattern
- If it's a violation:
  * Record the EXACT file path from the dataset
  * Record the EXACT line number from the "ADDED LINES" section
  * Copy the EXACT code snippet (you may strip leading/trailing whitespace)
  * Write a clear, actionable message using the rule's template

### Step 4: Validate Your Output
Before submitting:
- Verify each violation references an actual line from "ADDED LINES"
- Confirm line numbers match exactly
- Ensure file paths are correct
- Check that code snippets match (ignoring whitespace)

## Common Mistakes to Avoid

- DO NOT report violations for lines not in the patch
- DO NOT use line numbers from context or removed lines
- DO NOT guess or estimate line numbers
- DO NOT read files with tools - use only the dataset below
- DO NOT report the same violation multiple times`;

function buildReviewPrompt(
  patchName: string,
  patchDataset: string,
  contextSection?: string,
): string {
  const contextBlock = contextSection
    ? `\n## Additional Source Context\n\n${contextSection}\n`
    : "";

  return `# Code Review Task: ${patchName}

You are reviewing a patch file against a specific coding rule. Your task is to identify violations with precision and accuracy.

${REVIEW_WORKFLOW_TEXT}

## Required JSON Schema

\`\`\`json
${JSON.stringify(VIOLATION_SCHEMA, null, 2)}
\`\`\`
${contextBlock}
## Patch Dataset (Your Single Source of Truth)

${patchDataset}

---

Now review the patch and output your findings as valid JSON matching the schema above.`;
}

// ── LLM interaction ─────────────────────────────────────────────────────────

async function callLLMForReview(
  provider: LLMProvider,
  systemPrompt: string,
  userPrompt: string,
): Promise<unknown> {
  const messages: Message[] = [
    { role: MessageRole.SYSTEM, content: systemPrompt },
    { role: MessageRole.USER, content: userPrompt },
  ];

  let fullResponse = "";

  for await (const chunk of provider.chat(messages)) {
    if (chunk.type === "text") {
      fullResponse += chunk.content;
    }
  }

  // Extract JSON from response (may be wrapped in markdown code blocks)
  const jsonMatch = fullResponse.match(/```(?:json)?\s*([\s\S]*?)```/);
  const jsonStr = jsonMatch ? jsonMatch[1]!.trim() : fullResponse.trim();

  try {
    return JSON.parse(jsonStr);
  } catch {
    // Try to find JSON object in the response
    const objectMatch = jsonStr.match(/\{[\s\S]*\}/);
    if (objectMatch) {
      return JSON.parse(objectMatch[0]);
    }
    throw new Error(`Failed to parse LLM review response as JSON: ${jsonStr.slice(0, 200)}`);
  }
}

// ── Pipeline ────────────────────────────────────────────────────────────────

/**
 * Run the full review pipeline:
 * 1. Parse patch file
 * 2. Extract rule pattern and filter files
 * 3. Chunk the patch for manageable LLM calls
 * 4. For each chunk: build prompt → call LLM → validate response
 * 5. Deduplicate and aggregate violations
 */
export async function runReviewPipeline(
  options: ReviewPipelineOptions,
  input: ReviewPipelineInput,
): Promise<ReviewResult> {
  const config: ReviewConfig = { ...DEFAULT_REVIEW_CONFIG, ...options.config };
  const prepared = prepareReviewInput(options.workspaceRoot, input);
  let files = sortPatchFiles(prepared.parsed.files);

  if (files.length === 0) {
    return {
      violations: [],
      summary: {
        totalViolations: 0,
        filesReviewed: 0,
        ruleName: prepared.ruleName,
      },
    };
  }

  const chunks = buildReviewChunks(files, prepared, config);
  const contextOrchestrator = createContextOrchestrator(options);

  const systemPrompt = `# Coding Rule: ${basename(prepared.rulePath)}

${prepared.ruleContent}

---

The above rule has been pre-loaded for your review. Do NOT use tools to read it.
Follow the workflow in the user prompt to analyze the patch against this rule.
Output your findings as valid JSON matching the provided schema.`;

  const review = await processReviewChunks({
    provider: options.provider,
    workspaceRoot: options.workspaceRoot,
    patchName: basename(prepared.patchPath),
    parsedPatch: prepared.parsed,
    filterPattern: prepared.filterPattern,
    contextOrchestrator,
    systemPrompt,
    chunks,
  });
  const deduplicated = deduplicateViolations(review.violations);

  return {
    violations: deduplicated,
    summary: {
      totalViolations: deduplicated.length,
      filesReviewed: review.files.size,
      ruleName: prepared.ruleName,
    },
  };
}

function prepareReviewInput(workspaceRoot: string, input: ReviewPipelineInput): {
  readonly patchPath: string;
  readonly rulePath: string;
  readonly ruleContent: string;
  readonly ruleName: string;
  readonly filterPattern: string | null;
  readonly parsed: ParsedPatch;
} {
  const patchPath = resolve(workspaceRoot, input.patchFile);
  const rulePath = resolve(workspaceRoot, input.ruleFile);
  const patchContent = readFileSync(patchPath, "utf-8");
  const ruleContent = readFileSync(rulePath, "utf-8");
  const filterPattern = extractAppliesToPattern(ruleContent);
  return {
    patchPath,
    rulePath,
    ruleContent,
    ruleName: basename(rulePath, ".md"),
    filterPattern,
    parsed: new PatchParser(patchContent, true).parse(filterPattern),
  };
}

function sortPatchFiles(files: FileEntry[]): FileEntry[] {
  return files.sort(
    (a, b) => a.path.localeCompare(b.path) || (a._chunkIndex ?? 0) - (b._chunkIndex ?? 0),
  );
}

function buildReviewChunks(
  files: FileEntry[],
  input: { readonly ruleContent: string; readonly filterPattern: string | null },
  config: ReviewConfig,
): FileEntry[][] {
  const dynamicLineLimit = computeDynamicLineLimit(config.maxLinesPerChunk, input.ruleContent, files);
  const splitFiles = sortPatchFiles(splitLargeFileEntries(files, {
    maxHunksPerChunk: config.maxHunksPerChunk,
    maxLinesPerChunk: dynamicLineLimit > 0 ? dynamicLineLimit : config.maxLinesPerChunk,
  }));
  const maxFilesPerChunk = computeDynamicFileLimit(config.maxFilesPerChunk, input.ruleContent, splitFiles);
  const chunks = chunkPatchFiles(splitFiles, {
    maxFilesPerChunk,
    maxLinesPerChunk: dynamicLineLimit,
    chunkOverlapLines: config.chunkOverlapLines,
  });
  return refineChunksForTokenBudget(chunks, input.ruleContent, input.filterPattern ?? undefined, config.tokenBudget);
}

function createContextOrchestrator(options: ReviewPipelineOptions): ContextOrchestrator {
  return new ContextOrchestrator(
    [
      new SourceContextProvider({
        padLines: options.contextPadLines ?? 40,
        maxLinesPerItem: options.contextMaxLinesPerItem ?? 600,
      }),
    ],
    { maxTotalLines: options.contextMaxTotalLines ?? 1500 },
  );
}

async function processReviewChunks(input: {
  readonly provider: LLMProvider;
  readonly workspaceRoot: string;
  readonly patchName: string;
  readonly parsedPatch: ParsedPatch;
  readonly filterPattern: string | null;
  readonly contextOrchestrator: ContextOrchestrator;
  readonly systemPrompt: string;
  readonly chunks: ReadonlyArray<FileEntry[]>;
}): Promise<{ readonly violations: Violation[]; readonly files: Set<string> }> {
  const allViolations: Violation[] = [];
  const allFiles = new Set<string>();

  for (let index = 0; index < input.chunks.length; index++) {
    const chunk = input.chunks[index]!;
    try {
      const result = await processReviewChunk(input, chunk, index);
      allViolations.push(...result.violations);
      for (const file of result.files) allFiles.add(file);
    } catch (err) {
      const msg = extractErrorMessage(err);
      console.error(`Review chunk ${index + 1}/${input.chunks.length} failed: ${msg}`);
      for (const f of chunk) allFiles.add(f.path);
    }
  }

  return { violations: allViolations, files: allFiles };
}

async function processReviewChunk(
  input: {
    readonly provider: LLMProvider;
    readonly workspaceRoot: string;
    readonly patchName: string;
    readonly parsedPatch: ParsedPatch;
    readonly filterPattern: string | null;
    readonly contextOrchestrator: ContextOrchestrator;
    readonly systemPrompt: string;
    readonly chunks: ReadonlyArray<FileEntry[]>;
  },
  chunk: FileEntry[],
  index: number,
): Promise<{ readonly violations: Violation[]; readonly files: Set<string> }> {
  const userPrompt = buildReviewPrompt(
    input.patchName,
    buildChunkDataset(input, chunk, index),
    input.contextOrchestrator.buildSection(input.workspaceRoot, chunk) || undefined,
  );
  const { addedLines, removedLines, parsedFiles } = collectPatchReviewData(chunk);
  const response = await callLLMForReview(input.provider, input.systemPrompt, userPrompt);
  const validated = validateReviewResponse(response, addedLines, removedLines, parsedFiles);
  return { violations: validated.violations, files: parsedFiles };
}

function buildChunkDataset(
  input: {
    readonly parsedPatch: ParsedPatch;
    readonly filterPattern: string | null;
    readonly chunks: ReadonlyArray<FileEntry[]>;
  },
  chunk: FileEntry[],
  index: number,
): string {
  const datasetText = formatPatchDataset({
    patchInfo: input.parsedPatch.patchInfo,
    files: chunk,
    summary: { totalFiles: 0, filesAdded: 0, filesModified: 0, filesDeleted: 0, totalAdditions: 0, totalDeletions: 0 },
  }, input.filterPattern ?? undefined);
  const chunkHeader = input.chunks.length > 1
    ? `(Chunk ${index + 1} of ${input.chunks.length})\n\n`
    : "";
  return chunkHeader + datasetText;
}
