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
import type { Violation, ReviewResult, ReviewSummary } from "./schema.js";
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

## Critical Instructions

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
- DO NOT report the same violation multiple times

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
  const { provider, workspaceRoot } = options;

  // Read input files
  const patchPath = resolve(workspaceRoot, input.patchFile);
  const rulePath = resolve(workspaceRoot, input.ruleFile);

  const patchContent = readFileSync(patchPath, "utf-8");
  const ruleContent = readFileSync(rulePath, "utf-8");
  const ruleName = basename(rulePath, ".md");

  // Parse patch
  const parser = new PatchParser(patchContent, true);
  const filterPattern = extractAppliesToPattern(ruleContent);
  const parsed: ParsedPatch = parser.parse(filterPattern);

  let files: FileEntry[] = parsed.files.sort(
    (a, b) => a.path.localeCompare(b.path) || (a._chunkIndex ?? 0) - (b._chunkIndex ?? 0),
  );

  if (files.length === 0) {
    return {
      violations: [],
      summary: {
        totalViolations: 0,
        filesReviewed: 0,
        ruleName,
      },
    };
  }

  // Dynamic limit computation
  const dynamicLineLimit = computeDynamicLineLimit(config.maxLinesPerChunk, ruleContent, files);

  // Split large files
  files = splitLargeFileEntries(files, {
    maxHunksPerChunk: config.maxHunksPerChunk,
    maxLinesPerChunk: dynamicLineLimit > 0 ? dynamicLineLimit : config.maxLinesPerChunk,
  }).sort(
    (a, b) => a.path.localeCompare(b.path) || (a._chunkIndex ?? 0) - (b._chunkIndex ?? 0),
  );

  // Compute dynamic file limit
  const maxFilesPerChunk = computeDynamicFileLimit(config.maxFilesPerChunk, ruleContent, files);

  // Chunk files
  let chunks = chunkPatchFiles(files, {
    maxFilesPerChunk,
    maxLinesPerChunk: dynamicLineLimit,
    chunkOverlapLines: config.chunkOverlapLines,
  });

  // Refine chunks for token budget
  chunks = refineChunksForTokenBudget(chunks, ruleContent, filterPattern ?? undefined, config.tokenBudget);

  // Build context orchestrator
  const contextOrchestrator = new ContextOrchestrator(
    [
      new SourceContextProvider({
        padLines: options.contextPadLines ?? 40,
        maxLinesPerItem: options.contextMaxLinesPerItem ?? 600,
      }),
    ],
    { maxTotalLines: options.contextMaxTotalLines ?? 1500 },
  );

  // System prompt with rule content
  const systemPrompt = `# Coding Rule: ${basename(rulePath)}

${ruleContent}

---

The above rule has been pre-loaded for your review. Do NOT use tools to read it.
Follow the workflow in the user prompt to analyze the patch against this rule.
Output your findings as valid JSON matching the provided schema.`;

  // Process each chunk
  const allViolations: Violation[] = [];
  const allFiles = new Set<string>();

  for (let i = 0; i < chunks.length; i++) {
    const chunk = chunks[i]!;

    // Build dataset text
    const subPatch: ParsedPatch = {
      patchInfo: parsed.patchInfo,
      files: chunk,
      summary: { totalFiles: 0, filesAdded: 0, filesModified: 0, filesDeleted: 0, totalAdditions: 0, totalDeletions: 0 },
    };
    const datasetText = formatPatchDataset(subPatch, filterPattern ?? undefined);

    // Build context section
    const contextSection = contextOrchestrator.buildSection(workspaceRoot, chunk);

    // Build chunk header
    const chunkHeader = chunks.length > 1
      ? `(Chunk ${i + 1} of ${chunks.length})\n\n`
      : "";

    // Build user prompt
    const userPrompt = buildReviewPrompt(
      basename(patchPath),
      chunkHeader + datasetText,
      contextSection || undefined,
    );

    // Collect review data for validation
    const { addedLines, removedLines, parsedFiles } = collectPatchReviewData(chunk);

    try {
      // Call LLM
      const response = await callLLMForReview(provider, systemPrompt, userPrompt);

      // Validate response
      const validated = validateReviewResponse(response, addedLines, removedLines, parsedFiles);

      allViolations.push(...validated.violations);
      for (const f of parsedFiles) allFiles.add(f);
    } catch (err) {
      // Log error but continue with other chunks
      const msg = extractErrorMessage(err);
      console.error(`Review chunk ${i + 1}/${chunks.length} failed: ${msg}`);
      for (const f of chunk) allFiles.add(f.path);
    }
  }

  // Deduplicate violations from overlapping chunks
  const deduplicated = deduplicateViolations(allViolations);

  return {
    violations: deduplicated,
    summary: {
      totalViolations: deduplicated.length,
      filesReviewed: allFiles.size,
      ruleName,
    },
  };
}
