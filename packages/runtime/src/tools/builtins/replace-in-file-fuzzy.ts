// ─── Levenshtein Distance ───────────────────────────────────

function levenshtein(a: string, b: string): number {
  if (a === b) return 0;
  if (a.length === 0) return b.length;
  if (b.length === 0) return a.length;

  const matrix: number[][] = [];
  for (let i = 0; i <= a.length; i++) {
    matrix[i] = [i];
  }
  for (let j = 0; j <= b.length; j++) {
    matrix[0]![j] = j;
  }

  for (let i = 1; i <= a.length; i++) {
    for (let j = 1; j <= b.length; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      matrix[i]![j] = Math.min(
        matrix[i - 1]![j]! + 1,
        matrix[i]![j - 1]! + 1,
        matrix[i - 1]![j - 1]! + cost,
      );
    }
  }

  return matrix[a.length]![b.length]!;
}

// ─── Replacer Types ─────────────────────────────────────────

/** Context passed to every replacer — lines are pre-split once for performance. */
interface ReplacerCtx {
  readonly content: string;
  readonly lines: string[];
  readonly find: string;
  readonly findLines: string[];
}

/** A replacer yields candidate exact substrings from content that match the search intent. */
type Replacer = (ctx: ReplacerCtx) => Generator<string>;

// Similarity thresholds for block-anchor matching
const SINGLE_CANDIDATE_SIMILARITY_THRESHOLD = 0.3;
const MULTIPLE_CANDIDATES_SIMILARITY_THRESHOLD = 0.7;

// ─── Individual Replacers ───────────────────────────────────

/** 1. Exact match — yields the search string itself. */
const SimpleReplacer: Replacer = function* ({ find }) {
  yield find;
};

/** 2. Line-trimmed — matches when line-level whitespace differs. */
const LineTrimmedReplacer: Replacer = function* ({ content, lines: originalLines, findLines }) {
  const searchLines = withoutTrailingEmptyLine(findLines);

  for (let i = 0; i <= originalLines.length - searchLines.length; i++) {
    if (!matchesTrimmedLines(originalLines, searchLines, i)) continue;
    yield substringByLineRange(content, originalLines, i, i + searchLines.length - 1);
  }
};

function withoutTrailingEmptyLine(lines: ReadonlyArray<string>): string[] {
  const next = [...lines];
  if (next[next.length - 1] === "") next.pop();
  return next;
}

function matchesTrimmedLines(
  originalLines: ReadonlyArray<string>,
  searchLines: ReadonlyArray<string>,
  startLine: number,
): boolean {
  return searchLines.every(
    (line, offset) => originalLines[startLine + offset]!.trim() === line.trim(),
  );
}

function substringByLineRange(
  content: string,
  lines: ReadonlyArray<string>,
  startLine: number,
  endLine: number,
): string {
  let matchStartIndex = 0;
  for (let k = 0; k < startLine; k++) matchStartIndex += lines[k]!.length + 1;
  let matchEndIndex = matchStartIndex;
  for (let k = startLine; k <= endLine; k++) {
    matchEndIndex += lines[k]!.length;
    if (k < endLine) matchEndIndex += 1;
  }
  return content.substring(matchStartIndex, matchEndIndex);
}

/** 3. Block-anchor — first/last lines as anchors, Levenshtein on middle. */
const BlockAnchorReplacer: Replacer = function* ({ content, lines: originalLines, findLines }) {
  const searchLines = withoutTrailingEmptyLine(findLines);

  if (searchLines.length < 3) return;

  const firstLineSearch = searchLines[0]!.trim();
  const lastLineSearch = searchLines[searchLines.length - 1]!.trim();
  const searchBlockSize = searchLines.length;

  const candidates = findAnchorCandidates(originalLines, firstLineSearch, lastLineSearch);

  if (candidates.length === 0) return;

  function extractSubstring(startLine: number, endLine: number): string {
    return substringByLineRange(content, originalLines, startLine, endLine);
  }

  function calcSimilarity(startLine: number, endLine: number, threshold: number): number {
    const actualBlockSize = endLine - startLine + 1;
    const linesToCheck = Math.min(searchBlockSize - 2, actualBlockSize - 2);

    if (linesToCheck <= 0) return 1.0;

    let similarity = 0;
    for (let j = 1; j < searchBlockSize - 1 && j < actualBlockSize - 1; j++) {
      const originalLine = originalLines[startLine + j]!.trim();
      const searchLine = searchLines[j]!.trim();
      const maxLen = Math.max(originalLine.length, searchLine.length);
      if (maxLen === 0) continue;
      const distance = levenshtein(originalLine, searchLine);
      similarity += (1 - distance / maxLen) / linesToCheck;
      if (similarity >= threshold) break;
    }
    return similarity;
  }

  if (candidates.length === 1) {
    const { startLine, endLine } = candidates[0]!;
    const similarity = calcSimilarity(startLine, endLine, SINGLE_CANDIDATE_SIMILARITY_THRESHOLD);
    if (similarity >= SINGLE_CANDIDATE_SIMILARITY_THRESHOLD) {
      yield extractSubstring(startLine, endLine);
    }
    return;
  }

  const best = findBestAnchorCandidate(candidates, calcSimilarity);

  if (best.similarity >= MULTIPLE_CANDIDATES_SIMILARITY_THRESHOLD && best.candidate) {
    yield extractSubstring(best.candidate.startLine, best.candidate.endLine);
  }
};

interface AnchorCandidate {
  readonly startLine: number;
  readonly endLine: number;
}

function findBestAnchorCandidate(
  candidates: ReadonlyArray<AnchorCandidate>,
  calcSimilarity: (startLine: number, endLine: number, threshold: number) => number,
): { readonly candidate: AnchorCandidate | null; readonly similarity: number } {
  let candidate: AnchorCandidate | null = null;
  let similarity = -1;

  for (const next of candidates) {
    const nextSimilarity = calcSimilarity(next.startLine, next.endLine, 0);
    if (nextSimilarity <= similarity) continue;
    similarity = nextSimilarity;
    candidate = next;
  }

  return { candidate, similarity };
}

function findAnchorCandidates(
  lines: ReadonlyArray<string>,
  firstLineSearch: string,
  lastLineSearch: string,
): AnchorCandidate[] {
  const candidates: AnchorCandidate[] = [];
  for (let i = 0; i < lines.length; i++) {
    if (lines[i]!.trim() !== firstLineSearch) continue;
    const endLine = findAnchorEndLine(lines, i + 2, lastLineSearch);
    if (endLine !== null) candidates.push({ startLine: i, endLine });
  }
  return candidates;
}

function findAnchorEndLine(
  lines: ReadonlyArray<string>,
  startLine: number,
  lastLineSearch: string,
): number | null {
  for (let j = startLine; j < lines.length; j++) {
    if (lines[j]!.trim() === lastLineSearch) return j;
  }
  return null;
}

/** 4. Whitespace-normalized — collapse all whitespace and match. */
const WhitespaceNormalizedReplacer: Replacer = function* ({ find, lines, findLines }) {
  const normalize = (text: string) => text.replace(/\s+/g, " ").trim();
  const normalizedFind = normalize(find);

  for (const line of lines) {
    if (normalize(line) === normalizedFind) {
      yield line;
      continue;
    }
    const fuzzyLineMatch = findWhitespaceNormalizedLineMatch(line, find, normalizedFind, normalize);
    if (fuzzyLineMatch) yield fuzzyLineMatch;
  }

  if (findLines.length > 1) {
    for (let i = 0; i <= lines.length - findLines.length; i++) {
      const block = lines.slice(i, i + findLines.length);
      if (normalize(block.join("\n")) === normalizedFind) {
        yield block.join("\n");
      }
    }
  }
};

function findWhitespaceNormalizedLineMatch(
  line: string,
  find: string,
  normalizedFind: string,
  normalize: (text: string) => string,
): string | null {
  if (!normalize(line).includes(normalizedFind)) return null;
  const words = find.trim().split(/\s+/);
  if (words.length === 0) return null;
  const pattern = words.map(escapeRegExp).join("\\s+");
  try {
    return line.match(new RegExp(pattern))?.[0] ?? null;
  } catch {
    return null;
  }
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** 5. Indentation-flexible — matches when indent level differs. */
const IndentationFlexibleReplacer: Replacer = function* ({ find, lines: contentLines, findLines }) {
  const normalizedFind = removeIndentation(find);

  for (let i = 0; i <= contentLines.length - findLines.length; i++) {
    const block = contentLines.slice(i, i + findLines.length).join("\n");
    if (removeIndentation(block) === normalizedFind) {
      yield block;
    }
  }
};

function removeIndentation(text: string): string {
  const textLines = text.split("\n");
  const nonEmpty = textLines.filter((line) => line.trim().length > 0);
  if (nonEmpty.length === 0) return text;

  const minIndent = Math.min(
    ...nonEmpty.map((line) => line.match(/^(\s*)/)?.[1]?.length ?? 0),
  );

  return textLines.map((line) => (line.trim().length === 0 ? line : line.slice(minIndent))).join("\n");
}

/** 6. Escape-normalized — handle escape sequence differences. */
const EscapeNormalizedReplacer: Replacer = function* ({ content, lines, find }) {
  const unescapedFind = unescapeSearchText(find);

  if (content.includes(unescapedFind)) {
    yield unescapedFind;
  }

  const unescapedFindLines = unescapedFind.split("\n");

  for (let i = 0; i <= lines.length - unescapedFindLines.length; i++) {
    const block = lines.slice(i, i + unescapedFindLines.length).join("\n");
    if (unescapeSearchText(block) === unescapedFind) {
      yield block;
    }
  }
};

function unescapeSearchText(value: string): string {
  return value.replace(/\\(n|t|r|'|"|`|\\|\n|\$)/g, (match, ch: string) => {
    switch (ch) {
      case "n": return "\n";
      case "t": return "\t";
      case "r": return "\r";
      case "'": return "'";
      case '"': return '"';
      case "`": return "`";
      case "\\": return "\\";
      case "\n": return "\n";
      case "$": return "$";
      default: return match;
    }
  });
}

/** 7. Trimmed-boundary — leading/trailing whitespace on the overall block. */
const TrimmedBoundaryReplacer: Replacer = function* ({ content, lines, find, findLines }) {
  const trimmedFind = find.trim();
  if (trimmedFind === find) return;

  if (content.includes(trimmedFind)) {
    yield trimmedFind;
  }

  for (let i = 0; i <= lines.length - findLines.length; i++) {
    const block = lines.slice(i, i + findLines.length).join("\n");
    if (block.trim() === trimmedFind) {
      yield block;
    }
  }
};

/** 8. Context-aware — block anchors + 50% middle-line similarity. */
const ContextAwareReplacer: Replacer = function* ({ lines: contentLines, findLines }) {
  const searchLines = withoutTrailingEmptyLine(findLines);
  if (searchLines.length < 3) return;

  const firstLine = searchLines[0]!.trim();
  const lastLine = searchLines[searchLines.length - 1]!.trim();

  for (let i = 0; i < contentLines.length; i++) {
    if (contentLines[i]!.trim() !== firstLine) continue;
    const match = findContextAwareBlock(contentLines, searchLines, i, lastLine);
    if (!match) continue;
    yield match;
    return;
  }
};

function findContextAwareBlock(
  contentLines: ReadonlyArray<string>,
  searchLines: ReadonlyArray<string>,
  startLine: number,
  lastLine: string,
): string | null {
  for (let j = startLine + 2; j < contentLines.length; j++) {
    if (contentLines[j]!.trim() !== lastLine) continue;
    const blockLines = contentLines.slice(startLine, j + 1);
    return isContextAwareMatch(blockLines, searchLines) ? blockLines.join("\n") : null;
  }
  return null;
}

function isContextAwareMatch(
  blockLines: ReadonlyArray<string>,
  searchLines: ReadonlyArray<string>,
): boolean {
  if (blockLines.length !== searchLines.length) return false;
  const score = countMiddleLineMatches(blockLines, searchLines);
  return score.totalNonEmpty === 0 || score.matchingLines / score.totalNonEmpty >= 0.5;
}

function countMiddleLineMatches(
  blockLines: ReadonlyArray<string>,
  searchLines: ReadonlyArray<string>,
): { readonly matchingLines: number; readonly totalNonEmpty: number } {
  let matchingLines = 0;
  let totalNonEmpty = 0;
  for (let k = 1; k < blockLines.length - 1; k++) {
    const blockLine = blockLines[k]!.trim();
    const findLine = searchLines[k]!.trim();
    if (blockLine.length === 0 && findLine.length === 0) continue;
    totalNonEmpty++;
    if (blockLine === findLine) matchingLines++;
  }
  return { matchingLines, totalNonEmpty };
}

// ─── Cascading Replace ──────────────────────────────────────

const REPLACERS: ReadonlyArray<Replacer> = [
  SimpleReplacer,
  LineTrimmedReplacer,
  BlockAnchorReplacer,
  WhitespaceNormalizedReplacer,
  IndentationFlexibleReplacer,
  EscapeNormalizedReplacer,
  TrimmedBoundaryReplacer,
  ContextAwareReplacer,
];

/** Build a ReplacerCtx from content and find strings, pre-splitting lines once. */
function makeCtx(content: string, find: string): ReplacerCtx {
  return {
    content,
    lines: content.split("\n"),
    find,
    findLines: find.split("\n"),
  };
}

/**
 * Try each replacer in order. The first one that yields a match found in content wins.
 * Returns `{ newContent, count }` or throws if no replacer matches.
 */
function fuzzyReplace(
  content: string,
  oldString: string,
  newString: string,
  replaceAll: boolean,
): { newContent: string; count: number } {
  const ctx = makeCtx(content, oldString);

  for (const replacer of REPLACERS) {
    for (const candidate of replacer(ctx)) {
      const result = applyCandidateReplacement(content, candidate, newString, replaceAll);
      if (result) return result;
    }
  }

  throwFuzzyReplaceFailure(content, ctx);
}

function applyCandidateReplacement(
  content: string,
  candidate: string,
  newString: string,
  replaceAll: boolean,
): { newContent: string; count: number } | null {
  const index = content.indexOf(candidate);
  if (index === -1 || candidate === newString) return null;
  return replaceAll
    ? applyReplaceAllCandidate(content, candidate, newString)
    : applySingleCandidate(content, candidate, newString, index);
}

function applyReplaceAllCandidate(
  content: string,
  candidate: string,
  newString: string,
): { newContent: string; count: number } | null {
  const count = content.split(candidate).length - 1;
  const newContent = content.replaceAll(candidate, newString);
  return newContent === content ? null : { newContent, count };
}

function applySingleCandidate(
  content: string,
  candidate: string,
  newString: string,
  index: number,
): { newContent: string; count: number } | null {
  if (index !== content.lastIndexOf(candidate)) return null;
  return {
    newContent: content.substring(0, index) + newString + content.substring(index + candidate.length),
    count: 1,
  };
}

function throwFuzzyReplaceFailure(content: string, ctx: ReplacerCtx): never {
  if (hasAmbiguousCandidate(content, ctx)) {
    throw new Error(
      "Found multiple matches for search string. Provide more surrounding context to make the match unique.",
    );
  }
  throw new Error(
    `Search string not found. Re-read the file for exact text.${buildFuzzyReplaceHint(ctx)}`,
  );
}

function hasAmbiguousCandidate(content: string, ctx: ReplacerCtx): boolean {
  for (const replacer of REPLACERS) {
    for (const candidate of replacer(ctx)) {
      const index = content.indexOf(candidate);
      if (index !== -1 && index !== content.lastIndexOf(candidate)) return true;
    }
  }
  return false;
}

function buildFuzzyReplaceHint(ctx: ReplacerCtx): string {
  const firstSearchLine = ctx.findLines[0]!.trim();
  if (firstSearchLine.length > 5) {
    const hint = findPartialMatchHint(ctx.lines, firstSearchLine);
    if (hint) return hint;
  }
  const preview = ctx.lines.slice(0, 15).map((line, i) => `${i + 1}: ${line}`).join("\n");
  return `\nFile has ${ctx.lines.length} lines. First 15:\n${preview}`;
}

function findPartialMatchHint(lines: ReadonlyArray<string>, firstSearchLine: string): string | null {
  const token = firstSearchLine.substring(0, Math.min(40, firstSearchLine.length));
  for (let i = 0; i < lines.length; i++) {
    if (!lines[i]!.includes(token)) continue;
    const start = Math.max(0, i - 2);
    const end = Math.min(lines.length, i + 3);
    return `\nPartial match near line ${i + 1}:\n` +
      lines.slice(start, end).map((line, idx) => `${start + idx + 1}: ${line}`).join("\n");
  }
  return null;
}

export {
  fuzzyReplace,
  levenshtein,
  makeCtx,
  LineTrimmedReplacer,
  BlockAnchorReplacer,
  WhitespaceNormalizedReplacer,
  IndentationFlexibleReplacer,
};
