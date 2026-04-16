import type {
  ToolFileChangePreview,
  ToolFileChangeLine,
  ToolFileChangeHunk,
} from "@devagent/runtime";
import { buildToolFileStructuredDiffFromUnifiedDiff } from "@devagent/runtime";
import {
  detectSyntaxLanguage,
  getFirstContentLine,
  highlightCodeForTerminal,
} from "./syntax-highlight.js";

interface PresentedDiffHunk {
  readonly key: string;
  readonly lines: ReadonlyArray<ToolFileChangeLine>;
}

interface PresentedFileEdit {
  readonly summary: string;
  readonly hunks: ReadonlyArray<PresentedDiffHunk>;
  readonly truncated: boolean;
}

export interface HighlightedDiffLine extends ToolFileChangeLine {
  readonly renderedText: string;
  readonly syntaxHighlighted: boolean;
}

interface HighlightedDiffHunk {
  readonly key: string;
  readonly lines: ReadonlyArray<HighlightedDiffLine>;
}

interface HighlightedFileEdit {
  readonly summary: string;
  readonly hunks: ReadonlyArray<HighlightedDiffHunk>;
  readonly truncated: boolean;
}

type PresentedDiffItem =
  | { readonly type: "separator"; readonly key: string }
  | { readonly type: "line"; readonly line: ToolFileChangeLine };

type HighlightedDiffItem =
  | { readonly type: "separator"; readonly key: string }
  | { readonly type: "line"; readonly line: HighlightedDiffLine };

const highlightedFileEditCache = new WeakMap<
  ToolFileChangePreview,
  Map<string, HighlightedFileEdit>
>();

export function formatFileEditSummary(fileEdit: ToolFileChangePreview): string {
  if (fileEdit.additions > 0 && fileEdit.deletions > 0) {
    return `Added ${fileEdit.additions} ${pluralize("line", fileEdit.additions)}, removed ${fileEdit.deletions} ${pluralize("line", fileEdit.deletions)}`;
  }
  if (fileEdit.additions > 0) {
    return `Added ${fileEdit.additions} ${pluralize("line", fileEdit.additions)}`;
  }
  if (fileEdit.deletions > 0) {
    return `Removed ${fileEdit.deletions} ${pluralize("line", fileEdit.deletions)}`;
  }
  return "Updated file";
}

function buildPresentedFileEdit(fileEdit: ToolFileChangePreview): PresentedFileEdit {
  const structuredDiff = fileEdit.structuredDiff ?? buildToolFileStructuredDiffFromUnifiedDiff(fileEdit.unifiedDiff);
  return {
    summary: formatFileEditSummary(fileEdit),
    hunks: structuredDiff
      ? structuredDiff.hunks.map((hunk, index) => presentHunk(hunk, index))
      : [],
    truncated: fileEdit.truncated,
  };
}

export function buildHighlightedFileEdit(
  fileEdit: ToolFileChangePreview,
  options?: { readonly bodyWidth?: number },
): HighlightedFileEdit {
  const bodyWidth = options?.bodyWidth;
  const cacheKey = `${bodyWidth ?? "full"}`;
  const cached = highlightedFileEditCache.get(fileEdit)?.get(cacheKey);
  if (cached) {
    return cached;
  }

  const presented = buildPresentedFileEdit(fileEdit);
  const firstLine = getFirstContentLine(fileEdit.after) ?? getFirstContentLine(fileEdit.before);
  const language = detectSyntaxLanguage(fileEdit.path, firstLine);
  const highlighted: HighlightedFileEdit = {
    summary: presented.summary,
    truncated: presented.truncated,
    hunks: presented.hunks.map((hunk) => ({
      key: hunk.key,
      lines: hunk.lines.map((line) => highlightLine(line, language, bodyWidth)),
    })),
  };

  let perFile = highlightedFileEditCache.get(fileEdit);
  if (!perFile) {
    perFile = new Map();
    highlightedFileEditCache.set(fileEdit, perFile);
  }
  perFile.set(cacheKey, highlighted);
  return highlighted;
}

function takeVisiblePresentedDiffItems(
  hunks: ReadonlyArray<PresentedDiffHunk>,
  maxLines: number,
): {
  readonly items: ReadonlyArray<PresentedDiffItem>;
  readonly hiddenLines: number;
} {
  const items: PresentedDiffItem[] = [];

  for (let index = 0; index < hunks.length; index++) {
    if (index > 0) {
      items.push({ type: "separator", key: `separator-${index}` });
    }
    for (const line of hunks[index]!.lines) {
      items.push({ type: "line", line });
    }
  }

  return {
    items: items.slice(0, maxLines),
    hiddenLines: Math.max(0, items.length - maxLines),
  };
}

export function takeVisibleHighlightedDiffItems(
  hunks: ReadonlyArray<HighlightedDiffHunk>,
  maxLines: number,
): {
  readonly items: ReadonlyArray<HighlightedDiffItem>;
  readonly hiddenLines: number;
} {
  const items: HighlightedDiffItem[] = [];

  for (let index = 0; index < hunks.length; index++) {
    if (index > 0) {
      items.push({ type: "separator", key: `separator-${index}` });
    }
    for (const line of hunks[index]!.lines) {
      items.push({ type: "line", line });
    }
  }

  return {
    items: items.slice(0, maxLines),
    hiddenLines: Math.max(0, items.length - maxLines),
  };
}

export function getPresentedDiffGutterWidth(hunks: ReadonlyArray<PresentedDiffHunk>): number {
  let maxLine = 0;
  for (const hunk of hunks) {
    for (const line of hunk.lines) {
      maxLine = Math.max(maxLine, line.oldLine ?? 0, line.newLine ?? 0);
    }
  }
  return Math.max(2, String(maxLine).length);
}

function presentHunk(hunk: ToolFileChangeHunk, index: number): PresentedDiffHunk {
  return {
    key: `hunk-${index}`,
    lines: hunk.lines,
  };
}

function highlightLine(
  line: ToolFileChangeLine,
  language: string | undefined,
  bodyWidth: number | undefined,
): HighlightedDiffLine {
  if (line.text.length === 0) {
    return {
      ...line,
      renderedText: "<blank>",
      syntaxHighlighted: false,
    };
  }

  const visibleText = truncateCode(line.text, bodyWidth);
  const highlighted = highlightCodeForTerminal(visibleText, language);
  return {
    ...line,
    renderedText: highlighted.text,
    syntaxHighlighted: highlighted.syntaxHighlighted,
  };
}

function pluralize(word: string, count: number): string {
  return count === 1 ? word : `${word}s`;
}

function truncateCode(text: string, width: number | undefined): string {
  if (!width || width <= 1 || text.length <= width) {
    return text;
  }
  return `${text.slice(0, Math.max(1, width - 1))}…`;
}
