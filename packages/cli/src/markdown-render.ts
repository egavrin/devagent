/**
 * Terminal markdown renderer — converts markdown to ANSI-formatted text.
 * Handles: headers, code fences (with syntax highlighting), bold, italic,
 * inline code, lists, horizontal rules.
 *
 * Uses cli-highlight for language-aware code block rendering.
 * Falls back to dim indented text when highlight fails or language unknown.
 */

let highlightLib: typeof import("cli-highlight") | null = null;
let highlightLoadAttempted = false;

function getHighlight(): typeof import("cli-highlight") | null {
  if (!highlightLoadAttempted) {
    highlightLoadAttempted = true;
    // Eagerly start loading (async) — first call to highlightCode may miss,
    // but subsequent calls will have it cached.
    void import("cli-highlight")
      .then((m) => { highlightLib = m; return m; })
      .catch(() => { highlightLib = null; return null; });
  }
  return highlightLib;
}

const NO_COLOR = !!process.env["NO_COLOR"];

// ─── ANSI Codes ─────────────────────────────────────────────

const BOLD = NO_COLOR ? "" : "\x1b[1m";
const DIM = NO_COLOR ? "" : "\x1b[2m";
const ITALIC = NO_COLOR ? "" : "\x1b[3m";
const RESET = NO_COLOR ? "" : "\x1b[0m";
const CYAN = NO_COLOR ? "" : "\x1b[36m";

// ─── Public API ─────────────────────────────────────────────

/**
 * Render markdown text with ANSI formatting for terminal display.
 * Returns raw markdown when NO_COLOR is set or formatting is disabled.
 */
export function renderMarkdown(text: string, enabled: boolean = true): string {
  if (!enabled) return text;

  const lines = text.split("\n");
  const state: MarkdownRenderState = {
    output: [],
    inCodeBlock: false,
    codeLang: "",
    codeLines: [],
  };

  for (let lineIndex = 0; lineIndex < lines.length; lineIndex++) {
    const line = lines[lineIndex]!;
    if (handleCodeBlockLine(state, line)) {
      continue;
    }
    const tableBlock = collectTableBlock(lines, lineIndex);
    if (tableBlock) {
      state.output.push(...renderTableBlock(tableBlock.lines));
      lineIndex = tableBlock.nextIndex;
      continue;
    }
    state.output.push(renderMarkdownLine(line));
  }

  closeCodeBlock(state);
  return state.output.join("\n");
}

interface MarkdownRenderState {
  readonly output: string[];
  inCodeBlock: boolean;
  codeLang: string;
  codeLines: string[];
}

function handleCodeBlockLine(state: MarkdownRenderState, line: string): boolean {
  if (line.trimStart().startsWith("```")) {
    toggleCodeBlock(state, line);
    return true;
  }
  if (!state.inCodeBlock) {
    return false;
  }
  state.codeLines.push(line);
  return true;
}

function toggleCodeBlock(state: MarkdownRenderState, line: string): void {
  if (!state.inCodeBlock) {
    state.inCodeBlock = true;
    state.codeLang = line.trimStart().slice(3).trim();
    state.codeLines = [];
    return;
  }
  closeCodeBlock(state);
}

function closeCodeBlock(state: MarkdownRenderState): void {
  if (!state.inCodeBlock) {
    return;
  }
  const label = state.codeLang ? `${DIM}─── ${state.codeLang} ───${RESET}` : `${DIM}───${RESET}`;
  state.output.push(label);
  state.output.push(highlightCode(state.codeLines.join("\n"), state.codeLang));
  state.output.push(`${DIM}───${RESET}`);
  state.inCodeBlock = false;
  state.codeLang = "";
  state.codeLines = [];
}

function collectTableBlock(
  lines: ReadonlyArray<string>,
  lineIndex: number,
): { readonly lines: string[]; readonly nextIndex: number } | null {
  const line = lines[lineIndex]!;
  if (!line.includes("|") || !line.trim().startsWith("|")) {
    return null;
  }
  const tableLines = [line];
  let nextIndex = lineIndex;
  while (nextIndex + 1 < lines.length && lines[nextIndex + 1]!.trim().startsWith("|")) {
    nextIndex++;
    tableLines.push(lines[nextIndex]!);
  }
  return { lines: tableLines, nextIndex };
}

function renderMarkdownLine(line: string): string {
  return renderHeaderLine(line)
    ?? renderHorizontalRule(line)
    ?? renderUnorderedListLine(line)
    ?? renderOrderedListLine(line)
    ?? formatInline(line);
}

function renderHeaderLine(line: string): string | null {
  const match = line.match(/^(#{1,6})\s+(.+)/);
  if (!match) return null;
  const headerText = match[2]!;
  return match[1]!.length <= 2
    ? `${BOLD}${headerText}${RESET}`
    : `${BOLD}${DIM}${headerText}${RESET}`;
}

function renderHorizontalRule(line: string): string | null {
  return /^[-*_]{3,}\s*$/.test(line) ? `${DIM}${"─".repeat(40)}${RESET}` : null;
}

function renderUnorderedListLine(line: string): string | null {
  const match = line.match(/^(\s*)[-*+]\s+(.*)/);
  if (!match) return null;
  return `${match[1] ?? ""}${DIM}•${RESET} ${formatInline(match[2] ?? "")}`;
}

function renderOrderedListLine(line: string): string | null {
  const match = line.match(/^(\s*)(\d+)[.)]\s+(.*)/);
  if (!match) return null;
  return `${match[1] ?? ""}${DIM}${match[2]!}.${RESET} ${formatInline(match[3] ?? "")}`;
}

function renderTableBlock(lines: ReadonlyArray<string>): ReadonlyArray<string> {
  const rows = lines.map(parseTableRow);
  const contentRows = rows.filter((row) => !row.isSeparator);
  if (contentRows.length === 0) {
    return [];
  }

  const columnCount = Math.max(...contentRows.map((row) => row.cells.length));
  const widths = Array.from({ length: columnCount }, (_, columnIndex) =>
    Math.max(...contentRows.map((row) => row.cells[columnIndex]?.length ?? 0)));

  return rows.map((row) => {
    if (row.isSeparator) {
      return renderTableSeparator(widths);
    }
    return renderTableContentRow(row.cells, widths);
  });
}

function parseTableRow(line: string): { readonly cells: ReadonlyArray<string>; readonly isSeparator: boolean } {
  const cells = line.split("|").slice(1, -1).map((cell) => cell.trim());
  return {
    cells,
    isSeparator: cells.length > 0 && cells.every((cell) => /^:?-{3,}:?$/.test(cell)),
  };
}

function renderTableContentRow(cells: ReadonlyArray<string>, widths: ReadonlyArray<number>): string {
  const formattedCells = widths.map((width, index) => {
    const cell = cells[index] ?? "";
    return formatInline(cell.padEnd(width, " "));
  });
  return formattedCells.join(`${DIM} │ ${RESET}`);
}

function renderTableSeparator(widths: ReadonlyArray<number>): string {
  return widths
    .map((width) => `${DIM}${"─".repeat(Math.max(3, width))}${RESET}`)
    .join(`${DIM}─┼─${RESET}`);
}

// ─── Code Highlighting ──────────────────────────────────────

function highlightCode(code: string, language: string): string {
  if (NO_COLOR || !code.trim()) {
    return code.split("\n").map((l) => `${DIM}  ${l}${RESET}`).join("\n");
  }

  const hl = getHighlight();
  if (hl) {
    try {
      const highlighted = hl.highlight(code, {
        language: language || undefined,
        ignoreIllegals: true,
      });
      // Indent each line
      return highlighted.split("\n").map((l: string) => `  ${l}`).join("\n");
    } catch {
      // Fall through to dim rendering
    }
  }

  // Fallback: dim + indent
  return code.split("\n").map((l) => `${DIM}  ${l}${RESET}`).join("\n");
}

// ─── Inline Formatting ──────────────────────────────────────

function formatInline(text: string): string {
  // Inline code (must be before bold/italic to avoid conflicts)
  text = text.replace(/`([^`]+)`/g, `${CYAN}$1${RESET}`);

  // Bold + italic (***text***)
  text = text.replace(/\*\*\*([^*]+)\*\*\*/g, `${BOLD}${ITALIC}$1${RESET}`);

  // Bold (**text**)
  text = text.replace(/\*\*([^*]+)\*\*/g, `${BOLD}$1${RESET}`);

  // Italic (*text*)
  text = text.replace(/(?<!\*)\*([^*]+)\*(?!\*)/g, `${ITALIC}$1${RESET}`);

  // Strikethrough (~~text~~)
  text = text.replace(/~~([^~]+)~~/g, `${DIM}$1${RESET}`);

  // Links [text](url) → clickable hyperlink (OSC 8) or fallback
  text = text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_match, linkText: string, url: string) => {
    // OSC 8 hyperlink: \x1b]8;;URL\x1b\\TEXT\x1b]8;;\x1b\\
    if (process.env["TERM_PROGRAM"] && !NO_COLOR) {
      return `\x1b]8;;${url}\x1b\\${CYAN}${linkText}${RESET}\x1b]8;;\x1b\\`;
    }
    return `${linkText} ${DIM}(${url})${RESET}`;
  });

  return text;
}
