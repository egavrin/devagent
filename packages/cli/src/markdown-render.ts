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
let highlightLoadPromise: Promise<typeof import("cli-highlight") | null> | null = null;

function getHighlight(): typeof import("cli-highlight") | null {
  if (!highlightLoadAttempted) {
    highlightLoadAttempted = true;
    // Eagerly start loading (async) — first call to highlightCode may miss,
    // but subsequent calls will have it cached.
    highlightLoadPromise = import("cli-highlight")
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
const GREEN = NO_COLOR ? "" : "\x1b[32m";
const RED = NO_COLOR ? "" : "\x1b[31m";

// ─── Public API ─────────────────────────────────────────────

/**
 * Render markdown text with ANSI formatting for terminal display.
 * Returns raw markdown when NO_COLOR is set or formatting is disabled.
 */
export function renderMarkdown(text: string, enabled: boolean = true): string {
  if (!enabled) return text;

  const lines = text.split("\n");
  const output: string[] = [];
  let inCodeBlock = false;
  let codeLang = "";
  let codeLines: string[] = [];

  for (let lineIndex = 0; lineIndex < lines.length; lineIndex++) {
    const line = lines[lineIndex]!;
    // Code fence start/end
    if (line.trimStart().startsWith("```")) {
      if (!inCodeBlock) {
        inCodeBlock = true;
        codeLang = line.trimStart().slice(3).trim();
        codeLines = [];
        continue;
      } else {
        // End of code block — render with syntax highlighting
        const label = codeLang ? `${DIM}─── ${codeLang} ───${RESET}` : `${DIM}───${RESET}`;
        output.push(label);
        output.push(highlightCode(codeLines.join("\n"), codeLang));
        output.push(`${DIM}───${RESET}`);
        inCodeBlock = false;
        codeLang = "";
        codeLines = [];
        continue;
      }
    }

    if (inCodeBlock) {
      codeLines.push(line);
      continue;
    }

    // Headers
    const headerMatch = line.match(/^(#{1,6})\s+(.+)/);
    if (headerMatch) {
      const level = headerMatch[1]!.length;
      const headerText = headerMatch[2]!;
      if (level <= 2) {
        output.push(`${BOLD}${headerText}${RESET}`);
      } else {
        output.push(`${BOLD}${DIM}${headerText}${RESET}`);
      }
      continue;
    }

    // Horizontal rule
    if (/^[-*_]{3,}\s*$/.test(line)) {
      output.push(`${DIM}${"─".repeat(40)}${RESET}`);
      continue;
    }

    // Table rows (pipe-delimited)
    if (line.includes("|") && line.trim().startsWith("|")) {
      const tableLines = [line];
      while (lineIndex + 1 < lines.length && lines[lineIndex + 1]!.trim().startsWith("|")) {
        lineIndex++;
        tableLines.push(lines[lineIndex]!);
      }
      output.push(...renderTableBlock(tableLines));
      continue;
    }

    // List items (unordered)
    const ulMatch = line.match(/^(\s*)[-*+]\s+(.*)/);
    if (ulMatch) {
      const indent = ulMatch[1] ?? "";
      const content = formatInline(ulMatch[2] ?? "");
      output.push(`${indent}${DIM}•${RESET} ${content}`);
      continue;
    }

    // List items (ordered)
    const olMatch = line.match(/^(\s*)(\d+)[.)]\s+(.*)/);
    if (olMatch) {
      const indent = olMatch[1] ?? "";
      const num = olMatch[2]!;
      const content = formatInline(olMatch[3] ?? "");
      output.push(`${indent}${DIM}${num}.${RESET} ${content}`);
      continue;
    }

    // Regular text — apply inline formatting
    output.push(formatInline(line));
  }

  // Close unclosed code block
  if (inCodeBlock) {
    const label = codeLang ? `${DIM}─── ${codeLang} ───${RESET}` : `${DIM}───${RESET}`;
    output.push(label);
    output.push(highlightCode(codeLines.join("\n"), codeLang));
    output.push(`${DIM}───${RESET}`);
  }

  return output.join("\n");
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

/**
 * Render a unified diff with colored additions/deletions.
 * Green for additions (+), red for deletions (-), dim for context.
 */
function renderDiff(diff: string): string {
  if (NO_COLOR) return diff;

  return diff.split("\n").map((line) => {
    if (line.startsWith("+") && !line.startsWith("+++")) {
      return `${GREEN}${line}${RESET}`;
    }
    if (line.startsWith("-") && !line.startsWith("---")) {
      return `${RED}${line}${RESET}`;
    }
    if (line.startsWith("@@")) {
      return `${CYAN}${line}${RESET}`;
    }
    if (line.startsWith("diff ") || line.startsWith("index ") || line.startsWith("---") || line.startsWith("+++")) {
      return `${BOLD}${line}${RESET}`;
    }
    return `${DIM}${line}${RESET}`;
  }).join("\n");
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
