import { basename, extname } from "node:path";
import { highlight } from "cli-highlight";

const EXTENSION_LANGUAGE_MAP: Readonly<Record<string, string>> = {
  ".ts": "typescript",
  ".tsx": "typescript",
  ".mts": "typescript",
  ".cts": "typescript",
  ".js": "javascript",
  ".jsx": "javascript",
  ".mjs": "javascript",
  ".cjs": "javascript",
  ".json": "json",
  ".md": "markdown",
  ".sh": "bash",
  ".bash": "bash",
  ".zsh": "bash",
  ".yml": "yaml",
  ".yaml": "yaml",
  ".toml": "toml",
  ".py": "python",
  ".rb": "ruby",
  ".rs": "rust",
  ".go": "go",
  ".java": "java",
  ".kt": "kotlin",
  ".swift": "swift",
  ".html": "html",
  ".htm": "html",
  ".xml": "xml",
  ".css": "css",
  ".scss": "scss",
  ".sql": "sql",
  ".php": "php",
  ".c": "c",
  ".h": "c",
  ".cc": "cpp",
  ".cpp": "cpp",
  ".cxx": "cpp",
  ".hpp": "cpp",
  ".hh": "cpp",
  ".cs": "csharp",
};

export interface TerminalHighlightResult {
  readonly text: string;
  readonly syntaxHighlighted: boolean;
}

type ThemeFormatter = (code: string) => string;

const CLI_THEME: Readonly<Record<string, ThemeFormatter>> = {
  keyword: color("34"),
  built_in: color("36"),
  type: color("36"),
  literal: color("34"),
  number: color("32"),
  regexp: color("31"),
  string: color("31"),
  class: color("34"),
  function: color("33"),
  comment: color("32"),
  doctag: color("32"),
  meta: color("90"),
  tag: color("90"),
  name: color("34"),
  attr: color("36"),
  addition: color("32"),
  deletion: color("31"),
  default: identity,
};

export function detectSyntaxLanguage(
  filePath: string,
  firstLine?: string | null,
): string | undefined {
  const base = basename(filePath).toLowerCase();
  if (base === "dockerfile") return "dockerfile";
  if (base === "makefile") return "makefile";

  const extension = extname(filePath).toLowerCase();
  const mapped = EXTENSION_LANGUAGE_MAP[extension];
  if (mapped) return mapped;

  const shebang = firstLine?.trimStart();
  if (shebang?.startsWith("#!")) {
    if (shebang.includes("node")) return "javascript";
    if (shebang.includes("python")) return "python";
    if (shebang.includes("ruby")) return "ruby";
    if (shebang.includes("bash") || shebang.includes("sh") || shebang.includes("zsh")) return "bash";
  }

  return undefined;
}

export function getFirstContentLine(text?: string): string | null {
  if (!text) return null;
  const firstLine = text.split("\n", 1)[0];
  return firstLine ?? null;
}

export function highlightCodeForTerminal(
  code: string,
  language?: string,
): TerminalHighlightResult {
  if (!isColorEnabled() || code.length === 0 || !language) {
    return { text: code, syntaxHighlighted: false };
  }

  try {
    return {
      text: highlight(code, {
        language,
        ignoreIllegals: true,
        theme: CLI_THEME,
      }),
      syntaxHighlighted: true,
    };
  } catch {
    return { text: code, syntaxHighlighted: false };
  }
}

function color(code: string): ThemeFormatter {
  return (text: string) => (isColorEnabled() ? `\x1b[${code}m${text}\x1b[0m` : text);
}

function identity(text: string): string {
  return text;
}

function isColorEnabled(): boolean {
  return !process.env["NO_COLOR"];
}
