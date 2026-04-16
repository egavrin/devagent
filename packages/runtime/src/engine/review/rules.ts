/**
 * Rule file handling -- extract "Applies To" patterns from rule markdown
 * and convert glob patterns to regular expressions.
 */

// ── Glob-to-regex conversion ────────────────────────────────────────────────

/**
 * Convert a simple glob pattern to a regular expression string.
 *
 * Supported syntax:
 *  - `**`  -> `.*`        (match any path segment including separators)
 *  - `*`   -> `[^/]*`     (match anything except path separator)
 *  - `?`   -> `.`         (match single character)
 *  - `.`   -> `\\.`       (literal dot)
 *
 * The returned regex is anchored with `$` at the end.
 */
function globToRegex(glob: string): string {
  let result = "";
  let i = 0;

  while (i < glob.length) {
    const ch = glob[i]!;

    if (ch === "*") {
      if (i + 1 < glob.length && glob[i + 1] === "*") {
        // `**` -- match anything including path separators
        result += ".*";
        i += 2;
        // Skip a trailing slash after ** (e.g. `**/`)
        if (i < glob.length && glob[i] === "/") {
          result += "(?:/|$)";
          i += 1;
        }
      } else {
        // single `*` -- match within one path segment
        result += "[^/]*";
        i += 1;
      }
    } else if (ch === "?") {
      result += ".";
      i += 1;
    } else if (ch === ".") {
      result += "\\.";
      i += 1;
    } else if (
      ch === "(" ||
      ch === ")" ||
      ch === "[" ||
      ch === "]" ||
      ch === "{" ||
      ch === "}" ||
      ch === "+" ||
      ch === "^" ||
      ch === "$" ||
      ch === "|" ||
      ch === "\\"
    ) {
      // Escape regex special characters that are not glob metacharacters
      result += "\\" + ch;
      i += 1;
    } else {
      result += ch;
      i += 1;
    }
  }

  return result + "$";
}

// ── Pattern normalisation ───────────────────────────────────────────────────

/**
 * Normalise an "Applies To" raw value into a single regex string that can
 * be used with `new RegExp(...)`.
 *
 * Tokens are split on commas and whitespace. Each token is handled as:
 *  - `regex:<pattern>` -- used verbatim as a regex fragment
 *  - contains regex metacharacters `( ) [ ] { } | \` -- used as-is
 *  - otherwise treated as a glob and converted via `globToRegex`
 *
 * Multiple tokens are combined with alternation (`|`).
 */
function normalizeAppliesToPattern(raw: string): string | null {
  const tokens = raw
    .split(/[,\s]+/)
    .map((t) => t.trim())
    .filter((t) => t.length > 0);

  if (tokens.length === 0) {
    return null;
  }

  const regexParts: string[] = [];

  for (const token of tokens) {
    // regex: prefix -- use custom regex verbatim
    if (token.toLowerCase().startsWith("regex:")) {
      const custom = token.slice(6).trim();
      if (custom.length > 0) {
        regexParts.push(custom);
      }
      continue;
    }

    // If the token already contains regex metacharacters, keep it as-is
    if (/[()[\]{}|\\]/.test(token)) {
      regexParts.push(token);
      continue;
    }

    // Treat as glob
    regexParts.push(globToRegex(token));
  }

  if (regexParts.length === 0) {
    return null;
  }

  if (regexParts.length === 1) {
    return regexParts[0]!;
  }

  return regexParts.map((part) => `(?:${part})`).join("|");
}

// ── Pattern extraction from rule content ────────────────────────────────────

/**
 * Patterns used to locate the "applies to" declaration inside a rule file.
 * Tried in order; the first match wins.
 */
const EXTRACTION_PATTERNS: RegExp[] = [
  /##\s*Applies\s+To\s*\n\s*([^\n]+)/i,
  /Applies\s+To:\s*([^\n]+)/i,
  /scope:\s*([^\n]+)/i,
  /##\s*Scope\s*\n\s*([^\n]+)/i,
];

/**
 * Extract an "Applies To" pattern from the text of a rule file and return
 * a normalised regex string (or `null` if no pattern is found).
 */
export function extractAppliesToPattern(ruleContent: string): string | null {
  for (const pattern of EXTRACTION_PATTERNS) {
    const match = pattern.exec(ruleContent);
    if (match) {
      // Strip surrounding quotes / backticks
      const extracted = (match[1] ?? "").trim().replace(/["'`]/g, "");
      if (extracted.length > 0) {
        return normalizeAppliesToPattern(extracted) ?? extracted;
      }
    }
  }
  return null;
}
