/**
 * Shared glob matching helpers for built-in file tools.
 */

export function globToRegex(pattern: string): RegExp {
  const normalized = toPosixPath(pattern);
  return new RegExp(`^${globToRegexFragment(normalized)}$`);
}

/** Auto-prepend **​/ for patterns without path separators so *.ts matches nested files. */
export function normalizeGlobPattern(pattern: string): string {
  return pattern.includes("/") ? pattern : `**/${pattern}`;
}

export function matchesGlob(regex: RegExp, candidates: readonly string[]): boolean {
  return candidates.some((candidate) => regex.test(candidate));
}

export function toPosixPath(path: string): string {
  return path.replaceAll("\\", "/");
}

function globToRegexFragment(pattern: string): string {
  let regex = "";

  for (let i = 0; i < pattern.length; i++) {
    const char = pattern[i]!;

    if (char === "*") {
      const next = pattern[i + 1];
      const afterNext = pattern[i + 2];
      if (next === "*") {
        if (afterNext === "/") {
          regex += "(?:.*/)?";
          i += 2;
        } else {
          regex += ".*";
          i += 1;
        }
      } else {
        regex += "[^/]*";
      }
      continue;
    }

    if (char === "?") {
      regex += "[^/]";
      continue;
    }

    if (char === "{") {
      const closingBrace = findMatchingBrace(pattern, i);
      if (closingBrace === -1) {
        regex += "\\{";
        continue;
      }

      const content = pattern.slice(i + 1, closingBrace);
      const alternatives = splitAlternatives(content);
      regex += `(?:${alternatives.map((part) => globToRegexFragment(part)).join("|")})`;
      i = closingBrace;
      continue;
    }

    regex += escapeRegexChar(char);
  }

  return regex;
}

function findMatchingBrace(value: string, start: number): number {
  let depth = 0;
  for (let i = start; i < value.length; i++) {
    const char = value[i];
    if (char === "{") {
      depth += 1;
      continue;
    }
    if (char === "}") {
      depth -= 1;
      if (depth === 0) {
        return i;
      }
    }
  }
  return -1;
}

function splitAlternatives(value: string): string[] {
  const alternatives: string[] = [];
  let depth = 0;
  let current = "";

  for (let i = 0; i < value.length; i++) {
    const char = value[i]!;
    if (char === "{") {
      depth += 1;
      current += char;
      continue;
    }
    if (char === "}") {
      depth -= 1;
      current += char;
      continue;
    }
    if (char === "," && depth === 0) {
      alternatives.push(current);
      current = "";
      continue;
    }
    current += char;
  }

  alternatives.push(current);
  return alternatives;
}

function escapeRegexChar(char: string): string {
  return /[\\^$+?.()|[\]{}]/.test(char) ? `\\${char}` : char;
}

/** Escape a full string for use as a literal pattern in a RegExp. */
export function escapeRegex(str: string): string {
  return str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
