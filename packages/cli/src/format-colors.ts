function isColorEnabled(): boolean {
  return !process.env["NO_COLOR"];
}

function wrap(code: string, text: string): string {
  return isColorEnabled() ? `\x1b[${code}m${text}\x1b[0m` : text;
}

export function dim(text: string): string { return wrap("90", text); }
export function red(text: string): string { return wrap("31", text); }
export function green(text: string): string { return wrap("32", text); }
export function yellow(text: string): string { return wrap("33", text); }
export function cyan(text: string): string { return wrap("36", text); }
export function bold(text: string): string { return wrap("1", text); }
export function dimBold(text: string): string {
  return isColorEnabled() ? `\x1b[90;1m${text}\x1b[0m` : text;
}

export function truncate(text: string, maxLen: number): string {
  if (text.length <= maxLen) return text;
  return text.substring(0, maxLen - 1) + "…";
}
