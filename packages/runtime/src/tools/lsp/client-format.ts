import type { DocumentSymbol, SymbolInformation } from "vscode-languageserver-protocol";

export interface DiagnosticResult {
  readonly file: string;
  readonly diagnostics: ReadonlyArray<{
    readonly line: number;
    readonly character: number;
    readonly message: string;
    readonly severity: string;
  }>;
}

export interface SymbolResult {
  readonly name: string;
  readonly kind: string;
  readonly line: number;
  readonly character: number;
  readonly containerName?: string;
}

export interface LocationResult {
  readonly file: string;
  readonly line: number;
  readonly character: number;
}

export interface WorkspaceSymbolResult extends SymbolResult {
  readonly file: string;
}

export interface CallHierarchyResult {
  readonly name: string;
  readonly kind: string;
  readonly file: string;
  readonly line: number;
  readonly character: number;
  readonly detail?: string;
}

interface LspLocationLike {
  readonly uri?: string;
  readonly range?: { readonly start?: { readonly line?: number; readonly character?: number } };
  readonly targetUri?: string;
  readonly targetSelectionRange?: { readonly start?: { readonly line?: number; readonly character?: number } };
  readonly targetRange?: { readonly start?: { readonly line?: number; readonly character?: number } };
}

interface LspWorkspaceSymbolLike {
  readonly name?: unknown;
  readonly kind?: unknown;
  readonly containerName?: unknown;
  readonly location?: {
    readonly uri?: string;
    readonly range?: { readonly start?: { readonly line?: number; readonly character?: number } };
  };
}

interface LspCallHierarchyItemLike {
  readonly name?: unknown;
  readonly kind?: unknown;
  readonly uri?: string;
  readonly selectionRange?: { readonly start?: { readonly line?: number; readonly character?: number } };
  readonly range?: { readonly start?: { readonly line?: number; readonly character?: number } };
  readonly detail?: unknown;
}

export function normalizeLocations(result: unknown, rootPath: string): ReadonlyArray<LocationResult> {
  if (!result) return [];
  const locations = Array.isArray(result) ? result : [result];
  return locations.map((loc) => locationToResult(loc, rootPath)).filter(isPresent);
}

function locationToResult(value: unknown, rootPath: string): LocationResult | null {
  if (!isObject(value)) return null;
  const loc = value as LspLocationLike;
  const uri = loc.targetUri ?? loc.uri;
  const range = loc.targetSelectionRange ?? loc.targetRange ?? loc.range;
  const start = range?.start;
  if (!uri || start?.line === undefined || start.character === undefined) return null;
  return {
    file: formatUriPath(uri, rootPath),
    line: start.line + 1,
    character: start.character + 1,
  };
}

function formatUriPath(uri: string, rootPath: string): string {
  const rootPrefix = `file://${rootPath.replace(/\/$/, "")}/`;
  if (uri.startsWith(rootPrefix)) return decodeURIComponent(uri.slice(rootPrefix.length));
  if (uri.startsWith("file://")) return decodeURIComponent(uri.slice("file://".length));
  return uri;
}

export function formatHoverContents(result: unknown): string | null {
  if (!isObject(result)) return null;
  const contents = (result as { readonly contents?: unknown }).contents;
  return formatMarkedContent(contents);
}

function formatMarkedContent(value: unknown): string | null {
  if (value === null || value === undefined) return null;
  if (typeof value === "string") return value.trim() || null;
  if (Array.isArray(value)) return joinMarkedContent(value);
  if (!isObject(value)) return String(value);
  return formatMarkedContentObject(value);
}

function joinMarkedContent(value: ReadonlyArray<unknown>): string | null {
  const parts = value.map(formatMarkedContent).filter((part): part is string => Boolean(part));
  return parts.length > 0 ? parts.join("\n\n") : null;
}

function formatMarkedContentObject(value: object): string | null {
  const content = value as { readonly value?: unknown; readonly language?: unknown };
  if (typeof content.value !== "string") return null;
  const text = content.value.trim();
  if (!text) return null;
  if (typeof content.language !== "string" || content.language.length === 0) return text;
  return `\`\`\`${content.language}\n${text}\n\`\`\``;
}

export function workspaceSymbolToResult(value: unknown, rootPath: string): WorkspaceSymbolResult | null {
  if (!isObject(value)) return null;
  const symbol = value as LspWorkspaceSymbolLike;
  const location = readWorkspaceSymbolLocation(symbol);
  if (!isValidNamedSymbol(symbol) || !location) return null;
  return {
    name: symbol.name,
    kind: symbolKindToString(symbol.kind),
    file: formatUriPath(location.uri, rootPath),
    line: (location.line ?? 0) + 1,
    character: (location.character ?? 0) + 1,
    ...(typeof symbol.containerName === "string" ? { containerName: symbol.containerName } : {}),
  };
}

export function callHierarchyItemToResult(value: unknown, rootPath: string): CallHierarchyResult | null {
  if (!isObject(value)) return null;
  const item = value as LspCallHierarchyItemLike;
  const location = readCallHierarchyLocation(item);
  if (!isValidNamedSymbol(item) || !location) return null;
  return {
    name: item.name,
    kind: symbolKindToString(item.kind),
    file: formatUriPath(location.uri, rootPath),
    line: (location.line ?? 0) + 1,
    character: (location.character ?? 0) + 1,
    ...(typeof item.detail === "string" && item.detail.length > 0 ? { detail: item.detail } : {}),
  };
}

function readWorkspaceSymbolLocation(
  symbol: LspWorkspaceSymbolLike,
): { readonly uri: string; readonly line?: number; readonly character?: number } | null {
  const start = symbol.location?.range?.start;
  if (!symbol.location?.uri || !start) return null;
  return { uri: symbol.location.uri, line: start.line, character: start.character };
}

function readCallHierarchyLocation(
  item: LspCallHierarchyItemLike,
): { readonly uri: string; readonly line?: number; readonly character?: number } | null {
  const start = item.selectionRange?.start ?? item.range?.start;
  if (!item.uri || !start) return null;
  return { uri: item.uri, line: start.line, character: start.character };
}

function isValidNamedSymbol<T extends { readonly name?: unknown; readonly kind?: unknown }>(
  value: T,
): value is T & { readonly name: string; readonly kind: number } {
  return typeof value.name === "string" && typeof value.kind === "number";
}

export function severityToString(severity: number | undefined): string {
  switch (severity) {
    case 1: return "error";
    case 2: return "warning";
    case 3: return "info";
    case 4: return "hint";
    case undefined: return "unknown";
    default: return "unknown";
  }
}

export function flattenSymbols(
  symbols: Array<SymbolInformation | DocumentSymbol>,
  containerName?: string,
): SymbolResult[] {
  const results: SymbolResult[] = [];
  for (const sym of symbols) {
    results.push(...flattenSymbol(sym, containerName));
  }
  return results;
}

function flattenSymbol(
  sym: SymbolInformation | DocumentSymbol,
  containerName?: string,
): SymbolResult[] {
  if (!("range" in sym)) {
    return [{
      name: sym.name,
      kind: symbolKindToString(sym.kind),
      line: sym.location.range.start.line + 1,
      character: sym.location.range.start.character + 1,
      containerName: sym.containerName,
    }];
  }
  return [
    {
      name: sym.name,
      kind: symbolKindToString(sym.kind),
      line: sym.range.start.line + 1,
      character: sym.range.start.character + 1,
      containerName,
    },
    ...flattenSymbols(sym.children ?? [], sym.name),
  ];
}

function symbolKindToString(kind: number): string {
  const kinds: Record<number, string> = {
    1: "file", 2: "module", 3: "namespace", 4: "package",
    5: "class", 6: "method", 7: "property", 8: "field",
    9: "constructor", 10: "enum", 11: "interface", 12: "function",
    13: "variable", 14: "constant", 15: "string", 16: "number",
    17: "boolean", 18: "array", 19: "object", 20: "key",
    21: "null", 22: "enum_member", 23: "struct", 24: "event",
    25: "operator", 26: "type_parameter",
  };
  return kinds[kind] ?? `kind_${kind}`;
}

function isObject(value: unknown): value is object {
  return Boolean(value) && typeof value === "object";
}

function isPresent<T>(value: T | null | undefined): value is T {
  return value !== null && value !== undefined;
}
