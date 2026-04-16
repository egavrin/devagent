import { basename } from "node:path";
import type { Session } from "@devagent/runtime";

export interface SessionPreview {
  readonly id: string;
  readonly updatedAt: number;
  readonly title: string;
  readonly repoLabel: string;
  readonly cost?: number;
}

const DEFAULT_TITLE = "Interactive session";
const DEFAULT_REPO_LABEL = "unknown repo";
const FILE_QUERY_TITLE = "Prompt from file";
const UNTITLED_TITLE = "Untitled session";
const TITLE_MAX_LENGTH = 48;
const PLACEHOLDER_QUERY = /^\(?interactive query\)?$/i;
const FILE_QUERY = /^\(?file query\)?$/i;
const GREETING_ONLY = /^(hi|hello|hey|yo|sup|hola|bonjour)[!. ]*$/i;
const SIMPLE_MATH = /^[\d\s()+\-*/%=?.]+$/;

export function deriveSessionTitle(input: string | null | undefined): string {
  if (!input) {
    return DEFAULT_TITLE;
  }

  let title = input.replace(/\s+/g, " ").trim();
  if (!title || PLACEHOLDER_QUERY.test(title)) {
    return DEFAULT_TITLE;
  }
  if (FILE_QUERY.test(title)) {
    return FILE_QUERY_TITLE;
  }

  while (hasMatchingOuterQuotes(title)) {
    title = title.slice(1, -1).trim();
  }

  title = title.replace(/[!?,.;:\s]+$/g, "").trim();
  if (!title || PLACEHOLDER_QUERY.test(title)) {
    return DEFAULT_TITLE;
  }
  if (FILE_QUERY.test(title)) {
    return FILE_QUERY_TITLE;
  }

  if (title.length <= TITLE_MAX_LENGTH) {
    return title;
  }

  return `${title.slice(0, TITLE_MAX_LENGTH - 3).trimEnd()}...`;
}

function deriveRepoLabel(
  repoLabel: string | null | undefined,
  repoRoot?: string | null | undefined,
): string {
  const normalizedLabel = normalizeRepoLabel(repoLabel);
  if (normalizedLabel) {
    return normalizedLabel;
  }

  if (repoRoot && repoRoot.trim()) {
    const fromPath = basename(repoRoot.trim());
    if (fromPath) {
      return fromPath;
    }
  }

  return DEFAULT_REPO_LABEL;
}

export function buildSessionPreview(
  session: Pick<Session, "id" | "updatedAt" | "metadata"> & Partial<Pick<Session, "messages">>,
): SessionPreview {
  const meta = asRecord(session.metadata);
  const title = chooseBestSessionTitle(
    getString(meta?.["title"]),
    getString(meta?.["query"]),
    session.messages,
  );
  const repoLabel = deriveRepoLabel(getString(meta?.["repoLabel"]), getString(meta?.["repoRoot"]));
  const cost = typeof meta?.["totalCost"] === "number" ? meta["totalCost"] as number : undefined;

  return {
    id: session.id,
    updatedAt: session.updatedAt,
    title,
    repoLabel,
    cost,
  };
}

function shortSessionId(sessionId: string): string {
  return sessionId.slice(0, 8);
}

export function formatRelativeUpdatedAt(updatedAt: number, now: number = Date.now()): string {
  const deltaMs = Math.max(0, now - updatedAt);
  const minute = 60 * 1000;
  const hour = 60 * minute;
  const day = 24 * hour;
  const week = 7 * day;

  if (deltaMs < minute) {
    return "just now";
  }
  if (deltaMs < hour) {
    return `${Math.floor(deltaMs / minute)}m ago`;
  }
  if (deltaMs < day) {
    return `${Math.floor(deltaMs / hour)}h ago`;
  }
  if (deltaMs < week) {
    return `${Math.floor(deltaMs / day)}d ago`;
  }
  return `${Math.floor(deltaMs / week)}w ago`;
}

export function renderSessionPreview(preview: SessionPreview, now: number = Date.now()): string {
  const subtitleParts = [
    shortSessionId(preview.id),
    formatRelativeUpdatedAt(preview.updatedAt, now),
  ];
  if (preview.repoLabel !== DEFAULT_REPO_LABEL) {
    subtitleParts.splice(1, 0, preview.repoLabel);
  }
  if (preview.cost !== undefined) {
    subtitleParts.push(`$${preview.cost.toFixed(4)}`);
  }

  return `  ${preview.title}\n    ${subtitleParts.join("  ")}`;
}

export function formatResumeCandidate(session: Pick<Session, "id" | "updatedAt" | "metadata">): string {
  const preview = buildSessionPreview(session);
  return `${shortSessionId(preview.id)}  ${preview.title}  ${preview.repoLabel}`;
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
  return value && typeof value === "object" ? value as Record<string, unknown> : undefined;
}

function getString(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

function normalizeRepoLabel(value: string | null | undefined): string | undefined {
  if (!value) {
    return undefined;
  }
  const trimmed = value.trim();
  return trimmed || undefined;
}

function hasMatchingOuterQuotes(value: string): boolean {
  if (value.length < 2) {
    return false;
  }

  const first = value[0];
  const last = value[value.length - 1];
  return (first === "\"" && last === "\"")
    || (first === "'" && last === "'")
    || (first === "`" && last === "`");
}

function chooseBestSessionTitle(
  storedTitle: string | undefined,
  storedQuery: string | undefined,
  messages: ReadonlyArray<Session["messages"][number]> | undefined,
): string {
  const preferred = deriveSessionTitle(storedTitle ?? storedQuery);
  if (!isLowSignalTitle(preferred)) {
    return preferred;
  }

  const recovered = findSubstantiveUserMessage(messages);
  if (recovered) {
    return recovered;
  }

  if (preferred === FILE_QUERY_TITLE) {
    return FILE_QUERY_TITLE;
  }
  if (preferred === DEFAULT_TITLE) {
    return DEFAULT_TITLE;
  }
  return UNTITLED_TITLE;
}

function findSubstantiveUserMessage(
  messages: ReadonlyArray<Session["messages"][number]> | undefined,
): string | null {
  if (!messages || messages.length === 0) {
    return null;
  }

  for (const message of messages) {
    if (message.role !== "user") {
      continue;
    }
    const title = deriveSessionTitle(message.content);
    if (!isLowSignalTitle(title)) {
      return title;
    }
  }

  return null;
}

function isLowSignalTitle(title: string): boolean {
  if (title === DEFAULT_TITLE || title === FILE_QUERY_TITLE) {
    return true;
  }
  if (title.length <= 2) {
    return true;
  }
  if (GREETING_ONLY.test(title)) {
    return true;
  }
  if (SIMPLE_MATH.test(title) && title.length <= 12) {
    return true;
  }
  return false;
}
