import { AgentType } from "../core/index.js";

export interface DelegationRequest {
  readonly objective: string;
  readonly laneLabel?: string;
  readonly scope?: string;
  readonly constraints?: ReadonlyArray<string>;
  readonly exclusions?: ReadonlyArray<string>;
  readonly successCriteria?: ReadonlyArray<string>;
  readonly parentContext?: string;
}

export interface AgentMeta {
  readonly agentId: string;
  readonly parentId: string | null;
  readonly depth: number;
  readonly agentType: AgentType;
}

export function buildExplorationLaneRequest(input: {
  readonly objective: string;
  readonly laneLabel: string;
  readonly scope?: string;
  readonly exclusions?: ReadonlyArray<string>;
  readonly successCriteria?: ReadonlyArray<string>;
  readonly parentContext?: string;
}): DelegationRequest {
  return {
    objective: input.objective.trim(),
    laneLabel: input.laneLabel.trim(),
    ...(input.scope?.trim() ? { scope: input.scope.trim() } : {}),
    ...(input.exclusions && input.exclusions.length > 0 ? { exclusions: input.exclusions } : {}),
    ...(input.successCriteria && input.successCriteria.length > 0
      ? { successCriteria: input.successCriteria }
      : {}),
    ...(input.parentContext?.trim() ? { parentContext: input.parentContext.trim() } : {}),
  };
}

export function normalizeDelegationRequest(
  request: unknown,
  legacyTask: unknown,
): DelegationRequest | null {
  if (request && typeof request === "object") {
    const raw = request as Record<string, unknown>;
    const objective = raw["objective"];
    if (typeof objective === "string" && objective.trim().length > 0) {
      return {
        objective: objective.trim(),
        ...(typeof raw["laneLabel"] === "string" && raw["laneLabel"].trim().length > 0
          ? { laneLabel: raw["laneLabel"].trim() }
          : {}),
        ...(typeof raw["scope"] === "string" && raw["scope"].trim().length > 0
          ? { scope: raw["scope"].trim() }
          : {}),
        ...(Array.isArray(raw["constraints"])
          ? { constraints: raw["constraints"].filter((item): item is string => typeof item === "string" && item.trim().length > 0) }
          : {}),
        ...(Array.isArray(raw["exclusions"])
          ? { exclusions: raw["exclusions"].filter((item): item is string => typeof item === "string" && item.trim().length > 0) }
          : {}),
        ...(Array.isArray(raw["successCriteria"])
          ? { successCriteria: raw["successCriteria"].filter((item): item is string => typeof item === "string" && item.trim().length > 0) }
          : {}),
        ...(typeof raw["parentContext"] === "string" && raw["parentContext"].trim().length > 0
          ? { parentContext: raw["parentContext"].trim() }
          : {}),
      };
    }
  }

  if (typeof legacyTask === "string" && legacyTask.trim().length > 0) {
    return { objective: legacyTask.trim() };
  }

  return null;
}

export function buildDelegationQuery(
  request: DelegationRequest,
  maxIterations: number,
): string {
  const lines = [
    "This task was intentionally delegated for focused sub-work. Complete only this delegated objective and return a structured result.",
    `Objective: ${request.objective}`,
    `Iteration budget: ${maxIterations}`,
    "Treat the iteration budget, scope, constraints, and success criteria as hard limits.",
  ];
  if (request.laneLabel) {
    lines.push(`Lane: ${request.laneLabel}`);
  }
  if (request.scope) {
    lines.push(`Scope: ${request.scope}`);
  }
  if (request.parentContext) {
    lines.push(`Delegated because: ${request.parentContext}`);
  }
  if (request.constraints && request.constraints.length > 0) {
    lines.push(`Constraints:\n- ${request.constraints.join("\n- ")}`);
  }
  if (request.exclusions && request.exclusions.length > 0) {
    lines.push(`Out of scope:\n- ${request.exclusions.join("\n- ")}`);
  }
  if (request.successCriteria && request.successCriteria.length > 0) {
    lines.push(`Success criteria:\n- ${request.successCriteria.join("\n- ")}`);
  }
  return lines.join("\n\n");
}

function parseJSONObject(text: string): Record<string, unknown> | null {
  const trimmed = text.trim();
  if (trimmed.length === 0) return null;

  const candidates = [trimmed];
  const fenceMatch = trimmed.match(/```json\s*([\s\S]*?)```/i);
  if (fenceMatch?.[1]) {
    candidates.unshift(fenceMatch[1].trim());
  }
  const leadingObject = extractLeadingJSONObject(trimmed);
  if (leadingObject) {
    candidates.unshift(leadingObject);
  }

  for (const candidate of candidates) {
    try {
      const parsed = JSON.parse(candidate);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        return parsed as Record<string, unknown>;
      }
    } catch {
      // Ignore and continue.
    }
  }

  return null;
}

function extractLeadingJSONObject(text: string): string | null {
  const trimmed = text.trimStart();
  if (!trimmed.startsWith("{")) return null;

  let depth = 0;
  let inString = false;
  let escaped = false;

  for (let index = 0; index < trimmed.length; index++) {
    const char = trimmed[index]!;

    if (inString) {
      if (escaped) {
        escaped = false;
        continue;
      }
      if (char === "\\") {
        escaped = true;
        continue;
      }
      if (char === "\"") {
        inString = false;
      }
      continue;
    }

    if (char === "\"") {
      inString = true;
      continue;
    }
    if (char === "{") {
      depth++;
      continue;
    }
    if (char === "}") {
      depth--;
      if (depth === 0) {
        return trimmed.slice(0, index + 1);
      }
    }
  }

  return null;
}

export function parseStructuredAgentOutput(
  agentType: AgentType,
  text: string,
): Record<string, unknown> | null {
  const parsed = parseJSONObject(text);
  if (!parsed) return null;

  const requiredKeys: Record<AgentType, ReadonlyArray<string>> = {
    [AgentType.EXPLORE]: ["answer", "evidence", "relatedFiles", "unresolved"],
    [AgentType.REVIEWER]: ["findings", "openQuestions", "summary"],
    [AgentType.ARCHITECT]: ["steps", "risks", "assumptions", "summary"],
    [AgentType.GENERAL]: ["summary", "filesTouched", "checksRun", "unresolved"],
  };

  return requiredKeys[agentType].every((key) => key in parsed) ? parsed : null;
}
