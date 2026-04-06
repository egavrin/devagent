/**
 * Approval system for interactive safety presets and legacy executor modes.
 *
 * Modern interactive presets:
 * - default: auto-allow in-repo edits and safe repo commands, ask at trust boundaries
 * - autopilot: allow all actions
 *
 * Legacy executor modes remain supported for machine-contract compatibility.
 */

import { isAbsolute, resolve } from "node:path";
import type {
  ApprovalPolicy,
  ToolCategory,
} from "./types.js";
import { ApprovalMode, SafetyMode } from "./types.js";
import { ApprovalDeniedError } from "./errors.js";
import type { EventBus } from "./events.js";

// ─── Decision Types ──────────────────────────────────────────

export type ApprovalDecision = "allow" | "deny" | "ask";

export interface ApprovalRequest {
  readonly toolName: string;
  readonly toolCategory: ToolCategory;
  readonly filePath: string | null;
  readonly description: string;
  readonly repoRoot?: string;
  readonly arguments?: Readonly<Record<string, unknown>>;
}

export interface ApprovalResult {
  readonly approved: boolean;
  readonly reason: string;
  readonly requiredUserInput: boolean;
}

// ─── Approval Gate ──────────────────────────────────────────

export class ApprovalGate {
  private readonly policy: ApprovalPolicy;
  private readonly bus: EventBus | null;
  private currentMode: ApprovalMode | SafetyMode;
  private userResponseResolver:
    | ((response: { approved: boolean; feedback?: string }) => void)
    | null = null;
  private readonly sessionAllowRules = new Set<string>();

  constructor(policy: ApprovalPolicy, bus?: EventBus) {
    this.policy = policy;
    this.bus = bus ?? null;
    this.currentMode = policy.mode;

    // Listen for user approval responses
    if (this.bus) {
      this.bus.on("approval:response", (event) => {
        if (this.userResponseResolver) {
          this.userResponseResolver({
            approved: event.approved,
            feedback: event.feedback,
          });
          this.userResponseResolver = null;
        }
      });
    }
  }

  /**
   * Check if a tool execution should proceed.
   * Returns immediately for allow/deny decisions.
   * For "ask" decisions, emits an approval:request event and waits for response.
   */
  async check(request: ApprovalRequest): Promise<ApprovalResult> {
    const decision = this.decide(request);

    if (decision === "allow") {
      this.logAudit(request, true, "auto-approved");
      return { approved: true, reason: "auto-approved", requiredUserInput: false };
    }

    if (decision === "deny") {
      this.logAudit(request, false, "auto-denied");
      return { approved: false, reason: "auto-denied", requiredUserInput: false };
    }

    // "ask" — request user approval via event bus
    if (!this.bus) {
      // No event bus = no way to ask user, fail fast
      throw new ApprovalDeniedError(
        request.toolName,
        "No event bus available to request user approval",
      );
    }

    const requestId = `approval-${Date.now()}`;
    this.bus.emit("approval:request", {
      id: requestId,
      action: request.toolName,
      toolName: request.toolName,
      details: request.description,
    });

    const response = await this.waitForUserResponse();
    if (response.approved && response.feedback === "session") {
      this.sessionAllowRules.add(this.getSessionRuleKey(request));
    }
    const approved = response.approved;
    const reason = approved ? "user-approved" : "user-denied";
    this.logAudit(request, approved, reason);

    if (!approved) {
      return { approved: false, reason: "user-denied", requiredUserInput: true };
    }

    return { approved: true, reason: "user-approved", requiredUserInput: true };
  }

  /**
   * Synchronous decision based on policy — determines allow/deny/ask
   * without waiting for user input.
   */
  decide(request: ApprovalRequest): ApprovalDecision {
    // 1. Check per-tool overrides first (highest priority)
    const toolOverride = this.policy.toolOverrides[request.toolName];
    if (toolOverride) return toolOverride;

    if (this.sessionAllowRules.has(this.getSessionRuleKey(request))) {
      return "allow";
    }

    // 2. Check per-path rules
    if (request.filePath) {
      const pathDecision = this.checkPathRules(request.filePath);
      if (pathDecision) return pathDecision;
    }

    // 3. Apply mode-based rules
    return this.decideByMode(request);
  }

  private checkPathRules(filePath: string): ApprovalDecision | null {
    for (const rule of this.policy.pathRules) {
      if (matchPath(filePath, rule.pattern)) {
        return rule.action;
      }
    }
    return null;
  }

  private decideByMode(request: ApprovalRequest): ApprovalDecision {
    if (this.shouldUseLegacyMode()) {
      return this.decideByLegacyMode(request.toolCategory);
    }

    const normalized = this.normalizeRequest(request);
    const approvalPolicy = this.policy.approvalPolicy ?? "on-request";
    const sandboxMode = this.policy.sandboxMode ?? "workspace-write";
    const networkAccess = this.policy.networkAccess ?? "off";

    if (request.toolCategory === "state") {
      return "allow";
    }

    if (request.toolCategory === "external") {
      return approvalPolicy === "never" || networkAccess === "on" ? "allow" : "ask";
    }

    if (normalized.hasSensitivePath) {
      return approvalPolicy === "never" ? "allow" : "ask";
    }

    if (request.toolCategory === "readonly") {
      if (normalized.filePath && normalized.isOutsideRepo) {
        return approvalPolicy === "never" ? "allow" : "ask";
      }
      return "allow";
    }

    if (request.toolCategory === "mutating") {
      if (request.toolName === "git_commit") {
        return approvalPolicy === "never" ? "allow" : "ask";
      }
      if (approvalPolicy === "never") {
        return "allow";
      }
      if (approvalPolicy === "strict") {
        return "ask";
      }
      if (sandboxMode === "danger-full-access") {
        return "allow";
      }
      if (normalized.filePath && normalized.isInsideRepo && !normalized.hasSensitivePath) {
        return "allow";
      }
      return "ask";
    }

    if (request.toolCategory === "workflow") {
      const command = normalized.command;
      if (approvalPolicy === "never") {
        return "allow";
      }
      if (approvalPolicy === "strict") {
        return "ask";
      }
      if (!command || !normalized.commandRunsInRepo) {
        return "ask";
      }
      return isDefaultSafeCommand(command) ? "allow" : "ask";
    }

    return "ask";
  }

  private shouldUseLegacyMode(): boolean {
    switch (this.currentMode) {
      case ApprovalMode.SUGGEST:
      case ApprovalMode.AUTO_EDIT:
      case ApprovalMode.FULL_AUTO:
        return true;
      default:
        return false;
    }
  }

  private decideByLegacyMode(category: ToolCategory): ApprovalDecision {
    switch (this.currentMode) {
      case ApprovalMode.SUGGEST:
        return this.decideSuggestMode(category);
      case ApprovalMode.AUTO_EDIT:
        return this.decideAutoEditMode(category);
      case ApprovalMode.FULL_AUTO:
        return this.decideFullAutoMode(category);
      default:
        return "ask";
    }
  }

  private decideSuggestMode(category: ToolCategory): ApprovalDecision {
    switch (category) {
      case "readonly":
        return "allow";
      case "state":
        return "allow"; // internal agent state, no workspace mutation
      case "mutating":
        return "ask"; // show diff, ask user
      case "workflow":
        return "ask";
      case "external":
        return "deny"; // network blocked in legacy suggest mode
      default:
        return "ask";
    }
  }

  private decideAutoEditMode(category: ToolCategory): ApprovalDecision {
    switch (category) {
      case "readonly":
        return "allow";
      case "state":
        return "allow"; // internal agent state, no workspace mutation
      case "mutating":
        return "allow"; // auto-approve file writes
      case "workflow":
        return "ask"; // shell commands still shown
      case "external":
        return "deny"; // network blocked
      default:
        return "ask";
    }
  }

  private decideFullAutoMode(category: ToolCategory): ApprovalDecision {
    // Everything allowed in legacy full-auto mode
    return "allow";
  }

  private waitForUserResponse(): Promise<{ approved: boolean; feedback?: string }> {
    return new Promise<{ approved: boolean; feedback?: string }>((resolve) => {
      this.userResponseResolver = resolve;
    });
  }

  private logAudit(
    request: ApprovalRequest,
    approved: boolean,
    reason: string,
  ): void {
    if (!this.policy.auditLog) return;
    if (!this.bus) return;

    this.bus.emit("tool:before", {
      name: `audit:${request.toolName}`,
      callId: `audit-${Date.now()}`,
      params: {
        approved,
        reason,
        filePath: request.filePath,
        description: request.description,
        mode: this.currentMode,
      },
    });
  }

  getMode(): ApprovalMode | SafetyMode {
    return this.currentMode;
  }

  setMode(mode: ApprovalMode | SafetyMode): void {
    this.currentMode = mode;
    this.sessionAllowRules.clear();
  }

  private getSessionRuleKey(request: ApprovalRequest): string {
    const normalized = this.normalizeRequest(request);
    if (request.toolName === "run_command" && normalized.command) {
      return `run_command:${normalized.command}`;
    }
    if (normalized.filePath) {
      return `${request.toolName}:${normalized.filePath}`;
    }
    return `${request.toolName}:${request.description}`;
  }

  private normalizeRequest(request: ApprovalRequest): {
    readonly filePath: string | null;
    readonly isInsideRepo: boolean;
    readonly isOutsideRepo: boolean;
    readonly hasSensitivePath: boolean;
    readonly command: string | null;
    readonly commandRunsInRepo: boolean;
  } {
    const repoRoot = request.repoRoot ? resolve(request.repoRoot) : null;
    const rawFilePath = request.filePath;
    const normalizedFilePath = rawFilePath && repoRoot
      ? resolvePath(rawFilePath, repoRoot)
      : rawFilePath;
    const isInsideRepo = normalizedFilePath && repoRoot
      ? isSubpath(normalizedFilePath, repoRoot)
      : false;
    const command = typeof request.arguments?.["command"] === "string"
      ? normalizeCommand(request.arguments["command"] as string)
      : null;
    const commandCwd = typeof request.arguments?.["cwd"] === "string"
      ? request.arguments["cwd"] as string
      : ".";
    const resolvedCommandCwd = repoRoot ? resolvePath(commandCwd, repoRoot) : null;
    const commandRunsInRepo = resolvedCommandCwd && repoRoot
      ? isSubpath(resolvedCommandCwd, repoRoot)
      : true;

    return {
      filePath: normalizedFilePath ?? null,
      isInsideRepo,
      isOutsideRepo: Boolean(normalizedFilePath && repoRoot && !isInsideRepo),
      hasSensitivePath: Boolean(normalizedFilePath && isSensitivePath(normalizedFilePath)),
      command,
      commandRunsInRepo,
    };
  }
}

// ─── Path Matching ──────────────────────────────────────────

/**
 * Simple glob-style path matching.
 * Supports: * (any segment), ** (any depth), ? (single char).
 */
function matchPath(filePath: string, pattern: string): boolean {
  // Convert glob to regex
  const regexStr = pattern
    .replace(/\./g, "\\.")
    .replace(/\*\*/g, "{{GLOBSTAR}}")
    .replace(/\*/g, "[^/]*")
    .replace(/\?/g, "[^/]")
    .replace(/\{\{GLOBSTAR\}\}/g, ".*");

  const regex = new RegExp(`^${regexStr}$`);
  return regex.test(filePath);
}

function resolvePath(pathValue: string, repoRoot: string): string {
  return isAbsolute(pathValue) ? resolve(pathValue) : resolve(repoRoot, pathValue);
}

function isSubpath(targetPath: string, rootPath: string): boolean {
  const normalizedRoot = rootPath.endsWith("/") ? rootPath : `${rootPath}/`;
  return targetPath === rootPath || targetPath.startsWith(normalizedRoot);
}

function isSensitivePath(filePath: string): boolean {
  const lowered = filePath.toLowerCase();
  return (
    lowered.includes("/.ssh/") ||
    lowered.includes("/.aws/") ||
    lowered.includes("/.config/") ||
    lowered.endsWith("/.npmrc") ||
    lowered.endsWith("/.pypirc") ||
    lowered.endsWith("/.git-credentials") ||
    lowered.endsWith("/credentials") ||
    lowered.endsWith("/config") ||
    lowered.includes("/secrets") ||
    lowered.endsWith("/id_rsa") ||
    lowered.endsWith("/id_ed25519") ||
    lowered.includes("/.env")
  );
}

function normalizeCommand(command: string): string {
  return command.trim().replace(/\s+/g, " ");
}

function isDefaultSafeCommand(command: string): boolean {
  const lower = command.toLowerCase();

  if (
    lower.includes("&&") ||
    lower.includes("||") ||
    lower.includes(";") ||
    lower.includes("|") ||
    lower.includes("`") ||
    lower.includes("$(") ||
    lower.includes(">") ||
    lower.includes("<")
  ) {
    return false;
  }

  if (
    /\b(npm|pnpm|yarn|bun)\s+(install|add|remove|unlink|update|upgrade)\b/.test(lower) ||
    /\b(rm|chmod|chown|sudo|curl|wget|scp|ssh|mv|cp|tee|launchctl|systemctl)\b/.test(lower) ||
    /\bsed\s+-i\b/.test(lower) ||
    /\bgit\s+(commit|push|tag|merge|rebase|reset|checkout\s+-b|switch\s+-c|branch\s+-d|branch\s+-D|publish)\b/.test(lower)
  ) {
    return false;
  }

  return (
    /^(ls|pwd|cat|head|tail|wc|rg|grep|find)\b/.test(lower) ||
    /^sed\s+-n\b/.test(lower) ||
    /^(bun|npm|pnpm|yarn)\s+(test|run test|run lint|run build|run dev|run typecheck|check|check:oss)\b/.test(lower) ||
    /^(pytest|go test|cargo (test|check|build)|vitest|jest)\b/.test(lower) ||
    /^git\s+(status|diff|log|show|rev-parse|remote -v|branch --show-current)\b/.test(lower)
  );
}
