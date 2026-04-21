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

import { ApprovalDeniedError } from "./errors.js";
import type { EventBus } from "./events.js";
import type {
  ApprovalPolicy,
  ApprovalPolicyMode,
  ToolCategory,
} from "./types.js";
import { ApprovalMode, SafetyMode } from "./types.js";

// ─── Decision Types ──────────────────────────────────────────

export type ApprovalDecision = "allow" | "deny" | "ask";

interface NormalizedApprovalRequest {
  readonly filePath: string | null;
  readonly isInsideRepo: boolean;
  readonly isOutsideRepo: boolean;
  readonly hasSensitivePath: boolean;
  readonly command: string | null;
  readonly commandRunsInRepo: boolean;
}

const SENSITIVE_PATH_MATCHERS: ReadonlyArray<(path: string) => boolean> = [
  (path) => path.includes("/.ssh/"),
  (path) => path.includes("/.aws/"),
  (path) => path.includes("/.config/"),
  (path) => path.endsWith("/.npmrc"),
  (path) => path.endsWith("/.pypirc"),
  (path) => path.endsWith("/.git-credentials"),
  (path) => path.endsWith("/credentials"),
  (path) => path.endsWith("/config"),
  (path) => path.includes("/secrets"),
  (path) => path.endsWith("/id_rsa"),
  (path) => path.endsWith("/id_ed25519"),
  (path) => path.includes("/.env"),
];

const UNSAFE_COMMAND_PATTERNS: ReadonlyArray<RegExp> = [
  /\b(npm|pnpm|yarn|bun)\s+(install|add|remove|unlink|update|upgrade)\b/,
  /\b(rm|chmod|chown|sudo|curl|wget|scp|ssh|mv|cp|tee|launchctl|systemctl)\b/,
  /\bsed\s+-i\b/,
  /\bgit\s+(commit|push|tag|merge|rebase|reset|checkout\s+-b|switch\s+-c|branch\s+-d|branch\s+-D|publish)\b/,
];

const SAFE_COMMAND_PATTERNS: ReadonlyArray<RegExp> = [
  /^(ls|pwd|cat|head|tail|wc|rg|grep|find)\b/,
  /^sed\s+-n\b/,
  /^(bun|npm|pnpm|yarn)\s+(test|run test|run lint|run build|run dev|run typecheck|check|check:oss)\b/,
  /^(pytest|go test|cargo (test|check|build)|vitest|jest)\b/,
  /^git\s+(status|diff|log|show|rev-parse|remote -v|branch --show-current)\b/,
];

const SHELL_OPERATOR_PATTERNS: ReadonlyArray<string> = [
  "&&",
  "||",
  ";",
  "|",
  "`",
  "$(",
  ">",
  "<",
];

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
  private activePolicy: ApprovalPolicy;
  private userResponseResolver:
    | ((response: { approved: boolean; feedback?: string }) => void)
    | null = null;
  private readonly sessionAllowRules = new Set<string>();

  constructor(policy: ApprovalPolicy, bus?: EventBus) {
    this.policy = policy;
    this.bus = bus ?? null;
    this.currentMode = policy.mode;
    this.activePolicy = policy;

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
    const toolOverride = this.activePolicy.toolOverrides[request.toolName];
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
    for (const rule of this.activePolicy.pathRules) {
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
    const approvalPolicy = this.activePolicy.approvalPolicy ?? "on-request";
    const sandboxMode = this.activePolicy.sandboxMode ?? "workspace-write";
    const networkAccess = this.activePolicy.networkAccess ?? "off";

    if (request.toolCategory === "state") {
      return "allow";
    }

    return decideModernRequest({
      request,
      normalized,
      approvalPolicy,
      sandboxMode,
      networkAccess,
    });
  }

  private shouldUseLegacyMode(): boolean {
    switch (this.currentMode) {
      case ApprovalMode.SUGGEST:
      case ApprovalMode.AUTO_EDIT:
      case ApprovalMode.FULL_AUTO:
        return true;
      case SafetyMode.DEFAULT:
      case SafetyMode.AUTOPILOT:
        return false;
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
      case SafetyMode.DEFAULT:
      case SafetyMode.AUTOPILOT:
        return "ask";
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

  private decideFullAutoMode(_category: ToolCategory): ApprovalDecision {
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
    this.activePolicy = this.getPolicyForMode(mode);
    this.sessionAllowRules.clear();
  }

  private getPolicyForMode(mode: ApprovalMode | SafetyMode): ApprovalPolicy {
    if (mode === this.policy.mode) {
      return this.policy;
    }

    switch (mode) {
      case SafetyMode.AUTOPILOT:
        return {
          ...this.policy,
          mode,
          approvalPolicy: "never",
          sandboxMode: "danger-full-access",
          networkAccess: "on",
        };
      case SafetyMode.DEFAULT:
        return {
          ...this.policy,
          mode,
          approvalPolicy: "on-request",
          sandboxMode: "workspace-write",
          networkAccess: "off",
        };
      case ApprovalMode.SUGGEST:
      case ApprovalMode.AUTO_EDIT:
      case ApprovalMode.FULL_AUTO:
        return {
          ...this.policy,
          mode,
        };
      default:
        return {
          ...this.policy,
          mode,
        };
    }
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
  private normalizeRequest(request: ApprovalRequest): NormalizedApprovalRequest {
    const repoRoot = request.repoRoot ? resolve(request.repoRoot) : null;
    const pathState = normalizeRequestPath(request.filePath, repoRoot);
    const commandState = normalizeRequestCommand(request.arguments, repoRoot);

    return {
      ...pathState,
      ...commandState,
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

function normalizeRequestPath(
  rawFilePath: string | null,
  repoRoot: string | null,
): Pick<
  NormalizedApprovalRequest,
  "filePath" | "isInsideRepo" | "isOutsideRepo" | "hasSensitivePath"
> {
  const filePath = rawFilePath && repoRoot
    ? resolvePath(rawFilePath, repoRoot)
    : rawFilePath;
  const isInsideRepo = Boolean(filePath && repoRoot && isSubpath(filePath, repoRoot));
  return {
    filePath: filePath ?? null,
    isInsideRepo,
    isOutsideRepo: Boolean(filePath && repoRoot && !isInsideRepo),
    hasSensitivePath: Boolean(filePath && isSensitivePath(filePath)),
  };
}

function normalizeRequestCommand(
  args: Readonly<Record<string, unknown>> | undefined,
  repoRoot: string | null,
): Pick<NormalizedApprovalRequest, "command" | "commandRunsInRepo"> {
  const command = typeof args?.["command"] === "string"
    ? normalizeCommand(args["command"])
    : null;
  const cwd = typeof args?.["cwd"] === "string" ? args["cwd"] : ".";
  const resolvedCwd = repoRoot ? resolvePath(cwd, repoRoot) : null;
  return {
    command,
    commandRunsInRepo: resolvedCwd && repoRoot ? isSubpath(resolvedCwd, repoRoot) : true,
  };
}

function decideModernRequest(options: {
  readonly request: ApprovalRequest;
  readonly normalized: NormalizedApprovalRequest;
  readonly approvalPolicy: ApprovalPolicyMode;
  readonly sandboxMode: string;
  readonly networkAccess: string;
}): ApprovalDecision {
  if (options.request.toolCategory === "external") {
    return allowWhen(options.approvalPolicy === "never" || options.networkAccess === "on");
  }
  if (options.normalized.hasSensitivePath) {
    return allowWhen(options.approvalPolicy === "never");
  }
  if (options.request.toolCategory === "readonly") {
    return decideReadonlyRequest(options.normalized, options.approvalPolicy);
  }
  if (options.request.toolCategory === "mutating") {
    return decideMutatingRequest(options);
  }
  if (options.request.toolCategory === "workflow") {
    return decideWorkflowRequest(options.normalized, options.approvalPolicy);
  }
  return "ask";
}

function decideReadonlyRequest(
  normalized: NormalizedApprovalRequest,
  approvalPolicy: ApprovalPolicyMode,
): ApprovalDecision {
  if (normalized.filePath && normalized.isOutsideRepo) {
    return allowWhen(approvalPolicy === "never");
  }
  return "allow";
}

function decideMutatingRequest(options: {
  readonly request: ApprovalRequest;
  readonly normalized: NormalizedApprovalRequest;
  readonly approvalPolicy: ApprovalPolicyMode;
  readonly sandboxMode: string;
}): ApprovalDecision {
  if (options.request.toolName === "git_commit") {
    return allowWhen(options.approvalPolicy === "never");
  }
  if (options.approvalPolicy === "never" || options.sandboxMode === "danger-full-access") {
    return "allow";
  }
  if (options.approvalPolicy === "strict") {
    return "ask";
  }
  return allowWhen(Boolean(options.normalized.filePath && options.normalized.isInsideRepo));
}

function decideWorkflowRequest(
  normalized: NormalizedApprovalRequest,
  approvalPolicy: ApprovalPolicyMode,
): ApprovalDecision {
  if (approvalPolicy === "never") {
    return "allow";
  }
  if (approvalPolicy === "strict" || !normalized.commandRunsInRepo) {
    return "ask";
  }
  return normalized.command && isDefaultSafeCommand(normalized.command) ? "allow" : "ask";
}

function allowWhen(condition: boolean): ApprovalDecision {
  return condition ? "allow" : "ask";
}

function isSensitivePath(filePath: string): boolean {
  const lowered = filePath.toLowerCase();
  return SENSITIVE_PATH_MATCHERS.some((matches) => matches(lowered));
}

function normalizeCommand(command: string): string {
  return command.trim().replace(/\s+/g, " ");
}
function isDefaultSafeCommand(command: string): boolean {
  const lower = command.toLowerCase();

  if (SHELL_OPERATOR_PATTERNS.some((operator) => lower.includes(operator))) {
    return false;
  }

  if (UNSAFE_COMMAND_PATTERNS.some((pattern) => pattern.test(lower))) {
    return false;
  }

  return SAFE_COMMAND_PATTERNS.some((pattern) => pattern.test(lower));
}
