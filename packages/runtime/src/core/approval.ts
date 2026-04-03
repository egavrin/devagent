/**
 * Approval system — three-tier model for tool execution control.
 *
 * | Mode       | File Read | File Write | Shell        | Network     |
 * |------------|-----------|------------|--------------|-------------|
 * | Suggest    | Yes       | Diff shown | Command shown| Blocked     |
 * | Auto-Edit  | Yes       | Yes        | Command shown| Blocked     |
 * | Full-Auto  | Yes       | Yes        | Sandboxed    | Sandboxed   |
 *
 * Per-tool and per-path overrides. Audit log via event bus.
 */

import type {
  ApprovalPolicy,
  ToolSpec,
  ToolCategory,
} from "./types.js";
import { ApprovalMode } from "./types.js";
import { ApprovalDeniedError } from "./errors.js";
import type { EventBus } from "./events.js";

// ─── Decision Types ──────────────────────────────────────────

export type ApprovalDecision = "allow" | "deny" | "ask";

export interface ApprovalRequest {
  readonly toolName: string;
  readonly toolCategory: ToolCategory;
  readonly filePath: string | null;
  readonly description: string;
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
  private currentMode: ApprovalMode;
  private userResponseResolver:
    | ((approved: boolean) => void)
    | null = null;

  constructor(policy: ApprovalPolicy, bus?: EventBus) {
    this.policy = policy;
    this.bus = bus ?? null;
    this.currentMode = policy.mode;

    // Listen for user approval responses
    if (this.bus) {
      this.bus.on("approval:response", (event) => {
        if (this.userResponseResolver) {
          this.userResponseResolver(event.approved);
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

    const approved = await this.waitForUserResponse();
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

    // 2. Check per-path rules
    if (request.filePath) {
      const pathDecision = this.checkPathRules(request.filePath);
      if (pathDecision) return pathDecision;
    }

    // 3. Apply mode-based rules
    return this.decideByMode(request.toolCategory);
  }

  private checkPathRules(filePath: string): ApprovalDecision | null {
    for (const rule of this.policy.pathRules) {
      if (matchPath(filePath, rule.pattern)) {
        return rule.action;
      }
    }
    return null;
  }

  private decideByMode(category: ToolCategory): ApprovalDecision {
    switch (this.currentMode) {
      case ApprovalMode.SUGGEST:
        return this.decideSuggestMode(category);
      case ApprovalMode.AUTO_EDIT:
        return this.decideAutoEditMode(category);
      case ApprovalMode.FULL_AUTO:
        return this.decideFullAutoMode(category);
      default:
        // Fail fast: unknown mode
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
        return "deny"; // network blocked in suggest mode
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
    // Everything allowed in full-auto (sandboxed execution)
    return "allow";
  }

  private waitForUserResponse(): Promise<boolean> {
    return new Promise<boolean>((resolve) => {
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

  getMode(): ApprovalMode {
    return this.currentMode;
  }

  setMode(mode: ApprovalMode): void {
    this.currentMode = mode;
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
