/**
 * SkillResolver — Argument substitution and shell preprocessing for skills.
 *
 * Substitution pipeline (order matters):
 * 1. Shell preprocessing: !`command` → execute, replace with stdout
 * 2. Positional arguments: $ARGUMENTS, $0, $1, $ARGUMENTS[N]
 * 3. Environment variables: ${SKILL_DIR}, ${SESSION_ID}
 */

import { execSync } from "node:child_process";

import type { Skill, ResolvedSkill } from "./types.js";

// ─── Types ───────────────────────────────────────────────────

export interface ResolveContext {
  readonly sessionId: string;
  readonly allowShellPreprocess: boolean;
}

export interface SkillResolverOptions {
  /** Timeout for shell commands in ms. Default: 5000. */
  readonly shellTimeoutMs?: number;
}

// ─── SkillResolver Class ─────────────────────────────────────

export class SkillResolver {
  private readonly shellTimeoutMs: number;

  constructor(options?: SkillResolverOptions) {
    this.shellTimeoutMs = options?.shellTimeoutMs ?? 5000;
  }

  async resolve(
    skill: Skill,
    args: string,
    context: ResolveContext,
  ): Promise<ResolvedSkill> {
    let text = skill.instructions;

    // 1. Shell preprocessing: !`command` → replace with sentinels to prevent
    //    shell output from being re-processed by argument substitution.
    const shellOutputs = new Map<string, string>();
    if (context.allowShellPreprocess) {
      let shellIndex = 0;
      text = text.replace(/!\`([^`]+)\`/g, (_match, command) => {
        const sentinel = `\x00SHELL_OUT_${shellIndex++}\x00`;
        shellOutputs.set(sentinel, this.executeShellCommand(command, skill.dirPath));
        return sentinel;
      });
    }

    // 2. Positional arguments (order matters: $ARGUMENTS[N] before $ARGUMENTS before $N)
    const argParts = args ? args.split(/\s+/) : [];
    text = text.replace(/\$ARGUMENTS\[(\d+)\]/g, (_match, index) => {
      const idx = parseInt(index, 10);
      return argParts[idx] ?? "";
    });
    text = text.replace(/\$ARGUMENTS/g, args);
    text = text.replace(/\$(\d+)/g, (_match, index) => {
      const idx = parseInt(index, 10);
      return argParts[idx] ?? "";
    });

    // 3. Environment variables
    text = text.replace(/\$\{SKILL_DIR\}/g, skill.dirPath);
    text = text.replace(/\$\{SESSION_ID\}/g, context.sessionId);

    // 4. Restore shell outputs (after all other substitutions)
    for (const [sentinel, output] of shellOutputs) {
      text = text.replace(sentinel, output);
    }

    return {
      ...skill,
      resolvedInstructions: text,
    };
  }

  // ─── Private ────────────────────────────────────────────────

  private executeShellCommand(command: string, cwd: string): string {
    try {
      const output = execSync(command, {
        cwd,
        timeout: this.shellTimeoutMs,
        encoding: "utf-8",
        stdio: ["pipe", "pipe", "pipe"],
      });
      return output.trim();
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      return `[shell error: ${message}]`;
    }
  }
}
