/**
 * Credential storage for API keys.
 * Stores credentials in ~/.config/devagent/credentials.json with 0o600 permissions.
 * Fail fast: throws on file system errors, never silently returns stale data.
 */

import { readFileSync, writeFileSync, existsSync, mkdirSync } from "node:fs";
import { join, dirname } from "node:path";
import { homedir } from "node:os";
import { CredentialError , extractErrorMessage } from "./errors.js";

// ─── Types ──────────────────────────────────────────────────

/** API key credential. */
export interface ApiCredential {
  readonly type: "api";
  readonly key: string;
  readonly storedAt: number; // Unix timestamp ms
}

/** OAuth credential with access token and optional refresh/expiry. */
export interface OAuthCredential {
  readonly type: "oauth";
  readonly accessToken: string;
  /** Refresh token — absent for providers that issue non-expiring tokens (GitHub). */
  readonly refreshToken?: string;
  /** Expiry timestamp in ms — absent for non-expiring tokens (GitHub OAuth). */
  readonly expiresAt?: number;
  readonly accountId?: string; // e.g., ChatGPT org ID
  readonly storedAt: number;
}

/** Backward-compatible alias for API key credentials. */
export type Credential = ApiCredential;

/** Discriminated union of all credential types. */
export type CredentialInfo = ApiCredential | OAuthCredential;

export interface CredentialStoreOptions {
  /** Override the default credentials file path. Useful for testing. */
  readonly filePath?: string;
}

// ─── Default Path ───────────────────────────────────────────

function getDefaultFilePath(): string {
  return join(homedir(), ".config", "devagent", "credentials.json");
}

// ─── Store ──────────────────────────────────────────────────

export class CredentialStore {
  private readonly filePath: string;

  constructor(options?: CredentialStoreOptions) {
    this.filePath = options?.filePath ?? getDefaultFilePath();
  }

  /** Get credential for a specific provider. Returns null if not stored. */
  get(providerId: string): CredentialInfo | null {
    const all = this.all();
    return all[providerId] ?? null;
  }

  /** Get all stored credentials. Returns empty object if file doesn't exist. */
  all(): Readonly<Record<string, CredentialInfo>> {
    if (!existsSync(this.filePath)) {
      return {};
    }

    let raw: string;
    try {
      raw = readFileSync(this.filePath, "utf-8");
    } catch (err) {
      const msg = extractErrorMessage(err);
      throw new CredentialError(`Failed to read credentials file: ${msg}`);
    }

    let parsed: unknown;
    try {
      parsed = JSON.parse(raw);
    } catch {
      throw new CredentialError(
        `Corrupted credentials file at ${this.filePath}. Delete it and re-run "devagent auth login".`,
      );
    }

    if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
      throw new CredentialError(
        `Invalid credentials file format at ${this.filePath}. Expected JSON object.`,
      );
    }

    // Validate and filter entries
    const result: Record<string, CredentialInfo> = {};
    for (const [key, value] of Object.entries(parsed as Record<string, unknown>)) {
      if (isValidCredential(value)) {
        result[key] = value as CredentialInfo;
      }
      // Skip invalid entries silently — they might come from a future version
    }

    return result;
  }

  /** Store or overwrite a credential for a provider. Creates file with 0o600. */
  set(providerId: string, credential: CredentialInfo): void {
    const existing = this.all();
    const updated = { ...existing, [providerId]: credential };
    this.writeFile(updated);
  }

  /** Remove a credential. Returns true if it existed. */
  remove(providerId: string): boolean {
    const existing = this.all();
    if (!(providerId in existing)) {
      return false;
    }
    const { [providerId]: _, ...rest } = existing;
    this.writeFile(rest);
    return true;
  }

  /** Check if a credential is stored for this provider. */
  has(providerId: string): boolean {
    return this.get(providerId) !== null;
  }

  // ─── Private ────────────────────────────────────────────────

  private writeFile(data: Record<string, CredentialInfo>): void {
    const dir = dirname(this.filePath);
    if (!existsSync(dir)) {
      mkdirSync(dir, { recursive: true });
    }

    try {
      writeFileSync(this.filePath, JSON.stringify(data, null, 2) + "\n", {
        mode: 0o600,
      });
    } catch (err) {
      const msg = extractErrorMessage(err);
      throw new CredentialError(`Failed to write credentials file: ${msg}`);
    }
  }
}

// ─── Validation ─────────────────────────────────────────────

function isValidCredential(value: unknown): boolean {
  if (typeof value !== "object" || value === null) return false;
  const obj = value as Record<string, unknown>;
  if (obj["type"] === "api") {
    return typeof obj["key"] === "string";
  }
  if (obj["type"] === "oauth") {
    // accessToken is required; refreshToken and expiresAt are optional
    // (GitHub OAuth tokens don't expire and have no refresh token)
    return (
      typeof obj["accessToken"] === "string" &&
      (obj["refreshToken"] === undefined || typeof obj["refreshToken"] === "string") &&
      (obj["expiresAt"] === undefined || obj["expiresAt"] === null || typeof obj["expiresAt"] === "number")
    );
  }
  return false;
}
