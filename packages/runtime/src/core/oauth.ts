/**
 * OAuth primitives — PKCE, callback server, token exchange, device code.
 * Protocol-level utilities with no provider-specific logic.
 * Uses Bun.serve() for the local callback server.
 */

import { randomBytes } from "node:crypto";

import { OAuthError } from "./errors.js";

// ─── Types ──────────────────────────────────────────────────

export interface PKCEPair {
  readonly verifier: string;
  readonly challenge: string;
}

export interface TokenResponse {
  readonly access_token: string;
  /** Refresh token — not all providers return one (GitHub OAuth does not). */
  readonly refresh_token?: string;
  /** Token lifetime in seconds — not all providers return one (GitHub OAuth does not). */
  readonly expires_in?: number;
  readonly id_token?: string;
}

export interface DeviceCodeResponse {
  readonly user_code: string;
  readonly device_code: string;
  readonly verification_uri: string;
  readonly interval: number;
  readonly expires_in: number;
}

export interface CallbackResult {
  readonly code: string;
  readonly state: string;
}

export interface CallbackServer {
  readonly promise: Promise<CallbackResult>;
  readonly shutdown: () => void;
}

// ─── PKCE ───────────────────────────────────────────────────

/**
 * Generate PKCE code verifier and S256 challenge.
 * Uses Web Crypto API (available in Bun and modern Node).
 */
export async function generatePKCE(): Promise<PKCEPair> {
  const verifier = randomBytes(32).toString("base64url");
  const digest = await crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(verifier),
  );
  const challenge = Buffer.from(digest).toString("base64url");
  return { verifier, challenge };
}

/** Generate a random state parameter for CSRF protection. */
export function generateState(): string {
  return randomBytes(16).toString("hex");
}

// ─── Local Callback Server ──────────────────────────────────

const HTML_SUCCESS = `<!DOCTYPE html>
<html><head><title>DevAgent - Authorization Successful</title>
<style>body{font-family:system-ui,sans-serif;display:flex;justify-content:center;align-items:center;height:100vh;margin:0;background:#1a1a2e;color:#e0e0e0}
.container{text-align:center;padding:2rem}h1{color:#4ade80;margin-bottom:1rem}</style></head>
<body><div class="container"><h1>Authorization Successful</h1><p>You can close this tab and return to DevAgent.</p></div>
<script>setTimeout(()=>window.close(),2000)</script></body></html>`;

const HTML_ERROR = (msg: string) => `<!DOCTYPE html>
<html><head><title>DevAgent - Authorization Failed</title>
<style>body{font-family:system-ui,sans-serif;display:flex;justify-content:center;align-items:center;height:100vh;margin:0;background:#1a1a2e;color:#e0e0e0}
.container{text-align:center;padding:2rem}h1{color:#f87171;margin-bottom:1rem}.error{color:#fca5a5;font-family:monospace;margin-top:1rem;padding:1rem;background:#2d1b1b;border-radius:0.5rem}</style></head>
<body><div class="container"><h1>Authorization Failed</h1><div class="error">${msg}</div></div></body></html>`;

/**
 * Start a local HTTP server to receive the OAuth callback.
 * Returns a promise that resolves with the authorization code.
 * The server auto-shuts down after receiving the callback or on timeout (5 min).
 */
export function startCallbackServer(
  port: number,
  expectedState: string,
): CallbackServer {
  let resolveCallback!: (result: CallbackResult) => void;
  let rejectCallback!: (error: Error) => void;

  const promise = new Promise<CallbackResult>((resolve, reject) => {
    resolveCallback = resolve;
    rejectCallback = reject;
  });

  const server = (globalThis as any).Bun?.serve({
    port,
    fetch(req: Request) {
      const url = new URL(req.url);

      if (url.pathname === "/auth/callback") {
        const code = url.searchParams.get("code");
        const state = url.searchParams.get("state");
        const error = url.searchParams.get("error");
        const errorDesc = url.searchParams.get("error_description");

        if (error) {
          const msg = errorDesc ?? error;
          rejectCallback(new OAuthError(`Authorization failed: ${msg}`));
          return new Response(HTML_ERROR(msg), {
            headers: { "Content-Type": "text/html" },
          });
        }

        if (!code || !state) {
          const msg = "Missing authorization code or state";
          rejectCallback(new OAuthError(msg));
          return new Response(HTML_ERROR(msg), {
            status: 400,
            headers: { "Content-Type": "text/html" },
          });
        }

        if (state !== expectedState) {
          const msg = "Invalid state — possible CSRF attack";
          rejectCallback(new OAuthError(msg));
          return new Response(HTML_ERROR(msg), {
            status: 403,
            headers: { "Content-Type": "text/html" },
          });
        }

        resolveCallback({ code, state });
        return new Response(HTML_SUCCESS, {
          headers: { "Content-Type": "text/html" },
        });
      }

      return new Response("Not found", { status: 404 });
    },
  });

  if (!server) {
    rejectCallback(new OAuthError("Bun.serve() not available — OAuth browser flow requires Bun runtime"));
  }

  // 5-minute timeout
  const timeout = setTimeout(() => {
    rejectCallback(new OAuthError("OAuth callback timed out after 5 minutes"));
    server?.stop?.();
  }, 5 * 60 * 1000);

  const shutdown = () => {
    clearTimeout(timeout);
    server?.stop?.();
  };

  // Clean up timeout on resolution
  promise.then(() => clearTimeout(timeout)).catch(() => clearTimeout(timeout));

  return { promise, shutdown };
}

// ─── Token Exchange ─────────────────────────────────────────

/**
 * Exchange an authorization code (or refresh token) for access tokens.
 * POST with application/x-www-form-urlencoded body.
 */
export async function exchangeCodeForTokens(
  tokenUrl: string,
  params: Record<string, string>,
): Promise<TokenResponse> {
  const response = await fetch(tokenUrl, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams(params).toString(),
  });

  if (!response.ok) {
    const body = await response.text();
    throw new OAuthError(`Token exchange failed (${response.status}): ${body}`);
  }

  return response.json() as Promise<TokenResponse>;
}

/**
 * Refresh an OAuth access token using a refresh token.
 */
export async function refreshAccessToken(
  tokenUrl: string,
  refreshToken: string,
  clientId: string,
): Promise<TokenResponse> {
  return exchangeCodeForTokens(tokenUrl, {
    grant_type: "refresh_token",
    refresh_token: refreshToken,
    client_id: clientId,
  });
}

// ─── Device Code Flow ───────────────────────────────────────

/**
 * Request a device code for device authorization flow (RFC 8628).
 */
export async function requestDeviceCode(
  deviceCodeUrl: string,
  clientId: string,
  scopes: ReadonlyArray<string>,
): Promise<DeviceCodeResponse> {
  const params: Record<string, string> = { client_id: clientId };
  if (scopes.length > 0) {
    params["scope"] = scopes.join(" ");
  }

  const response = await fetch(deviceCodeUrl, {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
      Accept: "application/json",
    },
    body: new URLSearchParams(params).toString(),
  });

  if (!response.ok) {
    const body = await response.text();
    throw new OAuthError(`Device code request failed (${response.status}): ${body}`);
  }

  return response.json() as Promise<DeviceCodeResponse>;
}

/**
 * Poll for a device code token until the user authorizes or the code expires.
 * Handles `authorization_pending` and `slow_down` responses per RFC 8628.
 */
export async function pollDeviceCodeToken(
  tokenUrl: string,
  deviceCode: string,
  clientId: string,
  interval: number,
  expiresIn: number,
): Promise<TokenResponse> {
  const deadline = Date.now() + expiresIn * 1000;
  let pollInterval = Math.max(interval, 5); // Minimum 5 seconds

  while (Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, pollInterval * 1000));

    const response = await fetch(tokenUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
        Accept: "application/json",
      },
      body: new URLSearchParams({
        grant_type: "urn:ietf:params:oauth:grant-type:device_code",
        device_code: deviceCode,
        client_id: clientId,
      }).toString(),
    });

    const body = (await response.json()) as Record<string, unknown>;

    if (body["access_token"]) {
      return body as unknown as TokenResponse;
    }

    const error = body["error"] as string | undefined;
    if (error === "authorization_pending") {
      continue;
    }
    if (error === "slow_down") {
      pollInterval += 5; // Back off per RFC 8628
      continue;
    }

    throw new OAuthError(`Device code authorization error: ${error ?? "unknown"}`);
  }

  throw new OAuthError("Device code authorization expired — user did not complete login in time");
}

// ─── ChatGPT Device Code Flow ────────────────────────────────
// ChatGPT uses a non-standard device code flow with JSON bodies
// and a 3-step process: usercode → poll → exchange.

export interface ChatGPTDeviceCodeResponse {
  readonly device_auth_id: string;
  readonly user_code: string;
  readonly interval: number;
  readonly expires_in: number;
}

/**
 * Request a ChatGPT device code. Uses JSON body (non-standard).
 * Endpoint: POST /api/accounts/deviceauth/usercode
 */
export async function requestChatGPTDeviceCode(
  deviceCodeUrl: string,
  clientId: string,
): Promise<ChatGPTDeviceCodeResponse> {
  const response = await fetch(deviceCodeUrl, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    body: JSON.stringify({ client_id: clientId }),
  });

  if (!response.ok) {
    const body = await response.text();
    throw new OAuthError(`ChatGPT device code request failed (${response.status}): ${body}`);
  }

  const data = (await response.json()) as Record<string, unknown>;
  return {
    device_auth_id: data["device_auth_id"] as string,
    user_code: (data["user_code"] ?? data["usercode"]) as string,
    interval: parseInt(String(data["interval"] ?? "5"), 10),
    expires_in: parseInt(String(data["expires_in"] ?? "900"), 10),
  };
}

interface ChatGPTAuthorizationResult {
  readonly authorization_code: string;
  readonly code_verifier: string;
}

/**
 * Poll ChatGPT device auth endpoint until user authorizes.
 * Returns an authorization_code + code_verifier for token exchange.
 * Endpoint: POST /api/accounts/deviceauth/token (JSON body)
 * Pending = HTTP 403/404, success = HTTP 2xx.
 */
export async function pollChatGPTDeviceAuth(
  pollUrl: string,
  deviceAuthId: string,
  userCode: string,
  interval: number,
  expiresIn: number,
): Promise<ChatGPTAuthorizationResult> {
  const deadline = Date.now() + expiresIn * 1000;
  const pollInterval = Math.max(interval, 5); // Minimum 5 seconds

  while (Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, pollInterval * 1000));

    const response = await fetch(pollUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify({
        device_auth_id: deviceAuthId,
        user_code: userCode,
      }),
    });

    // 403/404 = authorization still pending
    if (response.status === 403 || response.status === 404) {
      continue;
    }

    if (!response.ok) {
      const body = await response.text();
      throw new OAuthError(`ChatGPT device auth polling error (${response.status}): ${body}`);
    }

    // Success — extract authorization_code and code_verifier
    const data = (await response.json()) as Record<string, unknown>;
    const authorizationCode = data["authorization_code"] as string | undefined;
    const codeVerifier = data["code_verifier"] as string | undefined;

    if (!authorizationCode) {
      throw new OAuthError("ChatGPT device auth response missing authorization_code");
    }

    return {
      authorization_code: authorizationCode,
      code_verifier: codeVerifier ?? "",
    };
  }

  throw new OAuthError("ChatGPT device code authorization expired — user did not complete login in time");
}

/**
 * Complete ChatGPT device code flow: exchange authorization_code for tokens.
 * Uses the server-provided code_verifier (PKCE generated server-side).
 */
export async function exchangeChatGPTDeviceToken(
  tokenUrl: string,
  clientId: string,
  authorizationCode: string,
  codeVerifier: string,
): Promise<TokenResponse> {
  return exchangeCodeForTokens(tokenUrl, {
    grant_type: "authorization_code",
    client_id: clientId,
    code: authorizationCode,
    code_verifier: codeVerifier,
    redirect_uri: "https://auth.openai.com/deviceauth/callback",
  });
}

// ─── GitHub Copilot Token Exchange ───────────────────────────

export interface CopilotSessionToken {
  /** Short-lived Copilot JWT session token (~30 min). */
  readonly token: string;
  /** Expiry as Unix timestamp in milliseconds. */
  readonly expiresAt: number;
  /** Optional API endpoint from the token response (e.g., "https://api.individual.githubcopilot.com"). */
  readonly endpoint?: string;
}

/**
 * Exchange a GitHub OAuth token for a short-lived Copilot session token.
 * The session token expires in ~30 minutes and must be refreshed by re-calling this.
 *
 * GET https://api.github.com/copilot_internal/v2/token
 * Authorization: token <github_oauth_token>
 *
 * The returned session JWT goes in `Authorization: Bearer` headers to the Copilot API.
 */
export async function exchangeCopilotSessionToken(
  githubOAuthToken: string,
): Promise<CopilotSessionToken> {
  const response = await fetch("https://api.github.com/copilot_internal/v2/token", {
    headers: {
      "Authorization": `token ${githubOAuthToken}`,
      "Accept": "application/json",
      "Editor-Version": "vscode/1.104.1",
      "Editor-Plugin-Version": "copilot-chat/0.26.7",
      "User-Agent": "GitHubCopilotChat/0.26.7",
    },
  });

  if (!response.ok) {
    const body = await response.text();
    throw new OAuthError(
      `Copilot token exchange failed (${response.status}): ${body}. ` +
      `Make sure your GitHub account has an active Copilot subscription.`,
    );
  }

  const data = (await response.json()) as Record<string, unknown>;
  const token = data["token"] as string | undefined;
  if (!token) {
    throw new OAuthError("Copilot token response missing 'token' field");
  }

  const expiresAt = data["expires_at"] as number | undefined;
  const endpoints = data["endpoints"] as Record<string, string> | undefined;

  return {
    token,
    expiresAt: expiresAt ? expiresAt * 1000 : Date.now() + 25 * 60 * 1000, // 25 min fallback
    endpoint: endpoints?.["api"],
  };
}

// ─── JWT Helpers ────────────────────────────────────────────

/**
 * Extract accountId from an OpenAI id_token (JWT).
 * Used for ChatGPT org subscriptions.
 */
export function extractAccountIdFromIdToken(idToken: string): string | undefined {
  const parts = idToken.split(".");
  if (parts.length !== 3) return undefined;
  try {
    const claims = JSON.parse(Buffer.from(parts[1]!, "base64url").toString()) as Record<string, unknown>;
    return (
      (claims["chatgpt_account_id"] as string | undefined) ??
      ((claims["https://api.openai.com/auth"] as Record<string, unknown> | undefined)?.["chatgpt_account_id"] as string | undefined) ??
      ((claims["organizations"] as Array<{ id: string }> | undefined)?.[0]?.id)
    );
  } catch {
    return undefined;
  }
}
