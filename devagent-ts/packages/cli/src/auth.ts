/**
 * Auth subcommands — login, status, logout.
 * Manages persistent API key and OAuth token storage via CredentialStore.
 * Supports:
 *   - API key login (anthropic, openai, custom)
 *   - Browser OAuth PKCE (chatgpt)
 *   - ChatGPT device code flow (non-standard 3-step)
 *   - GitHub device code flow (RFC 8628)
 */

import { createInterface } from "node:readline";
import {
  CredentialStore,
  getOAuthProvider,
  generatePKCE,
  generateState,
  startCallbackServer,
  exchangeCodeForTokens,
  requestChatGPTDeviceCode,
  pollChatGPTDeviceAuth,
  exchangeChatGPTDeviceToken,
  requestDeviceCode,
  pollDeviceCodeToken,
  extractAccountIdFromIdToken,
  openUrl,
} from "@devagent/core";
import type { OAuthProviderConfig, CredentialInfo } from "@devagent/core";
import { bold, cyan, dim, green, red, yellow } from "./format.js";

// ─── Known Providers ────────────────────────────────────────

interface ProviderEntry {
  readonly id: string;
  readonly name: string;
  readonly hint: string;
  readonly envVar: string;
  readonly authMethods: readonly ("api-key" | "browser-oauth" | "device-code")[];
}

const KNOWN_PROVIDERS: readonly ProviderEntry[] = [
  {
    id: "anthropic",
    name: "Anthropic (API key)",
    hint: "Starts with sk-ant-. Get one at https://console.anthropic.com/settings/keys",
    envVar: "ANTHROPIC_API_KEY",
    authMethods: ["api-key"],
  },
  {
    id: "openai",
    name: "OpenAI (API key)",
    hint: "Starts with sk-. Get one at https://platform.openai.com/api-keys",
    envVar: "OPENAI_API_KEY",
    authMethods: ["api-key"],
  },
  {
    id: "chatgpt",
    name: "ChatGPT (Pro/Plus account)",
    hint: "Login with your ChatGPT subscription",
    envVar: "",
    authMethods: ["device-code"],
  },
  {
    id: "github-copilot",
    name: "GitHub Copilot",
    hint: "Login with your GitHub account",
    envVar: "",
    authMethods: ["device-code"],
  },
];

// ─── Entry Point ────────────────────────────────────────────

export async function runAuthCommand(subcommand: string): Promise<void> {
  switch (subcommand) {
    case "login":
      await authLogin();
      break;
    case "status":
      authStatus();
      break;
    case "logout":
      await authLogout();
      break;
    default:
      process.stderr.write(
        red(`Unknown auth command: ${subcommand}`) + "\n" +
        dim("Available: login, status, logout") + "\n",
      );
      process.exit(1);
  }
}

// ─── Login ──────────────────────────────────────────────────

async function authLogin(): Promise<void> {
  const rl = createInterface({
    input: process.stdin,
    output: process.stderr,
  });

  const question = (prompt: string): Promise<string> =>
    new Promise((resolve) => rl.question(prompt, resolve));

  try {
    process.stderr.write(bold("Add credential") + "\n\n");

    // Show provider choices
    process.stderr.write("Select provider:\n");
    for (let i = 0; i < KNOWN_PROVIDERS.length; i++) {
      process.stderr.write(`  ${cyan(String(i + 1))}. ${KNOWN_PROVIDERS[i]!.name}\n`);
    }
    process.stderr.write("\n");

    const choice = await question(cyan("> "));
    const choiceNum = parseInt(choice.trim(), 10);

    let providerId: string;
    let providerEntry: ProviderEntry | undefined;

    if (choiceNum >= 1 && choiceNum <= KNOWN_PROVIDERS.length) {
      providerEntry = KNOWN_PROVIDERS[choiceNum - 1]!;
      providerId = providerEntry.id;
    } else {
      // Try treating input as provider name directly
      const directMatch = KNOWN_PROVIDERS.find(
        (p) => p.id === choice.trim().toLowerCase(),
      );
      if (directMatch) {
        providerEntry = directMatch;
        providerId = directMatch.id;
      } else {
        process.stderr.write(red("Invalid selection.") + "\n");
        return;
      }
    }

    process.stderr.write("\n");

    // Check if this provider supports OAuth
    const oauthConfig = getOAuthProvider(providerId);
    const authMethods = providerEntry?.authMethods ?? ["api-key"];

    if (oauthConfig && authMethods.some((m) => m !== "api-key")) {
      // OAuth provider — choose auth method
      let authMethod: "browser-oauth" | "device-code" | "api-key";

      if (authMethods.length === 1) {
        authMethod = authMethods[0]!;
      } else {
        process.stderr.write("Authentication method:\n");
        const options: { label: string; method: "browser-oauth" | "device-code" | "api-key" }[] = [];
        if (authMethods.includes("browser-oauth")) {
          options.push({ label: "Browser login (opens browser)", method: "browser-oauth" });
        }
        if (authMethods.includes("device-code")) {
          options.push({ label: "Device code (paste code in browser)", method: "device-code" });
        }
        if (authMethods.includes("api-key")) {
          options.push({ label: "API key", method: "api-key" });
        }

        for (let i = 0; i < options.length; i++) {
          process.stderr.write(`  ${cyan(String(i + 1))}. ${options[i]!.label}\n`);
        }
        process.stderr.write("\n");

        const methodChoice = await question(cyan("> "));
        const methodNum = parseInt(methodChoice.trim(), 10);
        if (methodNum >= 1 && methodNum <= options.length) {
          authMethod = options[methodNum - 1]!.method;
        } else {
          authMethod = options[0]!.method; // Default to first
        }
      }

      process.stderr.write("\n");

      // Close the readline before OAuth flows (they manage their own I/O)
      rl.close();

      if (authMethod === "browser-oauth") {
        await authLoginBrowserOAuth(providerId, oauthConfig);
      } else if (authMethod === "device-code") {
        if (providerId === "chatgpt") {
          await authLoginChatGPTDeviceCode(providerId, oauthConfig);
        } else {
          await authLoginDeviceCode(providerId, oauthConfig);
        }
      } else {
        // Fall through to API key — reopen rl
        const rl2 = createInterface({ input: process.stdin, output: process.stderr });
        const q2 = (prompt: string): Promise<string> =>
          new Promise((resolve) => rl2.question(prompt, resolve));
        try {
          await authLoginApiKey(providerId, providerEntry?.hint, q2);
        } finally {
          rl2.close();
        }
      }
      return;
    }

    // API key flow
    if (providerEntry?.hint) {
      process.stderr.write(dim(providerEntry.hint) + "\n");
    }
    await authLoginApiKey(providerId, providerEntry?.hint, question);
  } finally {
    // rl might already be closed for OAuth flows — safe to call close() multiple times
    rl.close();
  }
}

// ─── API Key Login ──────────────────────────────────────────

async function authLoginApiKey(
  providerId: string,
  _hint: string | undefined,
  question: (prompt: string) => Promise<string>,
): Promise<void> {
  const apiKey = (await question("API key: ")).trim();

  if (!apiKey) {
    process.stderr.write(red("API key cannot be empty.") + "\n");
    return;
  }

  const store = new CredentialStore();
  store.set(providerId, {
    type: "api",
    key: apiKey,
    storedAt: Date.now(),
  });

  process.stderr.write("\n" + green("\u2713 Credential stored for " + providerId) + "\n");
  process.stderr.write(dim("Use with: devagent --provider " + providerId + ' "your query"') + "\n");
}

// ─── Browser OAuth Login (PKCE) ─────────────────────────────

async function authLoginBrowserOAuth(
  providerId: string,
  oauthConfig: OAuthProviderConfig,
): Promise<void> {
  if (!oauthConfig.authorizationUrl) {
    process.stderr.write(red("Provider does not support browser OAuth.") + "\n");
    return;
  }

  process.stderr.write(dim("Starting browser login...") + "\n\n");

  // 1. Generate PKCE pair and state
  const pkce = await generatePKCE();
  const state = generateState();

  // 2. Start local callback server
  const port = oauthConfig.callbackPort ?? 1455;
  const callbackUrl = `http://localhost:${port}/auth/callback`;
  const server = startCallbackServer(port, state);

  // 3. Build authorization URL
  const params = new URLSearchParams({
    response_type: "code",
    client_id: oauthConfig.clientId,
    redirect_uri: callbackUrl,
    state,
    code_challenge: pkce.challenge,
    code_challenge_method: "S256",
    scope: oauthConfig.scopes.join(" "),
    audience: "https://api.openai.com/v1",
  });
  const authUrl = `${oauthConfig.authorizationUrl}?${params.toString()}`;

  // 4. Open browser
  openUrl(authUrl);
  process.stderr.write(cyan("Opening browser for authentication...") + "\n");
  process.stderr.write(dim("If the browser doesn't open, visit:") + "\n");
  process.stderr.write(dim(authUrl) + "\n\n");
  process.stderr.write(dim("Waiting for callback... (press Ctrl+C to cancel)") + "\n");

  try {
    // 5. Wait for callback
    const result = await server.promise;

    // 6. Exchange code for tokens
    const tokens = await exchangeCodeForTokens(oauthConfig.tokenUrl, {
      grant_type: "authorization_code",
      client_id: oauthConfig.clientId,
      code: result.code,
      redirect_uri: callbackUrl,
      code_verifier: pkce.verifier,
    });

    // 7. Extract accountId from id_token (for ChatGPT)
    const accountId = tokens.id_token
      ? extractAccountIdFromIdToken(tokens.id_token)
      : undefined;

    // 8. Store credential
    storeOAuthCredential(providerId, tokens, accountId);

    process.stderr.write("\n" + green("\u2713 Authentication successful!") + "\n");
    if (accountId) {
      process.stderr.write(dim("Account ID: " + accountId) + "\n");
    }
    process.stderr.write(dim("Use with: devagent --provider " + providerId + ' "your query"') + "\n");
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    process.stderr.write("\n" + red("\u2717 Authentication failed: " + msg) + "\n");
  } finally {
    server.shutdown();
  }
}

// ─── ChatGPT Device Code Login (non-standard 3-step) ────────

async function authLoginChatGPTDeviceCode(
  providerId: string,
  oauthConfig: OAuthProviderConfig,
): Promise<void> {
  if (!oauthConfig.deviceCodeUrl) {
    process.stderr.write(red("Provider does not support device code flow.") + "\n");
    return;
  }

  process.stderr.write(dim("Requesting device code...") + "\n\n");

  try {
    // Step 1: Request device code (JSON body)
    const deviceCode = await requestChatGPTDeviceCode(
      oauthConfig.deviceCodeUrl,
      oauthConfig.clientId,
    );

    // Step 2: Show user instructions
    const verificationUrl = oauthConfig.deviceVerificationUrl ?? "https://auth.openai.com/codex/device";
    process.stderr.write(bold("To authenticate, visit:") + "\n");
    process.stderr.write("  " + cyan(verificationUrl) + "\n\n");
    process.stderr.write(bold("Enter code: ") + yellow(deviceCode.user_code) + "\n\n");

    // Try to open the verification URL in browser
    openUrl(verificationUrl);

    process.stderr.write(dim("Waiting for authorization... (press Ctrl+C to cancel)") + "\n");

    // Step 3: Poll for authorization (JSON body, HTTP 403/404 = pending)
    const pollUrl = oauthConfig.deviceCodeUrl.replace("/usercode", "/token");
    const authResult = await pollChatGPTDeviceAuth(
      pollUrl,
      deviceCode.device_auth_id,
      deviceCode.user_code,
      deviceCode.interval,
      deviceCode.expires_in,
    );

    // Step 4: Exchange authorization_code for tokens (form-urlencoded)
    const tokens = await exchangeChatGPTDeviceToken(
      oauthConfig.tokenUrl,
      oauthConfig.clientId,
      authResult.authorization_code,
      authResult.code_verifier,
    );

    // Step 5: Extract accountId from id_token
    const accountId = tokens.id_token
      ? extractAccountIdFromIdToken(tokens.id_token)
      : undefined;

    // Step 6: Store credential
    storeOAuthCredential(providerId, tokens, accountId);

    process.stderr.write("\n" + green("\u2713 Authentication successful!") + "\n");
    if (accountId) {
      process.stderr.write(dim("Account ID: " + accountId) + "\n");
    }
    process.stderr.write(dim("Use with: devagent --provider " + providerId + ' "your query"') + "\n");
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    process.stderr.write("\n" + red("\u2717 Authentication failed: " + msg) + "\n");
  }
}

// ─── Standard Device Code Login (RFC 8628 — GitHub Copilot) ─

async function authLoginDeviceCode(
  providerId: string,
  oauthConfig: OAuthProviderConfig,
): Promise<void> {
  if (!oauthConfig.deviceCodeUrl) {
    process.stderr.write(red("Provider does not support device code flow.") + "\n");
    return;
  }

  process.stderr.write(dim("Requesting device code...") + "\n\n");

  try {
    // 1. Request device code (standard RFC 8628 form-urlencoded)
    const deviceCode = await requestDeviceCode(
      oauthConfig.deviceCodeUrl,
      oauthConfig.clientId,
      oauthConfig.scopes,
    );

    // 2. Show user instructions
    const verificationUrl = oauthConfig.deviceVerificationUrl ?? deviceCode.verification_uri;
    process.stderr.write(bold("To authenticate, visit:") + "\n");
    process.stderr.write("  " + cyan(verificationUrl) + "\n\n");
    process.stderr.write(bold("Enter code: ") + yellow(deviceCode.user_code) + "\n\n");

    // Try to open the verification URL in browser
    openUrl(verificationUrl);

    process.stderr.write(dim("Waiting for authorization... (press Ctrl+C to cancel)") + "\n");

    // 3. Poll for token (standard RFC 8628)
    const tokens = await pollDeviceCodeToken(
      oauthConfig.tokenUrl,
      deviceCode.device_code,
      oauthConfig.clientId,
      deviceCode.interval,
      deviceCode.expires_in,
    );

    // 4. Extract accountId if id_token present
    const accountId = tokens.id_token
      ? extractAccountIdFromIdToken(tokens.id_token)
      : undefined;

    // 5. Store credential
    storeOAuthCredential(providerId, tokens, accountId);

    process.stderr.write("\n" + green("\u2713 Authentication successful!") + "\n");
    if (accountId) {
      process.stderr.write(dim("Account ID: " + accountId) + "\n");
    }
    process.stderr.write(dim("Use with: devagent --provider " + providerId + ' "your query"') + "\n");
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    process.stderr.write("\n" + red("\u2717 Authentication failed: " + msg) + "\n");
  }
}

// ─── Status ─────────────────────────────────────────────────

function authStatus(): void {
  const store = new CredentialStore();
  const stored = store.all();

  process.stderr.write(bold("Credential status") + "\n\n");

  // Check stored credentials
  const providers = new Map<string, { source: string; masked: string; extra?: string }>();

  for (const [id, cred] of Object.entries(stored)) {
    if (cred.type === "oauth") {
      const expiryStr = cred.expiresAt != null
        ? formatExpiry(cred.expiresAt)
        : "no expiry";
      providers.set(id, {
        source: "oauth",
        masked: maskToken(cred.accessToken),
        extra: expiryStr,
      });
    } else {
      providers.set(id, {
        source: "credentials",
        masked: maskKey(cred.key),
      });
    }
  }

  // Check env vars
  const envKey = process.env["DEVAGENT_API_KEY"];
  if (envKey) {
    providers.set("(DEVAGENT_API_KEY)", {
      source: "env",
      masked: maskKey(envKey),
    });
  }
  for (const p of KNOWN_PROVIDERS) {
    if (!p.envVar) continue;
    const val = process.env[p.envVar];
    if (val) {
      providers.set(p.id, {
        source: `env:${p.envVar}`,
        masked: maskKey(val),
      });
    }
  }

  if (providers.size === 0) {
    process.stderr.write(
      dim("No credentials configured.") + "\n" +
      dim('Run "devagent auth login" to add one.') + "\n",
    );
    return;
  }

  // Print table
  const maxId = Math.max(...[...providers.keys()].map((k) => k.length), 8);
  const maxSrc = Math.max(
    ...[...providers.values()].map((v) => v.source.length),
    6,
  );

  process.stderr.write(
    dim("Provider".padEnd(maxId + 2) + "Source".padEnd(maxSrc + 2) + "Key") +
      "\n",
  );

  for (const [id, info] of providers) {
    let line = cyan(id.padEnd(maxId + 2)) +
      info.source.padEnd(maxSrc + 2) +
      dim(info.masked);
    if (info.extra) {
      line += "  " + dim(info.extra);
    }
    process.stderr.write(line + "\n");
  }
}

// ─── Logout ─────────────────────────────────────────────────

async function authLogout(): Promise<void> {
  const store = new CredentialStore();
  const stored = store.all();
  const ids = Object.keys(stored);

  if (ids.length === 0) {
    process.stderr.write(dim("No stored credentials to remove.") + "\n");
    return;
  }

  const rl = createInterface({
    input: process.stdin,
    output: process.stderr,
  });

  const question = (prompt: string): Promise<string> =>
    new Promise((resolve) => rl.question(prompt, resolve));

  try {
    process.stderr.write(bold("Remove stored credential") + "\n\n");

    for (let i = 0; i < ids.length; i++) {
      const id = ids[i]!;
      const cred = stored[id]!;
      const masked = credentialMask(cred);
      process.stderr.write(
        `  ${cyan(String(i + 1))}. ${id} ${dim(masked)}\n`,
      );
    }
    if (ids.length > 1) {
      process.stderr.write(`  ${cyan(String(ids.length + 1))}. ${yellow("All")}\n`);
    }
    process.stderr.write("\n");

    const choice = await question(cyan("> "));
    const choiceNum = parseInt(choice.trim(), 10);

    if (choiceNum >= 1 && choiceNum <= ids.length) {
      const id = ids[choiceNum - 1]!;
      store.remove(id);
      process.stderr.write(green("\u2713 Removed credential for " + id) + "\n");
    } else if (ids.length > 1 && choiceNum === ids.length + 1) {
      for (const id of ids) {
        store.remove(id);
      }
      process.stderr.write(green("\u2713 Removed all stored credentials") + "\n");
    } else {
      process.stderr.write(red("Invalid selection.") + "\n");
    }
  } finally {
    rl.close();
  }
}

// ─── Helpers ────────────────────────────────────────────────

function maskKey(key: string): string {
  if (key.length <= 8) return "****";
  return key.slice(0, 4) + "..." + key.slice(-4);
}

function maskToken(token: string): string {
  if (token.length <= 12) return "****";
  return token.slice(0, 6) + "..." + token.slice(-4);
}

/** Get a masked display string for any credential type. */
function credentialMask(cred: CredentialInfo): string {
  if (cred.type === "oauth") {
    return `(oauth) ${maskToken(cred.accessToken)}`;
  }
  return maskKey(cred.key);
}

/** Format an expiry timestamp for display. */
function formatExpiry(expiresAt: number): string {
  const remaining = expiresAt - Date.now();
  if (remaining <= 0) return red("expired");
  const minutes = Math.floor(remaining / 60_000);
  if (minutes < 60) return yellow(`expires in ${minutes}m`);
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `expires in ${hours}h`;
  const days = Math.floor(hours / 24);
  return `expires in ${days}d`;
}

/** Store an OAuth credential from a token response. */
function storeOAuthCredential(
  providerId: string,
  tokens: { access_token: string; refresh_token?: string; expires_in?: number },
  accountId: string | undefined,
): void {
  const store = new CredentialStore();
  store.set(providerId, {
    type: "oauth",
    accessToken: tokens.access_token,
    // GitHub OAuth tokens don't have refresh_token or expires_in
    ...(tokens.refresh_token ? { refreshToken: tokens.refresh_token } : {}),
    ...(tokens.expires_in ? { expiresAt: Date.now() + tokens.expires_in * 1000 } : {}),
    ...(accountId ? { accountId } : {}),
    storedAt: Date.now(),
  });
}
