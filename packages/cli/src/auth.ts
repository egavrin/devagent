/**
 * Auth subcommands — login, status, logout.
 * Manages persistent API key and OAuth token storage via CredentialStore.
 * Supports:
 *   - API key login (anthropic, openai, custom)
 *   - Browser OAuth PKCE (chatgpt)
 *   - ChatGPT device code flow (non-standard 3-step)
 *   - GitHub device code flow (RFC 8628)
 */

import {
  CredentialStore,
  getOAuthProvider,
  listProviderCredentialDescriptors,
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
  extractErrorMessage,
} from "@devagent/runtime";
import { createInterface } from "node:readline";

import { bold, cyan, dim, green, red, yellow } from "./format.js";
import type { OAuthProviderConfig, CredentialInfo } from "@devagent/runtime";

// ─── Known Providers ────────────────────────────────────────

interface ProviderEntry {
  readonly id: string;
  readonly name: string;
  readonly hint: string;
  readonly authMethods: readonly ("api-key" | "browser-oauth" | "device-code")[];
}

type AuthMethod = "api-key" | "browser-oauth" | "device-code";
type AuthQuestion = (prompt: string) => Promise<string>;

interface PromptSession {
  readonly question: AuthQuestion;
  readonly close: () => void;
}

const PROVIDER_NAMES: Readonly<Record<string, string>> = {
  anthropic: "Anthropic (API key)",
  openai: "OpenAI (API key)",
  "devagent-api": "Devagent API (gateway key)",
  chatgpt: "ChatGPT (Pro/Plus account)",
  deepseek: "DeepSeek (API key)",
  openrouter: "OpenRouter (API key)",
  "github-copilot": "GitHub Copilot",
};

const PROVIDER_AUTH_METHODS: Readonly<Record<string, readonly ("api-key" | "browser-oauth" | "device-code")[]>> = {
  anthropic: ["api-key"],
  openai: ["api-key"],
  "devagent-api": ["api-key"],
  chatgpt: ["device-code"],
  deepseek: ["api-key"],
  openrouter: ["api-key"],
  "github-copilot": ["device-code"],
};

const KNOWN_PROVIDERS: readonly ProviderEntry[] = listProviderCredentialDescriptors()
  .filter((provider) => provider.id !== "ollama")
  .map((provider) => ({
    id: provider.id,
    name: PROVIDER_NAMES[provider.id] ?? provider.id,
    hint: provider.hint,
    authMethods: PROVIDER_AUTH_METHODS[provider.id] ?? ["api-key"],
  }));

function renderAuthHelpText(): string {
  return `Usage:
  devagent auth login
  devagent auth status
  devagent auth logout [provider]
  devagent auth logout --all

Manage stored provider credentials for DevAgent.`;
}

// ─── Entry Point ────────────────────────────────────────────

export async function runAuthCommand(subcommand: string, args: ReadonlyArray<string> = []): Promise<void> {
  if (subcommand === "--help" || subcommand === "-h" || args.includes("--help") || args.includes("-h")) {
    process.stdout.write(renderAuthHelpText() + "\n");
    return;
  }

  switch (subcommand) {
    case "login":
      await authLogin();
      break;
    case "status":
      authStatus();
      break;
    case "logout":
      await authLogout(args);
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
  const prompt = createPromptSession();

  try {
    process.stderr.write(bold("Add credential") + "\n\n");
    const providerEntry = await promptProviderSelection(prompt.question);
    if (!providerEntry) return;
    process.stderr.write("\n");

    const oauthConfig = getOAuthProvider(providerEntry.id);
    const authMethods = providerEntry?.authMethods ?? ["api-key"];
    if (oauthConfig && authMethods.some((m) => m !== "api-key")) {
      const authMethod = await promptAuthMethod(authMethods, prompt.question);
      process.stderr.write("\n");
      prompt.close();
      await runOAuthLoginMethod(providerEntry.id, oauthConfig, authMethod);
      return;
    }

    if (providerEntry.hint) {
      process.stderr.write(dim(providerEntry.hint) + "\n");
    }
    await authLoginApiKey(providerEntry.id, prompt.question);
  } finally {
    prompt.close();
  }
}

function createPromptSession(): PromptSession {
  const rl = createInterface({ input: process.stdin, output: process.stderr });
  return {
    question: (prompt: string) => new Promise((resolve) => rl.question(prompt, resolve)),
    close: () => rl.close(),
  };
}

async function promptProviderSelection(question: AuthQuestion): Promise<ProviderEntry | undefined> {
  process.stderr.write("Select provider:\n");
  for (let i = 0; i < KNOWN_PROVIDERS.length; i++) {
    process.stderr.write(`  ${cyan(String(i + 1))}. ${KNOWN_PROVIDERS[i]!.name}\n`);
  }
  process.stderr.write("\n");

  const choice = await question(cyan("> "));
  const provider = resolveProviderSelection(choice);
  if (!provider) {
    process.stderr.write(red("Invalid selection.") + "\n");
  }
  return provider;
}

function resolveProviderSelection(choice: string): ProviderEntry | undefined {
  const choiceNum = parseInt(choice.trim(), 10);
  if (choiceNum >= 1 && choiceNum <= KNOWN_PROVIDERS.length) {
    return KNOWN_PROVIDERS[choiceNum - 1]!;
  }
  return KNOWN_PROVIDERS.find((provider) => provider.id === choice.trim().toLowerCase());
}

async function promptAuthMethod(
  authMethods: readonly AuthMethod[],
  question: AuthQuestion,
): Promise<AuthMethod> {
  if (authMethods.length === 1) {
    return authMethods[0]!;
  }
  process.stderr.write("Authentication method:\n");
  const options = buildAuthMethodOptions(authMethods);
  for (let i = 0; i < options.length; i++) {
    process.stderr.write(`  ${cyan(String(i + 1))}. ${options[i]!.label}\n`);
  }
  process.stderr.write("\n");

  const methodChoice = await question(cyan("> "));
  const methodNum = parseInt(methodChoice.trim(), 10);
  return methodNum >= 1 && methodNum <= options.length
    ? options[methodNum - 1]!.method
    : options[0]!.method;
}

function buildAuthMethodOptions(authMethods: readonly AuthMethod[]): Array<{ label: string; method: AuthMethod }> {
  return [
    authMethods.includes("browser-oauth")
      ? { label: "Browser login (opens browser)", method: "browser-oauth" as const }
      : null,
    authMethods.includes("device-code")
      ? { label: "Device code (paste code in browser)", method: "device-code" as const }
      : null,
    authMethods.includes("api-key") ? { label: "API key", method: "api-key" as const } : null,
  ].filter((option): option is { label: string; method: AuthMethod } => option !== null);
}

async function runOAuthLoginMethod(
  providerId: string,
  oauthConfig: OAuthProviderConfig,
  authMethod: AuthMethod,
): Promise<void> {
  if (authMethod === "browser-oauth") {
    await authLoginBrowserOAuth(providerId, oauthConfig);
    return;
  }
  if (authMethod === "device-code") {
    await authLoginDeviceFlow(providerId, oauthConfig);
    return;
  }
  const prompt = createPromptSession();
  try {
    await authLoginApiKey(providerId, prompt.question);
  } finally {
    prompt.close();
  }
}

async function authLoginDeviceFlow(providerId: string, oauthConfig: OAuthProviderConfig): Promise<void> {
  if (providerId === "chatgpt") {
    await authLoginChatGPTDeviceCode(providerId, oauthConfig);
    return;
  }
  await authLoginDeviceCode(providerId, oauthConfig);
}

// ─── API Key Login ──────────────────────────────────────────

async function authLoginApiKey(
  providerId: string,
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

    showAuthSuccess(providerId, accountId);
  } catch (err) {
    showAuthFailure(err);
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

    showAuthSuccess(providerId, accountId);
  } catch (err) {
    showAuthFailure(err);
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

    showAuthSuccess(providerId, accountId);
  } catch (err) {
    showAuthFailure(err);
  }
}

// ─── Status ─────────────────────────────────────────────────

function authStatus(): void {
  const store = new CredentialStore();
  const stored = store.all();

  process.stderr.write(bold("Credential status") + "\n\n");
  const providers = collectCredentialStatusEntries(stored, process.env);

  if (providers.length === 0) {
    process.stderr.write(
      dim("No credentials configured.") + "\n" +
      dim('Run "devagent auth login" to add one.') + "\n",
    );
    return;
  }

  // Print table
  const maxId = Math.max(...providers.map((entry) => entry.id.length), 8);
  const maxSrc = Math.max(
    ...providers.map((entry) => entry.source.length),
    6,
  );

  process.stderr.write(
    dim("Provider".padEnd(maxId + 2) + "Source".padEnd(maxSrc + 2) + "Key") +
      "\n",
  );

  for (const entry of providers) {
    let line = cyan(entry.id.padEnd(maxId + 2)) +
      entry.source.padEnd(maxSrc + 2) +
      dim(entry.masked);
    if (entry.extra) {
      line += "  " + dim(entry.extra);
    }
    process.stderr.write(line + "\n");
  }
}

export function collectCredentialStatusEntries(
  stored: Readonly<Record<string, CredentialInfo>>,
  env: NodeJS.ProcessEnv,
): Array<{ id: string; source: string; masked: string; extra?: string }> {
  const providers = new Map<string, { source: string; masked: string; extra?: string }>();

  for (const [id, cred] of Object.entries(stored)) {
    if (cred.type === "oauth") {
      providers.set(id, {
        source: "oauth",
        masked: maskToken(cred.accessToken),
        extra: cred.expiresAt != null ? formatExpiry(cred.expiresAt) : "no expiry",
      });
    } else {
      providers.set(id, {
        source: "credentials",
        masked: maskKey(cred.key),
      });
    }
  }

  for (const descriptor of listProviderCredentialDescriptors()) {
    if (!descriptor.envVar) continue;
    const value = env[descriptor.envVar];
    if (!value) continue;
    providers.set(descriptor.id, {
      source: `env:${descriptor.envVar}`,
      masked: maskKey(value),
    });
  }

  return [...providers.entries()]
    .map(([id, info]) => ({ id, ...info }))
    .sort((a, b) => a.id.localeCompare(b.id));
}

// ─── Logout ─────────────────────────────────────────────────
async function authLogout(args: ReadonlyArray<string> = []): Promise<void> {
  const store = new CredentialStore();
  const stored = store.all();
  const ids = Object.keys(stored);

  if (ids.length === 0) {
    process.stderr.write(dim("No stored credentials to remove.") + "\n");
    return;
  }

  if (handleDirectLogout(args, store, stored, ids)) return;

  const prompt = createPromptSession();
  try {
    await promptLogoutSelection(prompt.question, store, stored, ids);
  } finally {
    prompt.close();
  }
}

function handleDirectLogout(
  args: ReadonlyArray<string>,
  store: CredentialStore,
  stored: Readonly<Record<string, CredentialInfo>>,
  ids: ReadonlyArray<string>,
): boolean {
  const target = args[0];
  if (args.length > 1) {
    writeLogoutUsageAndExit(2);
  }
  if (target === "--all") {
    removeAllCredentials(store, ids);
    return true;
  }
  if (target) {
    removeTargetCredential(store, stored, target);
    return true;
  }
  if (!process.stdin.isTTY) {
    writeLogoutUsageAndExit(2);
  }
  return false;
}

function writeLogoutUsageAndExit(code: number): never {
  process.stderr.write(red("Usage: devagent auth logout [provider] | --all") + "\n");
  process.exit(code);
}

function removeAllCredentials(store: CredentialStore, ids: ReadonlyArray<string>): void {
  for (const id of ids) {
    store.remove(id);
  }
  process.stderr.write(green("\u2713 Removed all stored credentials") + "\n");
}

function removeTargetCredential(
  store: CredentialStore,
  stored: Readonly<Record<string, CredentialInfo>>,
  target: string,
): void {
  if (!stored[target]) {
    process.stderr.write(red(`No stored credential found for ${target}.`) + "\n");
    process.exit(1);
  }
  store.remove(target);
  process.stderr.write(green("\u2713 Removed credential for " + target) + "\n");
}

async function promptLogoutSelection(
  question: AuthQuestion,
  store: CredentialStore,
  stored: Readonly<Record<string, CredentialInfo>>,
  ids: ReadonlyArray<string>,
): Promise<void> {
  process.stderr.write(bold("Remove stored credential") + "\n\n");
  writeLogoutOptions(stored, ids);

  const choice = await question(cyan("> "));
  const choiceNum = parseInt(choice.trim(), 10);
  if (choiceNum >= 1 && choiceNum <= ids.length) {
    removeSelectedCredential(store, ids[choiceNum - 1]!);
    return;
  }
  if (ids.length > 1 && choiceNum === ids.length + 1) {
    removeAllCredentials(store, ids);
    return;
  }
  process.stderr.write(red("Invalid selection.") + "\n");
}

function writeLogoutOptions(
  stored: Readonly<Record<string, CredentialInfo>>,
  ids: ReadonlyArray<string>,
): void {
  for (let i = 0; i < ids.length; i++) {
    const id = ids[i]!;
    const masked = credentialMask(stored[id]!);
    process.stderr.write(`  ${cyan(String(i + 1))}. ${id} ${dim(masked)}\n`);
  }
  if (ids.length > 1) {
    process.stderr.write(`  ${cyan(String(ids.length + 1))}. ${yellow("All")}\n`);
  }
  process.stderr.write("\n");
}

function removeSelectedCredential(store: CredentialStore, id: string): void {
  store.remove(id);
  process.stderr.write(green("\u2713 Removed credential for " + id) + "\n");
}

// ─── Auth Result Display ────────────────────────────────────

/** Shared success message for all OAuth and device-code auth flows. */
function showAuthSuccess(providerId: string, accountId?: string): void {
  process.stderr.write("\n" + green("\u2713 Authentication successful!") + "\n");
  if (accountId) {
    process.stderr.write(dim("Account ID: " + accountId) + "\n");
  }
  process.stderr.write(dim("Use with: devagent --provider " + providerId + ' "your query"') + "\n");
}

/** Shared error message for all OAuth and device-code auth flows. */
function showAuthFailure(err: unknown): void {
  const msg = extractErrorMessage(err);
  process.stderr.write("\n" + red("\u2717 Authentication failed: " + msg) + "\n");
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
