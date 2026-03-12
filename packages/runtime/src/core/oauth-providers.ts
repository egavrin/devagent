/**
 * Static configuration for OAuth-capable providers.
 * Client IDs and endpoints sourced from OpenCode's implementation.
 */

// ─── Types ──────────────────────────────────────────────────

export interface OAuthProviderConfig {
  readonly providerId: string;
  readonly displayName: string;
  readonly tokenUrl: string;
  readonly clientId: string;
  readonly scopes: readonly string[];
  // Browser PKCE flow
  readonly supportsBrowser: boolean;
  readonly authorizationUrl?: string;
  readonly callbackPort?: number;
  // Device code flow
  readonly supportsDeviceCode: boolean;
  readonly deviceCodeUrl?: string;
  readonly deviceVerificationUrl?: string;
}

// ─── Registry ───────────────────────────────────────────────

export const OAUTH_PROVIDERS: readonly OAuthProviderConfig[] = [
  {
    providerId: "chatgpt",
    displayName: "ChatGPT (Pro/Plus)",
    tokenUrl: "https://auth.openai.com/oauth/token",
    clientId: "app_EMoamEEZ73f0CkXaXp7hrann",
    scopes: ["openid", "profile", "email", "offline_access"],
    supportsBrowser: true,
    authorizationUrl: "https://auth.openai.com/oauth/authorize",
    callbackPort: 1455,
    supportsDeviceCode: true,
    deviceCodeUrl: "https://auth.openai.com/api/accounts/deviceauth/usercode",
    deviceVerificationUrl: "https://auth.openai.com/codex/device",
  },
  {
    providerId: "github-copilot",
    displayName: "GitHub Copilot",
    tokenUrl: "https://github.com/login/oauth/access_token",
    // This is GitHub Copilot's public OAuth App client ID (Iv1. prefix = OAuth App).
    // All official Copilot extensions (VS Code, Neovim, JetBrains) use this same ID.
    // The gho_* token obtained with this client_id grants access to copilot_internal APIs.
    clientId: "Iv1.b507a08c87ecfe98",
    scopes: ["read:user"],
    supportsBrowser: false,
    supportsDeviceCode: true,
    deviceCodeUrl: "https://github.com/login/device/code",
    deviceVerificationUrl: "https://github.com/login/device",
  },
];

/**
 * Look up OAuth configuration by provider ID.
 * Returns null if the provider doesn't support OAuth.
 */
export function getOAuthProvider(id: string): OAuthProviderConfig | null {
  return OAUTH_PROVIDERS.find((p) => p.providerId === id) ?? null;
}
