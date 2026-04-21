import { CredentialStore } from "./credentials.js";
import type { OAuthCredential } from "./credentials.js";
import { OAuthError, extractErrorMessage } from "./errors.js";
import { getOAuthProvider } from "./oauth-providers.js";
import { exchangeCopilotSessionToken, refreshAccessToken } from "./oauth.js";
import type { DevAgentConfig, ProviderConfig } from "./types.js";

export async function resolveProviderCredentials(
  config: DevAgentConfig,
): Promise<DevAgentConfig> {
  const credentialStore = new CredentialStore();
  const updatedProviders: Record<string, ProviderConfig> = { ...config.providers };
  const providersToCheck = new Set(Object.keys(updatedProviders));
  providersToCheck.add(config.provider);

  for (const key of providersToCheck) {
    const provConfig = updatedProviders[key];
    if (provConfig?.apiKey || provConfig?.oauthToken) continue;
    const stored = credentialStore.get(key);
    if (stored?.type !== "oauth") continue;
    const accessToken = await resolveAccessToken({ key, stored, credentialStore });
    updatedProviders[key] = await buildOAuthProviderConfig(key, accessToken, stored.accountId, updatedProviders[key] ?? { model: config.model });
  }

  return { ...config, providers: updatedProviders };
}

async function resolveAccessToken(options: {
  readonly key: string;
  readonly stored: OAuthCredential;
  readonly credentialStore: CredentialStore;
}): Promise<string> {
  const isExpired = options.stored.expiresAt != null && options.stored.expiresAt < Date.now() + 60_000;
  if (!isExpired) return options.stored.accessToken;
  if (!options.stored.refreshToken) {
    throw new OAuthError(
      `OAuth token for "${options.key}" is expired and cannot be refreshed (no refresh token). Run "devagent auth login" to re-authenticate.`,
    );
  }
  const oauthConfig = getOAuthProvider(options.key);
  if (!oauthConfig) {
    throw new OAuthError(
      `OAuth token for "${options.key}" is expired but no OAuth config found to refresh. Run "devagent auth login" to re-authenticate.`,
    );
  }
  try {
    const newTokens = await refreshAccessToken(oauthConfig.tokenUrl, options.stored.refreshToken, oauthConfig.clientId);
    const updated: OAuthCredential = {
      type: "oauth",
      accessToken: newTokens.access_token,
      ...(newTokens.refresh_token ? { refreshToken: newTokens.refresh_token } : {}),
      ...(newTokens.expires_in ? { expiresAt: Date.now() + newTokens.expires_in * 1000 } : {}),
      accountId: options.stored.accountId,
      storedAt: Date.now(),
    };
    options.credentialStore.set(options.key, updated);
    return updated.accessToken;
  } catch (err) {
    throw new OAuthError(
      `Failed to refresh OAuth token for "${options.key}": ${extractErrorMessage(err)}. Run "devagent auth login" to re-authenticate.`,
    );
  }
}

async function buildOAuthProviderConfig(
  key: string,
  accessToken: string,
  accountId: string | undefined,
  existingConfig: ProviderConfig,
): Promise<ProviderConfig> {
  if (key !== "github-copilot") {
    return { ...existingConfig, oauthToken: accessToken, oauthAccountId: accountId };
  }
  try {
    const session = await exchangeCopilotSessionToken(accessToken);
    return {
      ...existingConfig,
      oauthToken: session.token,
      baseUrl: session.endpoint ?? existingConfig.baseUrl,
    };
  } catch (err) {
    throw new OAuthError(
      `Failed to obtain Copilot session token: ${extractErrorMessage(err)}. Run "devagent auth login" to re-authenticate.`,
    );
  }
}
