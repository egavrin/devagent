import type { ProviderConfig } from "./types.js";
import type { CredentialInfo } from "./credentials.js";

export type ProviderCredentialMode = "api" | "oauth" | "none";

export interface ProviderCredentialDescriptor {
  readonly id: string;
  readonly envVar: string | null;
  readonly hint: string;
  readonly credentialMode: ProviderCredentialMode;
}

export interface ResolvedProviderCredentialStatus {
  readonly providerId: string;
  readonly credentialMode: ProviderCredentialMode;
  readonly hasCredential: boolean;
  readonly source: "provider-config" | "top-level-config" | "env" | "stored" | "missing" | "not-required";
  readonly envVar?: string;
  readonly apiKey?: string;
}

const PROVIDER_CREDENTIALS: readonly ProviderCredentialDescriptor[] = [
  { id: "anthropic", envVar: "ANTHROPIC_API_KEY", hint: "set ANTHROPIC_API_KEY or devagent auth login", credentialMode: "api" },
  { id: "openai", envVar: "OPENAI_API_KEY", hint: "set OPENAI_API_KEY or devagent auth login", credentialMode: "api" },
  { id: "devagent-api", envVar: "DEVAGENT_API_KEY", hint: "set DEVAGENT_API_KEY or devagent auth login", credentialMode: "api" },
  { id: "deepseek", envVar: "DEEPSEEK_API_KEY", hint: "set DEEPSEEK_API_KEY or devagent auth login", credentialMode: "api" },
  { id: "openrouter", envVar: "OPENROUTER_API_KEY", hint: "set OPENROUTER_API_KEY or devagent auth login", credentialMode: "api" },
  { id: "chatgpt", envVar: null, hint: "devagent auth login (ChatGPT Plus/Pro)", credentialMode: "oauth" },
  { id: "github-copilot", envVar: null, hint: "devagent auth login (GitHub device flow)", credentialMode: "oauth" },
  { id: "ollama", envVar: null, hint: "local — no API key needed (ollama must be running)", credentialMode: "none" },
] as const;

export function listProviderCredentialDescriptors(): ReadonlyArray<ProviderCredentialDescriptor> {
  return PROVIDER_CREDENTIALS;
}

export function getProviderCredentialDescriptor(providerId: string): ProviderCredentialDescriptor | undefined {
  return PROVIDER_CREDENTIALS.find((provider) => provider.id === providerId);
}

export function getProviderCredentialEnvVar(providerId: string): string | null {
  return getProviderCredentialDescriptor(providerId)?.envVar ?? null;
}

export function resolveProviderCredentialStatus(opts: {
  readonly providerId: string;
  readonly providerConfig?: ProviderConfig;
  readonly topLevelApiKey?: string;
  readonly storedCredential?: CredentialInfo | null;
  readonly env?: NodeJS.ProcessEnv;
}): ResolvedProviderCredentialStatus {
  const descriptor = getProviderCredentialDescriptor(opts.providerId);
  const credentialMode = descriptor?.credentialMode ?? "api";
  const providerConfig = opts.providerConfig;
  const env = opts.env ?? process.env;
  const storedCredential = opts.storedCredential ?? null;

  if (credentialMode === "none") {
    return {
      providerId: opts.providerId,
      credentialMode,
      hasCredential: true,
      source: "not-required",
    };
  }

  if (credentialMode === "oauth") {
    if (providerConfig?.oauthToken) {
      return {
        providerId: opts.providerId,
        credentialMode,
        hasCredential: true,
        source: "provider-config",
      };
    }
    if (storedCredential?.type === "oauth") {
      return {
        providerId: opts.providerId,
        credentialMode,
        hasCredential: true,
        source: "stored",
      };
    }
    return {
      providerId: opts.providerId,
      credentialMode,
      hasCredential: false,
      source: "missing",
    };
  }

  if (providerConfig?.apiKey) {
    return {
      providerId: opts.providerId,
      credentialMode,
      hasCredential: true,
      source: "provider-config",
      apiKey: providerConfig.apiKey,
    };
  }

  if (opts.topLevelApiKey) {
    return {
      providerId: opts.providerId,
      credentialMode,
      hasCredential: true,
      source: "top-level-config",
      apiKey: opts.topLevelApiKey,
    };
  }

  if (descriptor?.envVar) {
    const envValue = env[descriptor.envVar];
    if (envValue) {
      return {
        providerId: opts.providerId,
        credentialMode,
        hasCredential: true,
        source: "env",
        envVar: descriptor.envVar,
        apiKey: envValue,
      };
    }
  }

  if (storedCredential?.type === "api") {
    return {
      providerId: opts.providerId,
      credentialMode,
      hasCredential: true,
      source: "stored",
      apiKey: storedCredential.key,
    };
  }

  return {
    providerId: opts.providerId,
    credentialMode,
    hasCredential: false,
    source: "missing",
  };
}

export function formatResolvedCredentialSource(status: ResolvedProviderCredentialStatus): string {
  switch (status.source) {
    case "provider-config":
      return "config";
    case "top-level-config":
      return "config";
    case "env":
      return status.envVar ? `env (${status.envVar})` : "env";
    case "stored":
      return status.credentialMode === "oauth" ? "stored oauth" : "stored api key";
    case "not-required":
      return `missing (not required for ${status.providerId})`;
    case "missing":
      return "missing";
  }
}
