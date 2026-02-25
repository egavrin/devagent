import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  generatePKCE,
  generateState,
  exchangeCodeForTokens,
  refreshAccessToken,
  requestDeviceCode,
  pollDeviceCodeToken,
  requestChatGPTDeviceCode,
  pollChatGPTDeviceAuth,
  exchangeChatGPTDeviceToken,
  extractAccountIdFromIdToken,
} from "./oauth.js";
import { OAuthError } from "./errors.js";

// ─── Mock fetch ─────────────────────────────────────────────

const mockFetch = vi.fn();

beforeEach(() => {
  mockFetch.mockReset();
  vi.stubGlobal("fetch", mockFetch);
});

afterEach(() => {
  vi.restoreAllMocks();
});

// ─── PKCE ───────────────────────────────────────────────────

describe("generatePKCE", () => {
  it("produces verifier and challenge strings", async () => {
    const pkce = await generatePKCE();
    expect(pkce.verifier).toBeTruthy();
    expect(pkce.challenge).toBeTruthy();
    expect(pkce.verifier).not.toBe(pkce.challenge);
  });

  it("produces base64url-encoded strings (no +, /, =)", async () => {
    const pkce = await generatePKCE();
    expect(pkce.verifier).not.toMatch(/[+/=]/);
    expect(pkce.challenge).not.toMatch(/[+/=]/);
  });

  it("produces unique pairs on each call", async () => {
    const a = await generatePKCE();
    const b = await generatePKCE();
    expect(a.verifier).not.toBe(b.verifier);
  });
});

describe("generateState", () => {
  it("produces a hex string", () => {
    const state = generateState();
    expect(state).toMatch(/^[0-9a-f]+$/);
  });

  it("produces a 32-character hex string (16 bytes)", () => {
    const state = generateState();
    expect(state).toHaveLength(32);
  });

  it("produces unique values", () => {
    const a = generateState();
    const b = generateState();
    expect(a).not.toBe(b);
  });
});

// ─── Token Exchange ─────────────────────────────────────────

describe("exchangeCodeForTokens", () => {
  it("sends correct POST body and returns tokens", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        access_token: "access-123",
        refresh_token: "refresh-456",
        expires_in: 3600,
        id_token: "id-789",
      }),
    });

    const tokens = await exchangeCodeForTokens("https://auth.example.com/token", {
      grant_type: "authorization_code",
      code: "auth-code",
      client_id: "my-client",
    });

    expect(tokens.access_token).toBe("access-123");
    expect(tokens.refresh_token).toBe("refresh-456");
    expect(tokens.expires_in).toBe(3600);

    expect(mockFetch).toHaveBeenCalledWith("https://auth.example.com/token", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: "grant_type=authorization_code&code=auth-code&client_id=my-client",
    });
  });

  it("throws OAuthError on non-200 response", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 400,
      text: async () => '{"error":"invalid_grant"}',
    });

    await expect(
      exchangeCodeForTokens("https://auth.example.com/token", {
        grant_type: "authorization_code",
        code: "bad-code",
        client_id: "my-client",
      }),
    ).rejects.toThrow(OAuthError);
  });
});

describe("refreshAccessToken", () => {
  it("calls exchangeCodeForTokens with refresh_token grant", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        access_token: "new-access",
        refresh_token: "new-refresh",
        expires_in: 3600,
      }),
    });

    const tokens = await refreshAccessToken(
      "https://auth.example.com/token",
      "old-refresh",
      "my-client",
    );

    expect(tokens.access_token).toBe("new-access");

    const body = mockFetch.mock.calls[0]![1]!.body as string;
    expect(body).toContain("grant_type=refresh_token");
    expect(body).toContain("refresh_token=old-refresh");
    expect(body).toContain("client_id=my-client");
  });
});

// ─── Device Code ────────────────────────────────────────────

describe("requestDeviceCode", () => {
  it("returns device code response", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        user_code: "ABCD-1234",
        device_code: "device-code-123",
        verification_uri: "https://example.com/device",
        interval: 5,
        expires_in: 900,
      }),
    });

    const result = await requestDeviceCode(
      "https://example.com/device/code",
      "client-id",
      ["read:user"],
    );

    expect(result.user_code).toBe("ABCD-1234");
    expect(result.device_code).toBe("device-code-123");
    expect(result.verification_uri).toBe("https://example.com/device");

    const body = mockFetch.mock.calls[0]![1]!.body as string;
    expect(body).toContain("client_id=client-id");
    expect(body).toContain("scope=read%3Auser");
  });

  it("sends no scope param when scopes array is empty", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        user_code: "X",
        device_code: "Y",
        verification_uri: "Z",
        interval: 5,
        expires_in: 900,
      }),
    });

    await requestDeviceCode("https://example.com/device/code", "client-id", []);

    const body = mockFetch.mock.calls[0]![1]!.body as string;
    expect(body).not.toContain("scope");
  });

  it("throws OAuthError on failure", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
      text: async () => "Internal Server Error",
    });

    await expect(
      requestDeviceCode("https://example.com/device/code", "client-id", []),
    ).rejects.toThrow(OAuthError);
  });
});

describe("pollDeviceCodeToken", () => {
  it("returns tokens after authorization_pending then success", async () => {
    // First call: authorization_pending
    mockFetch.mockResolvedValueOnce({
      json: async () => ({ error: "authorization_pending" }),
    });
    // Second call: success
    mockFetch.mockResolvedValueOnce({
      json: async () => ({
        access_token: "access-ok",
        refresh_token: "refresh-ok",
        expires_in: 3600,
      }),
    });

    // Note: minimum poll interval is enforced at 5s internally, but we set
    // expiresIn long enough and just verify the final result
    const tokens = await pollDeviceCodeToken(
      "https://example.com/token",
      "device-code",
      "client-id",
      0.01, // Will be clamped to minimum 5s
      120,
    );

    expect(tokens.access_token).toBe("access-ok");
    expect(tokens.refresh_token).toBe("refresh-ok");
    // At least 2 calls: one pending, one success
    expect(mockFetch.mock.calls.length).toBeGreaterThanOrEqual(2);
  }, 30000);

  it("throws on non-pending/slow_down error", async () => {
    mockFetch.mockResolvedValueOnce({
      json: async () => ({ error: "access_denied" }),
    });

    await expect(
      pollDeviceCodeToken(
        "https://example.com/token",
        "device-code",
        "client-id",
        0.01,
        120,
      ),
    ).rejects.toThrow("access_denied");
  }, 30000);
});

// ─── ChatGPT Device Code Flow ────────────────────────────────

describe("requestChatGPTDeviceCode", () => {
  it("sends JSON body and returns parsed response", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        device_auth_id: "dauth-123",
        user_code: "ABCD-1234",
        interval: "5",
        expires_in: "900",
      }),
    });

    const result = await requestChatGPTDeviceCode(
      "https://auth.openai.com/api/accounts/deviceauth/usercode",
      "app_test",
    );

    expect(result.device_auth_id).toBe("dauth-123");
    expect(result.user_code).toBe("ABCD-1234");
    expect(result.interval).toBe(5); // Parsed from string
    expect(result.expires_in).toBe(900); // Parsed from string

    // Verify JSON body was sent
    const call = mockFetch.mock.calls[0]!;
    const headers = call[1]!.headers as Record<string, string>;
    expect(headers["Content-Type"]).toBe("application/json");
    const body = JSON.parse(call[1]!.body as string);
    expect(body.client_id).toBe("app_test");
  });

  it("handles usercode alias", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        device_auth_id: "dauth-456",
        usercode: "WXYZ-5678", // Note: alias for user_code
        interval: 5,
      }),
    });

    const result = await requestChatGPTDeviceCode(
      "https://auth.openai.com/api/accounts/deviceauth/usercode",
      "app_test",
    );

    expect(result.user_code).toBe("WXYZ-5678");
  });

  it("throws OAuthError on failure", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 400,
      text: async () => '{"error":"bad_request"}',
    });

    await expect(
      requestChatGPTDeviceCode("https://example.com/usercode", "bad-client"),
    ).rejects.toThrow(OAuthError);
  });
});

describe("pollChatGPTDeviceAuth", () => {
  it("polls past 403 responses then returns authorization_code", async () => {
    // First call: 403 (pending)
    mockFetch.mockResolvedValueOnce({ status: 403, ok: false });
    // Second call: success
    mockFetch.mockResolvedValueOnce({
      status: 200,
      ok: true,
      json: async () => ({
        authorization_code: "auth-code-123",
        code_verifier: "verifier-abc",
      }),
    });

    const result = await pollChatGPTDeviceAuth(
      "https://auth.openai.com/api/accounts/deviceauth/token",
      "dauth-123",
      "ABCD-1234",
      0.01, // Clamped to 5s internally
      120,
    );

    expect(result.authorization_code).toBe("auth-code-123");
    expect(result.code_verifier).toBe("verifier-abc");
    expect(mockFetch.mock.calls.length).toBeGreaterThanOrEqual(2);
  }, 30000);

  it("polls past 404 responses", async () => {
    // 404 = also pending
    mockFetch.mockResolvedValueOnce({ status: 404, ok: false });
    mockFetch.mockResolvedValueOnce({
      status: 200,
      ok: true,
      json: async () => ({
        authorization_code: "auth-code-456",
        code_verifier: "verifier-def",
      }),
    });

    const result = await pollChatGPTDeviceAuth(
      "https://example.com/token",
      "dauth",
      "CODE",
      0.01,
      120,
    );

    expect(result.authorization_code).toBe("auth-code-456");
  }, 30000);

  it("throws on non-403/404 error", async () => {
    mockFetch.mockResolvedValueOnce({
      status: 500,
      ok: false,
      text: async () => "Internal Server Error",
    });

    await expect(
      pollChatGPTDeviceAuth("https://example.com/token", "dauth", "CODE", 0.01, 120),
    ).rejects.toThrow("polling error");
  }, 30000);
});

describe("exchangeChatGPTDeviceToken", () => {
  it("exchanges authorization_code with server-provided code_verifier", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        access_token: "access-final",
        refresh_token: "refresh-final",
        expires_in: 3600,
        id_token: "id-token-jwt",
      }),
    });

    const tokens = await exchangeChatGPTDeviceToken(
      "https://auth.openai.com/oauth/token",
      "app_test",
      "auth-code-123",
      "verifier-from-server",
    );

    expect(tokens.access_token).toBe("access-final");
    expect(tokens.id_token).toBe("id-token-jwt");

    // Verify form-urlencoded body
    const body = mockFetch.mock.calls[0]![1]!.body as string;
    expect(body).toContain("grant_type=authorization_code");
    expect(body).toContain("code=auth-code-123");
    expect(body).toContain("code_verifier=verifier-from-server");
    expect(body).toContain("redirect_uri=https%3A%2F%2Fauth.openai.com%2Fdeviceauth%2Fcallback");
  });
});

// ─── GitHub Copilot Token Exchange ───────────────────────────

describe("exchangeCopilotSessionToken", () => {
  it("returns session token on success", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        token: "tid=copilot-session-jwt",
        expires_at: Math.floor(Date.now() / 1000) + 1800,
        endpoints: { api: "https://api.individual.githubcopilot.com" },
      }),
    });

    const { exchangeCopilotSessionToken } = await import("./oauth.js");
    const result = await exchangeCopilotSessionToken("gho_test_token");

    expect(result.token).toBe("tid=copilot-session-jwt");
    expect(result.expiresAt).toBeGreaterThan(Date.now());
    expect(result.endpoint).toBe("https://api.individual.githubcopilot.com");

    // Verify headers
    const headers = mockFetch.mock.calls[0]![1]!.headers as Record<string, string>;
    expect(headers["Authorization"]).toBe("token gho_test_token");
    expect(headers["Editor-Version"]).toBeDefined();
    expect(headers["User-Agent"]).toContain("GitHubCopilot");
  });

  it("throws OAuthError on HTTP error", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 401,
      text: async () => '{"message":"Bad credentials"}',
    });

    const { exchangeCopilotSessionToken } = await import("./oauth.js");
    await expect(
      exchangeCopilotSessionToken("gho_invalid"),
    ).rejects.toThrow("Copilot token exchange failed");
  });

  it("throws when token field is missing", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ expires_at: 1234567890 }),
    });

    const { exchangeCopilotSessionToken } = await import("./oauth.js");
    await expect(
      exchangeCopilotSessionToken("gho_test"),
    ).rejects.toThrow("missing 'token' field");
  });
});

// ─── JWT Helpers ────────────────────────────────────────────

describe("extractAccountIdFromIdToken", () => {
  function makeJwt(claims: Record<string, unknown>): string {
    const header = Buffer.from(JSON.stringify({ alg: "none" })).toString("base64url");
    const payload = Buffer.from(JSON.stringify(claims)).toString("base64url");
    return `${header}.${payload}.signature`;
  }

  it("extracts chatgpt_account_id from claims", () => {
    const token = makeJwt({ chatgpt_account_id: "acct-123" });
    expect(extractAccountIdFromIdToken(token)).toBe("acct-123");
  });

  it("extracts from nested https://api.openai.com/auth claim", () => {
    const token = makeJwt({
      "https://api.openai.com/auth": { chatgpt_account_id: "acct-nested" },
    });
    expect(extractAccountIdFromIdToken(token)).toBe("acct-nested");
  });

  it("extracts from organizations array", () => {
    const token = makeJwt({ organizations: [{ id: "org-789" }] });
    expect(extractAccountIdFromIdToken(token)).toBe("org-789");
  });

  it("returns undefined for invalid token", () => {
    expect(extractAccountIdFromIdToken("not.a.jwt")).toBeUndefined();
    expect(extractAccountIdFromIdToken("")).toBeUndefined();
  });

  it("returns undefined when no account claims present", () => {
    const token = makeJwt({ sub: "user-123", email: "test@test.com" });
    expect(extractAccountIdFromIdToken(token)).toBeUndefined();
  });
});
