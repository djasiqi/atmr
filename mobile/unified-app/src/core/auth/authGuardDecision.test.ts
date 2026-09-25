/**
 * P0-4/5 — AuthGuard decision + classification recovery + single-flight.
 *
 * Les mocks doivent être déclarés avant l'import du coordinateur (hoisting Jest).
 */
import { beforeEach, describe, expect, it, jest } from "@jest/globals";

const mockRefreshAuthTokenNow = jest.fn();
const mockSessionResumeRequest = jest.fn();
const mockGetLastRefreshErrorCode = jest.fn();
const mockReadPendingResume = jest.fn();

jest.mock("../api/client", () => ({
  refreshAuthTokenNow: (...args: unknown[]) => mockRefreshAuthTokenNow(...args),
  sessionResumeRequest: (...args: unknown[]) => mockSessionResumeRequest(...args),
  getLastRefreshErrorCode: (...args: unknown[]) => mockGetLastRefreshErrorCode(...args),
  setAuthToken: jest.fn(),
  revokeSessionPending: jest.fn(),
}));

jest.mock("./pendingResumeOperation", () => ({
  readPendingResumeOperation: (...args: unknown[]) => mockReadPendingResume(...args),
}));

jest.mock("../observability/sessionJournal", () => ({
  appendSessionJournalEvent: jest.fn(),
}));

import {
  resolveAuthGuardDecisionState,
  resolveAuthGuardRedirect,
} from "./authGuardDecision";
import {
  __resetRecoveryInFlightForTests,
  attemptRestRecovery,
  classifyAuthErrorCode,
} from "./authRecoveryCoordinator";

const bootstrapAuth = {
  is_authenticated: true,
} as import("../contracts/auth").BootstrapResponse;

const bootstrapAnon = {
  is_authenticated: false,
} as import("../contracts/auth").BootstrapResponse;

describe("P0-4 authGuardDecision", () => {
  it("A — RECOVERING → AuthGuard redirect = null", () => {
    expect(
      resolveAuthGuardDecisionState({
        bootstrap: bootstrapAnon,
        mobileSessionStatus: "auth_recovering",
      })
    ).toBe("RECOVERING");
    expect(resolveAuthGuardRedirect(bootstrapAnon, "auth_recovering")).toBeNull();
  });

  it("DEGRADED_AUTHENTICATED → AuthGuard redirect = null", () => {
    expect(
      resolveAuthGuardDecisionState({
        bootstrap: bootstrapAnon,
        mobileSessionStatus: "authenticated_offline",
      })
    ).toBe("DEGRADED_AUTHENTICATED");
    expect(resolveAuthGuardRedirect(bootstrapAnon, "authenticated_offline")).toBeNull();
  });

  it("AUTHENTICATED → pas de redirect", () => {
    expect(
      resolveAuthGuardDecisionState({
        bootstrap: bootstrapAuth,
        mobileSessionStatus: "authenticated_online",
      })
    ).toBe("AUTHENTICATED");
    expect(resolveAuthGuardRedirect(bootstrapAuth, "authenticated_online")).toBeNull();
  });

  it("BOOTSTRAPPING (bootstrap null) → pas de redirect / ≠ logout", () => {
    expect(
      resolveAuthGuardDecisionState({
        bootstrap: null,
        mobileSessionStatus: "initializing",
      })
    ).toBe("BOOTSTRAPPING");
    expect(resolveAuthGuardRedirect(null, "initializing")).toBeNull();
    expect(resolveAuthGuardRedirect(null, "anonymous")).toBeNull();
  });

  it("TERMINAL_UNAUTHENTICATED → /(public)", () => {
    expect(
      resolveAuthGuardDecisionState({
        bootstrap: bootstrapAnon,
        mobileSessionStatus: "revoked",
      })
    ).toBe("TERMINAL_UNAUTHENTICATED");
    expect(resolveAuthGuardRedirect(bootstrapAnon, "revoked")).toBe("/(public)");
    expect(resolveAuthGuardRedirect(bootstrapAnon, "anonymous")).toBe("/(public)");
  });

  it("H — storage_locked (SecureStore unavailable) → RECOVERING, pas login", () => {
    expect(
      resolveAuthGuardDecisionState({
        bootstrap: null,
        mobileSessionStatus: "storage_locked",
      })
    ).toBe("RECOVERING");
    expect(resolveAuthGuardRedirect(null, "storage_locked")).toBeNull();
  });
});

describe("P0-5 classifyAuthErrorCode", () => {
  it("E — codes terminaux", () => {
    expect(classifyAuthErrorCode("session_revoked")).toBe("terminal");
    expect(classifyAuthErrorCode("refresh_replay_detected")).toBe("terminal");
    expect(classifyAuthErrorCode("account_disabled")).toBe("terminal");
  });

  it("B — codes non terminaux keep_local", () => {
    expect(classifyAuthErrorCode("store_unavailable")).toBe("keep_local");
    expect(classifyAuthErrorCode("refresh_store_unavailable")).toBe("keep_local");
    expect(classifyAuthErrorCode("service_unavailable")).toBe("keep_local");
    expect(classifyAuthErrorCode("ERR_NETWORK")).toBe("keep_local");
    expect(classifyAuthErrorCode("timeout")).toBe("keep_local");
    expect(classifyAuthErrorCode("temporarily_unavailable")).toBe("keep_local");
    expect(classifyAuthErrorCode("rate_limited")).toBe("keep_local");
  });
});

describe("P0-5 attemptRestRecovery", () => {
  beforeEach(() => {
    __resetRecoveryInFlightForTests();
    mockRefreshAuthTokenNow.mockReset();
    mockSessionResumeRequest.mockReset();
    mockGetLastRefreshErrorCode.mockReset();
    mockReadPendingResume.mockReset();
    mockReadPendingResume.mockResolvedValue(null);
  });

  it("F — multi-trigger → un seul refresh réseau", async () => {
    let resolveRefresh!: (v: boolean) => void;
    const refreshPromise = new Promise<boolean>((r) => {
      resolveRefresh = r;
    });
    mockRefreshAuthTokenNow.mockReturnValue(refreshPromise);

    const a = attemptRestRecovery("foreground");
    const b = attemptRestRecovery("401");
    const c = attemptRestRecovery("socket");
    const d = attemptRestRecovery("bootstrap");

    await Promise.resolve();
    await Promise.resolve();
    expect(mockRefreshAuthTokenNow).toHaveBeenCalledTimes(1);
    resolveRefresh(true);
    const results = await Promise.all([a, b, c, d]);
    expect(results).toEqual(["recovered", "recovered", "recovered", "recovered"]);
    expect(mockRefreshAuthTokenNow).toHaveBeenCalledTimes(1);
  });

  it("B — 503 store_unavailable → keep_local (pas terminal)", async () => {
    mockRefreshAuthTokenNow.mockResolvedValue(false);
    mockGetLastRefreshErrorCode.mockReturnValue("store_unavailable");
    const outcome = await attemptRestRecovery("warm_503");
    expect(outcome).toBe("keep_local");
    expect(mockSessionResumeRequest).not.toHaveBeenCalled();
  });

  it("A — refresh fail unknown puis session-resume OK → recovered", async () => {
    mockRefreshAuthTokenNow.mockResolvedValue(false);
    mockGetLastRefreshErrorCode.mockReturnValue("invalid_token");
    mockSessionResumeRequest.mockResolvedValue({
      ok: true,
      code: null,
      retryable: false,
    });
    const outcome = await attemptRestRecovery("bootstrap_unauthenticated");
    expect(outcome).toBe("recovered");
    expect(mockSessionResumeRequest).toHaveBeenCalledTimes(1);
  });

  it("E — session_revoked → terminal", async () => {
    mockRefreshAuthTokenNow.mockResolvedValue(false);
    mockGetLastRefreshErrorCode.mockReturnValue("session_revoked");
    const outcome = await attemptRestRecovery("revoke");
    expect(outcome).toBe("terminal");
    expect(mockSessionResumeRequest).not.toHaveBeenCalled();
  });

  it("C — offline ERR_NETWORK → keep_local", async () => {
    mockRefreshAuthTokenNow.mockResolvedValue(false);
    mockGetLastRefreshErrorCode.mockReturnValue("ERR_NETWORK");
    const outcome = await attemptRestRecovery("offline");
    expect(outcome).toBe("keep_local");
  });
});
