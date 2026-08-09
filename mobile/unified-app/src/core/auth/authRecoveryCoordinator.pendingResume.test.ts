/**
 * PendingResumeOperation doit être réconciliée avant le refresh normal.
 */
import { beforeEach, describe, expect, it, jest } from "@jest/globals";

const mockSecureMemory = new Map<string, string>();
const mockAsyncMemory = new Map<string, string>();

jest.mock("expo-secure-store", () => ({
  AFTER_FIRST_UNLOCK: 0,
  getItemAsync: jest.fn(async (key: string) => mockSecureMemory.get(key) ?? null),
  setItemAsync: jest.fn(async (key: string, value: string) => {
    mockSecureMemory.set(key, value);
  }),
  deleteItemAsync: jest.fn(async (key: string) => {
    mockSecureMemory.delete(key);
  }),
}));

jest.mock("@react-native-async-storage/async-storage", () => ({
  __esModule: true,
  default: {
    getItem: jest.fn(async (key: string) => mockAsyncMemory.get(key) ?? null),
    setItem: jest.fn(async (key: string, value: string) => {
      mockAsyncMemory.set(key, value);
    }),
    removeItem: jest.fn(async (key: string) => {
      mockAsyncMemory.delete(key);
    }),
    clear: jest.fn(async () => {
      mockAsyncMemory.clear();
    }),
  },
}));

const mockRefreshAuthTokenNow = jest.fn(async () => true);
const mockSessionResumeRequest = jest.fn(async () => ({
  ok: true,
  code: null as string | null,
  retryable: false,
}));

jest.mock("../api/client", () => ({
  refreshAuthTokenNow: (...args: unknown[]) => mockRefreshAuthTokenNow(...args),
  sessionResumeRequest: (...args: unknown[]) => mockSessionResumeRequest(...args),
  revokeSessionPending: jest.fn(async () => true),
  setAuthToken: jest.fn(),
  getLastRefreshErrorCode: jest.fn(() => null),
  logoutSession: jest.fn(async () => undefined),
}));

jest.mock("../observability/sessionJournal", () => ({
  appendSessionJournalEvent: jest.fn(),
}));

jest.mock("../network/networkState", () => ({
  getNetworkSnapshot: () => ({ connected: true }),
}));

import { attemptRestRecovery } from "./authRecoveryCoordinator";
import { ensurePendingResumeOperation } from "./pendingResumeOperation";

describe("attemptRestRecovery + PendingResumeOperation", () => {
  beforeEach(() => {
    mockSecureMemory.clear();
    mockAsyncMemory.clear();
    mockRefreshAuthTokenNow.mockClear();
    mockSessionResumeRequest.mockClear();
    mockRefreshAuthTokenNow.mockResolvedValue(true);
    mockSessionResumeRequest.mockResolvedValue({
      ok: true,
      code: null,
      retryable: false,
    });
  });

  it("réconcilie PendingResume avant refreshAuthTokenNow", async () => {
    await ensurePendingResumeOperation({
      sessionId: "sess-crash",
      sourceCredentialGeneration: 6,
    });

    const outcome = await attemptRestRecovery("cold_start");

    expect(outcome).toBe("recovered");
    expect(mockSessionResumeRequest).toHaveBeenCalledTimes(1);
    expect(mockRefreshAuthTokenNow).not.toHaveBeenCalled();
  });

  it("ne déclare pas recovered via refresh si pending resume échoue", async () => {
    await ensurePendingResumeOperation({
      sessionId: "sess-crash",
      sourceCredentialGeneration: null,
    });
    mockSessionResumeRequest.mockResolvedValue({
      ok: false,
      code: "idempotency_result_expired",
      retryable: false,
    });

    const outcome = await attemptRestRecovery("cold_start");

    expect(outcome).toBe("no_action");
    expect(mockRefreshAuthTokenNow).not.toHaveBeenCalled();
  });

  it("utilise le refresh normal seulement sans pending resume", async () => {
    const outcome = await attemptRestRecovery("foreground");
    expect(outcome).toBe("recovered");
    expect(mockRefreshAuthTokenNow).toHaveBeenCalledTimes(1);
    expect(mockSessionResumeRequest).not.toHaveBeenCalled();
  });
});
