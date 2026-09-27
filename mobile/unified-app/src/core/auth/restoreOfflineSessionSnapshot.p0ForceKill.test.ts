/**
 * P0 force-kill — invariant RESTORABLE_AUTH_SESSION + crash-safety purge.
 * T1–T7 automatisables + incident exact + purge mid-flight.
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

jest.mock("react-native", () => ({
  Platform: { OS: "android" },
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
  },
}));

jest.mock("../api/client", () => ({
  refreshAuthTokenNow: jest.fn(async () => false),
  revokeSessionPending: jest.fn(async () => true),
  sessionResumeRequest: jest.fn(async () => ({ ok: false, code: null, retryable: false })),
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

import {
  __resetSessionGenerationForTests,
  clearLocalAuthCredentialsLocked,
  createAndPersistInstallationId,
  deleteSessionEnvelope,
  invalidateRefreshToken,
  LEGACY_REFRESH_TOKEN_KEY,
  writeRecoveryCredential,
  writeRefreshToken,
  writeSessionEnvelope,
} from "./authCredentialStore";
import { __resetCredentialStoreLockForTests, withCredentialStoreLock } from "./sessionCredentialMutex";
import { restoreOfflineSessionSnapshot } from "./authRecoveryCoordinator";
import * as SecureStore from "expo-secure-store";

const ENVELOPE_BASE = {
  schema_version: 1 as const,
  session_id: "sess-p0",
  device_installation_id: "atmr-install-1",
  user_public_id: "u1",
  driver_id: null as number | null,
  role: "company",
  active_context_id: "company:1",
  refresh_generation: 1,
  last_authenticated_at: new Date().toISOString(),
  cached_active_context: {
    context_id: "company:1",
    context_type: "company" as const,
    label: "Co",
    permissions: [] as string[],
  },
  cached_bootstrap: null,
};

async function seedRestorableBundle(opts?: { withRefresh?: boolean; withLegacy?: boolean }) {
  const withRefresh = opts?.withRefresh !== false;
  mockSecureMemory.set("atmr.auth.installation_id", "atmr-install-1");
  await writeRecoveryCredential("recovery-tok");
  if (withRefresh) {
    await writeRefreshToken("refresh-tok");
  }
  if (opts?.withLegacy) {
    mockSecureMemory.set(LEGACY_REFRESH_TOKEN_KEY, "legacy-refresh-tok");
  }
  await writeSessionEnvelope(ENVELOPE_BASE);
}

describe("P0 force-kill restoreOfflineSessionSnapshot", () => {
  beforeEach(() => {
    mockSecureMemory.clear();
    mockAsyncMemory.clear();
    __resetSessionGenerationForTests();
    __resetCredentialStoreLockForTests();
  });

  it("T1 — normal cold start → restored", async () => {
    await seedRestorableBundle({ withRefresh: true });
    const snap = await restoreOfflineSessionSnapshot();
    expect(snap.kind).toBe("restored");
    if (snap.kind === "restored") {
      expect(snap.activeContext?.context_id).toBe("company:1");
    }
  });

  it("T2 / incident — refresh MISSING + envelope/recovery/install → incoherent_local", async () => {
    await seedRestorableBundle({ withRefresh: false });
    const snap = await restoreOfflineSessionSnapshot();
    expect(snap.kind).toBe("incoherent_local");
    expect(snap.kind).not.toBe("restored");
  });

  it("T3 — TEMPORARILY_UNAVAILABLE → storage_locked (pas terminal)", async () => {
    await seedRestorableBundle({ withRefresh: true });
    const getSpy = SecureStore.getItemAsync as jest.MockedFunction<
      typeof SecureStore.getItemAsync
    >;
    getSpy.mockImplementationOnce(async () => {
      throw new Error("keystore_busy");
    });
    const snap = await restoreOfflineSessionSnapshot();
    expect(snap.kind).toBe("storage_locked");
  });

  it("T4 — PERMANENTLY_INVALIDATED → revoked", async () => {
    await seedRestorableBundle({ withRefresh: true });
    await invalidateRefreshToken("session_revoked");
    const snap = await restoreOfflineSessionSnapshot();
    expect(snap.kind).toBe("revoked");
  });

  it("T5 — legacy orphan + strict MISSING + installation FOUND → incoherent_local (no resurrect)", async () => {
    await seedRestorableBundle({ withRefresh: false, withLegacy: true });
    expect(mockSecureMemory.get(LEGACY_REFRESH_TOKEN_KEY)).toBe("legacy-refresh-tok");
    const snap = await restoreOfflineSessionSnapshot();
    expect(snap.kind).toBe("incoherent_local");
    // Legacy toujours présent — non lu / non migré
    expect(mockSecureMemory.get(LEGACY_REFRESH_TOKEN_KEY)).toBe("legacy-refresh-tok");
  });

  it("T6 — legacy migration N/A : pas de resurrection même si legacy seul", async () => {
    mockSecureMemory.set("atmr.auth.installation_id", "atmr-install-1");
    mockSecureMemory.set(LEGACY_REFRESH_TOKEN_KEY, "only-legacy");
    await writeRecoveryCredential("recovery-tok");
    await writeSessionEnvelope(ENVELOPE_BASE);
    const snap = await restoreOfflineSessionSnapshot();
    expect(snap.kind).toBe("incoherent_local");
  });

  it("purge crash-safety — envelope deleted puis crash simulé → MUST NOT restored", async () => {
    await seedRestorableBundle({ withRefresh: true });
    // Simule kill après étape 1 de clearLocalAuthCredentialsLocked
    await deleteSessionEnvelope();
    const snap = await restoreOfflineSessionSnapshot();
    expect(snap.kind).not.toBe("restored");
    expect(snap.kind).toBe("anonymous");
  });

  it("purge complète crash-safe order + legacy deleted + installation kept", async () => {
    await seedRestorableBundle({ withRefresh: true, withLegacy: true });
    await withCredentialStoreLock(async () => {
      await clearLocalAuthCredentialsLocked();
    });
    expect(mockSecureMemory.get("atmr.auth.session_envelope")).toBeUndefined();
    expect(mockSecureMemory.get("atmr.auth.refresh_token")).toBeUndefined();
    expect(mockSecureMemory.get("atmr.auth.recovery_credential")).toBeUndefined();
    expect(mockSecureMemory.get(LEGACY_REFRESH_TOKEN_KEY)).toBeUndefined();
    expect(mockSecureMemory.get("atmr.auth.installation_id")).toBe("atmr-install-1");
    const snap = await restoreOfflineSessionSnapshot();
    expect(snap.kind).toBe("anonymous");
  });

  it("createAndPersistInstallationId ne régénère pas si présent", async () => {
    mockSecureMemory.set("atmr.auth.installation_id", "atmr-existing");
    const id = await createAndPersistInstallationId();
    expect(id.status).toBe("found");
    if (id.status === "found") {
      expect(id.value).toBe("atmr-existing");
    }
  });
});
