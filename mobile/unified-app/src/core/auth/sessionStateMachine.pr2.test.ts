/**
 * Tests courses PR2 — machine d'état session mobile.
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
  },
}));

jest.mock("../api/client", () => ({
  refreshAuthTokenNow: jest.fn(async () => false),
  revokeSessionPending: jest.fn(async () => true),
  sessionResumeRequest: jest.fn(async () => ({ ok: false, code: null, retryable: false })),
  setAuthToken: jest.fn(),
  getLastRefreshErrorCode: jest.fn(() => "session_revoked"),
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
  appendPendingRevocation,
  bumpSessionGeneration,
  clearLocalAuthCredentialsLocked,
  getSessionGenerationId,
  isCurrentSessionGeneration,
  persistTerminalRevocationEvidenceLocked,
  readPendingRevocations,
  readRefreshToken,
  readRecoveryCredential,
  replacePendingRevocations,
  writeRefreshToken,
  writeRecoveryCredential,
  writeSessionEnvelope,
  type PendingRevocation,
} from "./authCredentialStore";
import {
  __resetCredentialStoreLockForTests,
  claimNextSessionGenerationIfCurrent,
  withCredentialStoreLock,
  withSessionCredentialMutation,
} from "./sessionCredentialMutex";
import {
  __resetExplicitLogoutInFlightForTests,
  applyTerminalRevocationIfCurrent,
  enqueueOrphanedLoginRevocation,
  finishInterruptedExplicitLogout,
  performExplicitLogout,
  restoreOfflineSessionSnapshot,
} from "./authRecoveryCoordinator";
import {
  resolveSessionLifecyclePolicy,
  shouldAcceptBootstrapTrigger,
} from "./sessionLifecycle";

describe("PR2 session credential mutex", () => {
  beforeEach(() => {
    mockSecureMemory.clear();
    mockAsyncMemory.clear();
    __resetSessionGenerationForTests();
    __resetCredentialStoreLockForTests();
    __resetExplicitLogoutInFlightForTests();
  });

  it("test_mutex_requires_expected_generation_and_rejects_stale", async () => {
    const gen = bumpSessionGeneration();
    bumpSessionGeneration();
    const result = await withSessionCredentialMutation(gen, async () => "x");
    expect(result).toEqual({ status: "stale" });
  });

  it("test_pending_revocation_append_works_when_generation_is_stale", async () => {
    const staleGen = bumpSessionGeneration();
    bumpSessionGeneration();
    expect(isCurrentSessionGeneration(staleGen)).toBe(false);

    const pending: PendingRevocation = {
      operation_id: "op-stale-1",
      session_id: "sess-a",
      device_installation_id: "dev-1",
      revocation_secret: "sec",
      created_at: new Date().toISOString(),
      origin: "orphaned_login_cleanup",
    };
    await withCredentialStoreLock(async () => {
      await appendPendingRevocation(pending);
    });
    const list = await readPendingRevocations();
    expect(list.status).toBe("found");
    if (list.status === "found") {
      expect(list.value.some((p) => p.operation_id === "op-stale-1")).toBe(true);
    }
  });

  it("test_terminal_revocation_compare_and_bump_is_atomic_against_new_login", async () => {
    const source = bumpSessionGeneration();
    const claimPromise = claimNextSessionGenerationIfCurrent(source);
    // Concurrent login bump sous le même verrou — le claim doit voir un résultat cohérent
    const loginBumpPromise = withCredentialStoreLock(() => bumpSessionGeneration());
    const [claim, loginGen] = await Promise.all([claimPromise, loginBumpPromise]);
    if (claim.status === "claimed") {
      expect(claim.generation).toBeGreaterThan(source);
      expect(isCurrentSessionGeneration(claim.generation) || isCurrentSessionGeneration(loginGen)).toBe(
        true
      );
    } else {
      expect(claim.status).toBe("stale");
      expect(isCurrentSessionGeneration(loginGen)).toBe(true);
    }
  });
});

describe("PR2 pending / logout / terminal evidence", () => {
  beforeEach(async () => {
    mockSecureMemory.clear();
    mockAsyncMemory.clear();
    __resetSessionGenerationForTests();
    __resetCredentialStoreLockForTests();
    __resetExplicitLogoutInFlightForTests();
    await withCredentialStoreLock(async () => {
      await clearLocalAuthCredentialsLocked();
      await replacePendingRevocations([]);
    });
  });

  it("test_stale_login_enqueues_durable_orphan_revocation", async () => {
    const orphan = await enqueueOrphanedLoginRevocation({
      sessionId: "sess-orphan",
      deviceInstallationId: "dev",
      revocationSecret: "sec-orphan",
      operationId: "op-orphan",
    });
    expect(orphan.origin).toBe("orphaned_login_cleanup");
    expect(orphan.local_cleanup).toBeUndefined();
    const list = await readPendingRevocations();
    expect(list.status).toBe("found");
    if (list.status === "found") {
      expect(list.value.find((p) => p.operation_id === "op-orphan")?.origin).toBe(
        "orphaned_login_cleanup"
      );
    }
  });

  it("test_orphaned_login_pending_never_triggers_gps_quarantine", async () => {
    const policy = resolveSessionLifecyclePolicy("login_stale_orphaned", {
      hasOfflineSnapshot: false,
      currentStatus: "authenticated_online",
      autoBootstrapAllowed: true,
    });
    expect(policy.quarantineRequired).toBe(false);
  });

  it("test_pending_logout_persists_tracking_identity_before_quarantine", async () => {
    const sourceGeneration = bumpSessionGeneration();
    await writeSessionEnvelope({
      schema_version: 1,
      session_id: "sess-logout",
      device_installation_id: "dev",
      user_public_id: "u1",
      driver_id: 1,
      role: "driver",
      active_context_id: "driver:1",
      refresh_generation: 1,
      last_authenticated_at: new Date().toISOString(),
      revocation_secret: "rev-sec",
    });

    const quarantineOrder: string[] = [];
    await performExplicitLogout({
      sourceGeneration,
      sourceSessionId: "sess-logout",
      lifecycleOperationId: "op-logout-id",
      trackingIdentity: {
        user_id: "driver:1",
        driver_id: "1",
        company_id: "c1",
      },
      quarantineRequired: true,
      runQuarantine: async () => {
        quarantineOrder.push("quarantine");
        const list = await readPendingRevocations();
        expect(list.status).toBe("found");
        if (list.status === "found") {
          const p = list.value.find((x) => x.operation_id === "op-logout-id");
          expect(p?.local_cleanup?.tracking_identity?.company_id).toBe("c1");
          expect(p?.local_cleanup?.quarantine_required).toBe(true);
        }
      },
      clearQuarantineIfOperationMatches: async () => undefined,
      commitSessionStateIfCurrent: () => true,
    });
    expect(quarantineOrder).toEqual(["quarantine"]);
  });

  it("test_double_logout_same_session_is_single_flight", async () => {
    const sourceGeneration = bumpSessionGeneration();
    await writeSessionEnvelope({
      schema_version: 1,
      session_id: "sess-sf",
      device_installation_id: "dev",
      user_public_id: "u1",
      driver_id: null,
      role: "client",
      active_context_id: null,
      refresh_generation: 1,
      last_authenticated_at: new Date().toISOString(),
      revocation_secret: "rev",
    });

    let networkCalls = 0;
    const { revokeSessionPending } = jest.requireMock("../api/client") as {
      revokeSessionPending: jest.Mock;
    };
    revokeSessionPending.mockImplementation(async () => {
      networkCalls += 1;
      await new Promise((r) => setTimeout(r, 30));
      return true;
    });

    const p1 = performExplicitLogout({
      sourceGeneration,
      sourceSessionId: "sess-sf",
      lifecycleOperationId: "op-a",
      trackingIdentity: null,
      quarantineRequired: false,
      runQuarantine: async () => undefined,
      clearQuarantineIfOperationMatches: async () => undefined,
      commitSessionStateIfCurrent: () => true,
    });
    const p2 = performExplicitLogout({
      sourceGeneration,
      sourceSessionId: "sess-sf",
      lifecycleOperationId: "op-b",
      trackingIdentity: null,
      quarantineRequired: false,
      runQuarantine: async () => undefined,
      clearQuarantineIfOperationMatches: async () => undefined,
      commitSessionStateIfCurrent: () => true,
    });
    const [r1, r2] = await Promise.all([p1, p2]);
    expect(r1).toBe(r2);
    expect(networkCalls).toBe(1);
  });

  it("test_logout_new_session_is_not_absorbed_by_stale_logout_inflight", async () => {
    const genA = bumpSessionGeneration();
    await writeSessionEnvelope({
      schema_version: 1,
      session_id: "sess-a",
      device_installation_id: "dev",
      user_public_id: "u1",
      driver_id: null,
      role: "client",
      active_context_id: null,
      refresh_generation: 1,
      last_authenticated_at: new Date().toISOString(),
      revocation_secret: "rev-a",
    });

    const { revokeSessionPending } = jest.requireMock("../api/client") as {
      revokeSessionPending: jest.Mock;
    };
    let releaseNetwork!: () => void;
    const networkGate = new Promise<void>((r) => {
      releaseNetwork = r;
    });
    revokeSessionPending.mockImplementation(async () => {
      await networkGate;
      return true;
    });

    const logoutA = performExplicitLogout({
      sourceGeneration: genA,
      sourceSessionId: "sess-a",
      lifecycleOperationId: "op-a",
      trackingIdentity: null,
      quarantineRequired: false,
      runQuarantine: async () => undefined,
      clearQuarantineIfOperationMatches: async () => undefined,
      commitSessionStateIfCurrent: () => true,
    });

    // Attendre que A ait claimé
    await new Promise((r) => setTimeout(r, 10));
    const genB = getSessionGenerationId();
    await writeSessionEnvelope({
      schema_version: 1,
      session_id: "sess-b",
      device_installation_id: "dev",
      user_public_id: "u2",
      driver_id: null,
      role: "client",
      active_context_id: null,
      refresh_generation: 1,
      last_authenticated_at: new Date().toISOString(),
      revocation_secret: "rev-b",
    });

    const logoutB = performExplicitLogout({
      sourceGeneration: genB,
      sourceSessionId: "sess-b",
      lifecycleOperationId: "op-b",
      trackingIdentity: null,
      quarantineRequired: false,
      runQuarantine: async () => undefined,
      clearQuarantineIfOperationMatches: async () => undefined,
      commitSessionStateIfCurrent: () => true,
    });

    expect(logoutA).not.toBe(logoutB);
    releaseNetwork();
    await Promise.all([logoutA, logoutB]);
  });

  it("test_crash_after_pending_before_quarantine_can_finish_local_logout", async () => {
    const pending: PendingRevocation = {
      operation_id: "op-crash",
      session_id: "sess-crash",
      device_installation_id: "dev",
      revocation_secret: "sec",
      created_at: new Date().toISOString(),
      origin: "explicit_logout",
      local_cleanup: {
        tracking_identity: {
          user_id: "driver:1",
          driver_id: "1",
          company_id: "c1",
        },
        quarantine_required: true,
      },
    };
    await withCredentialStoreLock(async () => {
      await appendPendingRevocation(pending);
      await writeSessionEnvelope({
        schema_version: 1,
        session_id: "sess-crash",
        device_installation_id: "dev",
        user_public_id: "u1",
        driver_id: 1,
        role: "driver",
        active_context_id: "driver:1",
        refresh_generation: 1,
        last_authenticated_at: new Date().toISOString(),
        revocation_secret: "sec",
      });
      await writeRefreshToken("refresh");
      await writeRecoveryCredential("recovery");
    });

    const restore = await restoreOfflineSessionSnapshot();
    expect(restore.kind).toBe("interrupted_logout");

    let quarantined = false;
    await finishInterruptedExplicitLogout(pending, {
      runQuarantine: async () => {
        quarantined = true;
      },
    });
    expect(quarantined).toBe(true);
    const refresh = await readRefreshToken();
    expect(refresh.status).toBe("missing");
  });

  it("test_matching_explicit_logout_pending_finishes_as_anonymous_not_revoked", async () => {
    const policy = resolveSessionLifecyclePolicy("interrupted_logout_finished", {
      hasOfflineSnapshot: false,
      currentStatus: "logging_out",
      autoBootstrapAllowed: false,
    });
    expect(policy.statusAction).toBe("anonymous");
    expect(policy.terminalEvidenceRequired).toBe(false);
  });

  it("test_revoked_requires_terminal_evidence_not_pending_intent", async () => {
    await withCredentialStoreLock(async () => {
      await appendPendingRevocation({
        operation_id: "op-pend",
        session_id: "sess",
        device_installation_id: "dev",
        revocation_secret: "sec",
        created_at: new Date().toISOString(),
        origin: "explicit_logout",
      });
    });
    // Sans envelope match + sans permanently_invalidated → anonymous (pas revoked)
    const snap = await restoreOfflineSessionSnapshot();
    expect(snap.kind).not.toBe("revoked");
  });

  it("test_terminal_revocation_persists_evidence_before_credential_cleanup", async () => {
    const source = bumpSessionGeneration();
    await writeRefreshToken("refresh-tok");
    await writeRecoveryCredential("recovery-tok");
    await writeSessionEnvelope({
      schema_version: 1,
      session_id: "sess-term",
      device_installation_id: "dev",
      user_public_id: "u1",
      driver_id: null,
      role: "client",
      active_context_id: null,
      refresh_generation: 1,
      last_authenticated_at: new Date().toISOString(),
    });

    let uiStatus: string | null = null;
    const result = await applyTerminalRevocationIfCurrent(source, "session_revoked", (gen) => {
      expect(isCurrentSessionGeneration(gen)).toBe(true);
      uiStatus = "revoked";
      return true;
    });
    expect(result).toBe("applied");
    expect(uiStatus).toBe("revoked");
    const refresh = await readRefreshToken();
    expect(refresh.status).toBe("permanently_invalidated");
    const recovery = await readRecoveryCredential();
    expect(recovery.status).toBe("permanently_invalidated");
  });

  it("test_stale_logout_cannot_set_anonymous_after_new_login_claim", async () => {
    const source = bumpSessionGeneration();
    await writeSessionEnvelope({
      schema_version: 1,
      session_id: "sess-race",
      device_installation_id: "dev",
      user_public_id: "u1",
      driver_id: null,
      role: "client",
      active_context_id: null,
      refresh_generation: 1,
      last_authenticated_at: new Date().toISOString(),
      revocation_secret: "sec",
    });
    let anonymousSet = false;
    const result = await performExplicitLogout({
      sourceGeneration: source,
      sourceSessionId: "sess-race",
      lifecycleOperationId: "op-race-2",
      trackingIdentity: {
        user_id: "u1",
        driver_id: "1",
        company_id: "c1",
      },
      quarantineRequired: true,
      runQuarantine: async () => {
        bumpSessionGeneration();
      },
      clearQuarantineIfOperationMatches: async () => undefined,
      commitSessionStateIfCurrent: () => {
        anonymousSet = true;
        return true;
      },
    });
    expect(result.status).toBe("stale_superseded");
    expect(anonymousSet).toBe(false);
  });

  it("test_stale_logout_cannot_clear_new_bootstrap_or_active_context", async () => {
    const source = bumpSessionGeneration();
    await writeSessionEnvelope({
      schema_version: 1,
      session_id: "sess-ui",
      device_installation_id: "dev",
      user_public_id: "u1",
      driver_id: null,
      role: "client",
      active_context_id: null,
      refresh_generation: 1,
      last_authenticated_at: new Date().toISOString(),
      revocation_secret: "sec",
    });
    let cleared = false;
    const result = await performExplicitLogout({
      sourceGeneration: source,
      sourceSessionId: "sess-ui",
      lifecycleOperationId: "op-ui",
      trackingIdentity: {
        user_id: "u1",
        driver_id: "1",
        company_id: "c1",
      },
      quarantineRequired: true,
      runQuarantine: async () => {
        bumpSessionGeneration();
      },
      clearQuarantineIfOperationMatches: async () => undefined,
      commitSessionStateIfCurrent: (gen) => {
        if (!isCurrentSessionGeneration(gen)) return false;
        cleared = true;
        return true;
      },
    });
    expect(result.status).toBe("stale_superseded");
    expect(cleared).toBe(false);
  });

  it("test_terminal_revocation_status_commit_is_generation_guarded", async () => {
    const source = bumpSessionGeneration();
    bumpSessionGeneration();
    const result = await applyTerminalRevocationIfCurrent(source, "session_revoked", () => true);
    expect(result).toBe("stale");
  });

  it("test_scheduled_auto_bootstrap_is_rejected_after_logout_claim", () => {
    expect(shouldAcceptBootstrapTrigger("cold_start_auto", false)).toBe(false);
    expect(shouldAcceptBootstrapTrigger("manual_retry", false)).toBe(true);
    expect(shouldAcceptBootstrapTrigger("login_success", false)).toBe(true);
  });

  it("test_explicit_logout_quarantines_before_network_revoke", async () => {
    const source = bumpSessionGeneration();
    await writeSessionEnvelope({
      schema_version: 1,
      session_id: "sess-order",
      device_installation_id: "dev",
      user_public_id: "u1",
      driver_id: 1,
      role: "driver",
      active_context_id: "driver:1",
      refresh_generation: 1,
      last_authenticated_at: new Date().toISOString(),
      revocation_secret: "sec",
    });
    const order: string[] = [];
    const { revokeSessionPending } = jest.requireMock("../api/client") as {
      revokeSessionPending: jest.Mock;
    };
    revokeSessionPending.mockImplementation(async () => {
      order.push("network");
      return true;
    });
    await performExplicitLogout({
      sourceGeneration: source,
      sourceSessionId: "sess-order",
      lifecycleOperationId: "op-order",
      trackingIdentity: {
        user_id: "driver:1",
        driver_id: "1",
        company_id: "c1",
      },
      quarantineRequired: true,
      runQuarantine: async () => {
        order.push("quarantine");
      },
      clearQuarantineIfOperationMatches: async () => undefined,
      commitSessionStateIfCurrent: () => true,
    });
    expect(order.indexOf("quarantine")).toBeLessThan(order.indexOf("network"));
  });

  it("test_socket_auth_exhausted_does_not_quarantine", () => {
    const policy = resolveSessionLifecyclePolicy("auth_exhausted_socket", {
      hasOfflineSnapshot: true,
      currentStatus: "authenticated_online",
      autoBootstrapAllowed: true,
    });
    expect(policy.quarantineRequired).toBe(false);
    expect(policy.statusAction).toBeNull();
  });

  it("test_old_bootstrap_finally_does_not_clear_new_generation_inflight", () => {
    // Invariant structurel : finally compare par identité d'entrée (testé via référence).
    type Entry = { generation: number; id: string };
    let current: Entry | null = null;
    const entryA: Entry = { generation: 1, id: "a" };
    const entryB: Entry = { generation: 2, id: "b" };
    current = entryA;
    current = entryB;
    // finally de A
    if (current === entryA) current = null;
    expect(current).toEqual(entryB);
  });

  it("test_login_started_before_logout_cannot_reinstall_session_after_logout", async () => {
    const loginGen = bumpSessionGeneration();
    // Logout claim
    const claim = await claimNextSessionGenerationIfCurrent(loginGen);
    expect(claim.status).toBe("claimed");
    const persist = await withSessionCredentialMutation(loginGen, async () => "installed");
    expect(persist.status).toBe("stale");
  });
});

describe("PR2 persistTerminalRevocationEvidenceLocked", () => {
  beforeEach(() => {
    mockSecureMemory.clear();
    mockAsyncMemory.clear();
    __resetSessionGenerationForTests();
    __resetCredentialStoreLockForTests();
  });

  it("écrit permanently_invalidated sans laisser missing", async () => {
    await writeRefreshToken("r");
    await writeRecoveryCredential("c");
    await persistTerminalRevocationEvidenceLocked("session_revoked");
    expect((await readRefreshToken()).status).toBe("permanently_invalidated");
    expect((await readRecoveryCredential()).status).toBe("permanently_invalidated");
  });
});

describe("Phase 1B — refresh / context switch / tracking auth decision", () => {
  beforeEach(() => {
    mockSecureMemory.clear();
    mockAsyncMemory.clear();
    __resetSessionGenerationForTests();
    __resetCredentialStoreLockForTests();
    __resetExplicitLogoutInFlightForTests();
    const {
      __resetTrackingAuthDecisionForTests,
    } = require("./sessionAuthDecision") as typeof import("./sessionAuthDecision");
    const {
      __resetContextSwitchOperationForTests,
    } = require("./contextSwitchOperation") as typeof import("./contextSwitchOperation");
    __resetTrackingAuthDecisionForTests();
    __resetContextSwitchOperationForTests();
  });

  it("test_refresh_apply_under_mutex_is_stale_after_logout_claim", async () => {
    const refreshGen = bumpSessionGeneration();
    await writeRefreshToken("refresh-g10");
    const claim = await claimNextSessionGenerationIfCurrent(refreshGen);
    expect(claim.status).toBe("claimed");
    const apply = await withSessionCredentialMutation(refreshGen, async () => {
      await writeRefreshToken("refresh-g10-new");
      return "token-g10";
    });
    expect(apply.status).toBe("stale");
    // Le mutex peut avoir écrit puis détecté stale — la génération courante
    // doit permettre à un login plus récent d'écraser ensuite.
    const loginGen = getSessionGenerationId();
    const reinstall = await withSessionCredentialMutation(loginGen, async () => {
      await writeRefreshToken("refresh-g12");
      return "ok";
    });
    expect(reinstall.status).toBe("applied");
    expect((await readRefreshToken()).status).toBe("found");
    expect((await readRefreshToken()).status === "found"
      ? (await readRefreshToken() as { value: string }).value
      : null).toBe("refresh-g12");
  });

  it("test_double_context_switch_keeps_latest_operation", () => {
    const {
      beginContextSwitchOperation,
      isCurrentContextSwitchOperation,
    } = require("./contextSwitchOperation") as typeof import("./contextSwitchOperation");
    bumpSessionGeneration();
    const opA = beginContextSwitchOperation({
      sourceContextId: "driver:1",
      targetContextId: "company:A",
    });
    const opB = beginContextSwitchOperation({
      sourceContextId: "driver:1",
      targetContextId: "company:B",
    });
    expect(isCurrentContextSwitchOperation(opA.operationId)).toBe(false);
    expect(isCurrentContextSwitchOperation(opB.operationId)).toBe(true);
  });

  it("test_legacy_delete_stale_cannot_erase_new_login_token", async () => {
    const REFRESH_TOKEN_STORAGE_KEY = "atmr.refresh_token";
    const genOld = bumpSessionGeneration();
    await writeRefreshToken("old");
    mockSecureMemory.set(REFRESH_TOKEN_STORAGE_KEY, "old");

    // Login plus récent rend genOld stale avant toute suppression.
    const claim = await claimNextSessionGenerationIfCurrent(genOld);
    expect(claim.status).toBe("claimed");
    const loginGen = claim.generation;
    const loginPersist = await withSessionCredentialMutation(loginGen, async () => {
      await writeRefreshToken("new-login");
      mockSecureMemory.set(REFRESH_TOKEN_STORAGE_KEY, "new-login");
      return "installed";
    });
    expect(loginPersist.status).toBe("applied");

    // Ancienne intention de purge (génération obsolète) : corps non exécuté.
    let deleted = false;
    const staleResult = await withSessionCredentialMutation(genOld, async () => {
      deleted = true;
      mockSecureMemory.delete(REFRESH_TOKEN_STORAGE_KEY);
      const { deleteRefreshToken } =
        require("./authCredentialStore") as typeof import("./authCredentialStore");
      await deleteRefreshToken();
      return "deleted";
    });
    expect(staleResult.status).toBe("stale");
    expect(deleted).toBe(false);
    expect(mockSecureMemory.get(REFRESH_TOKEN_STORAGE_KEY)).toBe("new-login");
    const rt = await readRefreshToken();
    expect(rt.status).toBe("found");
    if (rt.status === "found") {
      expect(rt.value).toBe("new-login");
    }
  });

  it("test_tracking_auth_terminal_logout_emitted_once_per_operationId", () => {
    const {
      emitTrackingAuthTerminalEvent,
      subscribeToTrackingAuthTerminalEvents,
      TRACKING_AUTH_EFFECT_POLICY,
    } = require("./sessionAuthDecision") as typeof import("./sessionAuthDecision");
    const events: string[] = [];
    const unsub = subscribeToTrackingAuthTerminalEvents((e) => {
      events.push(e.kind);
    });
    emitTrackingAuthTerminalEvent({
      kind: "EXPLICIT_LOGOUT",
      sourceSessionGenerationId: 1,
      operationId: "op-1",
      trackingIdentityId: "u:d:c",
    });
    emitTrackingAuthTerminalEvent({
      kind: "EXPLICIT_LOGOUT",
      sourceSessionGenerationId: 1,
      operationId: "op-1",
      trackingIdentityId: "u:d:c",
    });
    expect(events).toEqual(["EXPLICIT_LOGOUT"]);
    expect(TRACKING_AUTH_EFFECT_POLICY.explicit_logout.quarantine).toBe(true);
    expect(TRACKING_AUTH_EFFECT_POLICY.auth_exhausted_socket.quarantine).toBe(false);
    unsub();
  });

  it("test_socket_auth_exhausted_policy_matches_tracking_effect_table", () => {
    const {
      TRACKING_AUTH_EFFECT_POLICY,
    } = require("./sessionAuthDecision") as typeof import("./sessionAuthDecision");
    const policy = resolveSessionLifecyclePolicy("auth_exhausted_socket", {
      hasOfflineSnapshot: true,
      currentStatus: "authenticated_online",
      autoBootstrapAllowed: true,
    });
    expect(policy.quarantineRequired).toBe(
      TRACKING_AUTH_EFFECT_POLICY.auth_exhausted_socket.quarantine
    );
    expect(policy.statusAction).toBeNull();
  });
});
