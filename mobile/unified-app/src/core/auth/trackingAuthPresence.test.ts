/**
 * P0-B — Tests presence persistée + hydrate headless + anti cross-driver.
 */
import { afterEach, beforeEach, describe, expect, it, jest } from "@jest/globals";

const mockReadRefresh = jest.fn();
const mockReadEnvelope = jest.fn();
const mockGetSessionGenerationId = jest.fn(() => 10);
const mockReadLease = jest.fn();

jest.mock("@react-native-async-storage/async-storage", () => {
  const store = new Map<string, string>();
  return {
    getItem: jest.fn(async (key: string) => store.get(key) ?? null),
    setItem: jest.fn(async (key: string, value: string) => {
      store.set(key, value);
    }),
    removeItem: jest.fn(async (key: string) => {
      store.delete(key);
    }),
    __store: store,
  };
});

jest.mock("./authCredentialStore", () => ({
  getSessionGenerationId: () => mockGetSessionGenerationId(),
  readRefreshToken: (...args: unknown[]) => mockReadRefresh(...args),
  readSessionEnvelope: (...args: unknown[]) => mockReadEnvelope(...args),
}));

jest.mock("../../features/driver/services/trackingContextLease", () => ({
  readTrackingContextLease: (...args: unknown[]) => mockReadLease(...args),
}));

import {
  __resetTrackingAuthDecisionForTests,
  getTrackingAuthAvailability,
  setTrackingAuthTemporarilyUnavailable,
} from "./sessionAuthDecision";
import {
  __resetTrackingAuthPresenceForTests,
  clearTrackingAuthSession,
  ensureTrackingAuthAvailabilityForHeadless,
  hydrateTrackingAuthFromPersistedState,
  publishTrackingAuthSessionAvailable,
  readTrackingAuthPresence,
  reassertTrackingAuthSessionAfterRefresh,
} from "./trackingAuthPresence";
import { validateNativeOwnerForHeadless } from "../../features/driver/services/trackingRuntimeRegistry";

function leaseFor(driverId: number, identity: string, sessionGenerationId = 10) {
  return {
    state: "driver_active" as const,
    contextId: `driver:${driverId}` as const,
    driverId,
    sessionGenerationId,
    trackingGenerationId: `gen-${driverId}`,
    trackingIdentityId: identity,
    missionId: 26,
    missionContextVersion: 1,
    updatedAt: Date.now(),
  };
}

describe("trackingAuthPresence (P0-B)", () => {
  beforeEach(async () => {
    await __resetTrackingAuthPresenceForTests();
    __resetTrackingAuthDecisionForTests();
    mockGetSessionGenerationId.mockReturnValue(10);
    mockReadRefresh.mockResolvedValue({ status: "found", value: "refresh-token" });
    mockReadEnvelope.mockResolvedValue({
      status: "found",
      value: {
        schema_version: 1,
        session_id: "s1",
        device_installation_id: "d1",
        user_public_id: "u1",
        driver_id: 19,
        role: "driver",
        active_context_id: "driver:19",
        refresh_generation: 1,
        last_authenticated_at: new Date().toISOString(),
      },
    });
    mockReadLease.mockResolvedValue(leaseFor(19, "driver:19:company:1"));
  });

  afterEach(async () => {
    await __resetTrackingAuthPresenceForTests();
  });

  it("1. login → SESSION_AVAILABLE + presence écrite", async () => {
    await publishTrackingAuthSessionAvailable({
      driverId: 19,
      trackingIdentityId: "driver:19:company:1",
      sessionGenerationId: 10,
    });
    const auth = getTrackingAuthAvailability();
    expect(auth.kind).toBe("SESSION_AVAILABLE");
    if (auth.kind === "SESSION_AVAILABLE") {
      expect(auth.driverId).toBe(19);
      expect(auth.trackingIdentityId).toBe("driver:19:company:1");
    }
    const presence = await readTrackingAuthPresence();
    expect(presence?.driverId).toBe(19);
    expect(presence?.logoutTombstoneAt).toBeNull();
  });

  it("2. cold start (mémoire détruite) + presence → hydrate SESSION_AVAILABLE", async () => {
    await publishTrackingAuthSessionAvailable({
      driverId: 19,
      trackingIdentityId: "driver:19:company:1",
    });
    __resetTrackingAuthDecisionForTests();
    expect(getTrackingAuthAvailability().kind).toBe("TRACKING_IDENTITY_UNAVAILABLE");

    const hydrated = await hydrateTrackingAuthFromPersistedState();
    expect(hydrated.kind).toBe("SESSION_AVAILABLE");
    expect(getTrackingAuthAvailability().kind).toBe("SESSION_AVAILABLE");
  });

  it("3. refresh en cours → AUTH_TEMPORARILY_UNAVAILABLE", async () => {
    await publishTrackingAuthSessionAvailable({
      driverId: 19,
      trackingIdentityId: "driver:19:company:1",
    });
    setTrackingAuthTemporarilyUnavailable("refreshing");
    expect(getTrackingAuthAvailability().kind).toBe("AUTH_TEMPORARILY_UNAVAILABLE");
  });

  it("4. refresh terminé → retour SESSION_AVAILABLE (pas UNAVAILABLE)", async () => {
    await publishTrackingAuthSessionAvailable({
      driverId: 19,
      trackingIdentityId: "driver:19:company:1",
    });
    setTrackingAuthTemporarilyUnavailable("refreshing");
    setTrackingAuthTemporarilyUnavailable(null);
    await reassertTrackingAuthSessionAfterRefresh();
    expect(getTrackingAuthAvailability().kind).toBe("SESSION_AVAILABLE");
  });

  it("5. logout → TRACKING_IDENTITY_UNAVAILABLE + tombstone", async () => {
    await publishTrackingAuthSessionAvailable({
      driverId: 19,
      trackingIdentityId: "driver:19:company:1",
    });
    await clearTrackingAuthSession({ reason: "logout" });
    expect(getTrackingAuthAvailability().kind).toBe("TRACKING_IDENTITY_UNAVAILABLE");
    const presence = await readTrackingAuthPresence();
    expect(presence?.logoutTombstoneAt).not.toBeNull();
  });

  it("6. headless runtime recréé + session valide → authUsable", async () => {
    await publishTrackingAuthSessionAvailable({
      driverId: 19,
      trackingIdentityId: "driver:19:company:1",
    });
    __resetTrackingAuthDecisionForTests();
    const auth = await ensureTrackingAuthAvailabilityForHeadless();
    expect(auth.kind).toBe("SESSION_AVAILABLE");
    const check = validateNativeOwnerForHeadless({
      owner: {
        driverId: 19,
        sessionGenerationId: 10,
        trackingGenerationId: "gen-19",
        trackingIdentityId: "driver:19:company:1",
        missionContextVersion: 1,
        missionId: 26,
      },
      lease: leaseFor(19, "driver:19:company:1"),
      authUsable: true,
    });
    expect(check).toEqual({ ok: true });
  });

  it("7. headless après logout → refuse", async () => {
    await publishTrackingAuthSessionAvailable({
      driverId: 19,
      trackingIdentityId: "driver:19:company:1",
    });
    await clearTrackingAuthSession({ reason: "logout" });
    __resetTrackingAuthDecisionForTests();
    const auth = await ensureTrackingAuthAvailabilityForHeadless();
    expect(auth.kind).toBe("TRACKING_IDENTITY_UNAVAILABLE");
    const check = validateNativeOwnerForHeadless({
      owner: {
        driverId: 19,
        sessionGenerationId: 10,
        trackingGenerationId: "gen-19",
        trackingIdentityId: "driver:19:company:1",
        missionContextVersion: 1,
        missionId: 26,
      },
      lease: leaseFor(19, "driver:19:company:1"),
      authUsable: false,
    });
    expect(check.ok).toBe(false);
    if (!check.ok) expect(check.reason).toBe("auth_not_usable");
  });

  it("8. A→B : ancien owner/lease A refusé ; headless B OK", async () => {
    await publishTrackingAuthSessionAvailable({
      driverId: 19,
      trackingIdentityId: "driver:19:company:1",
      sessionGenerationId: 10,
    });
    // Login B
    mockGetSessionGenerationId.mockReturnValue(11);
    mockReadEnvelope.mockResolvedValue({
      status: "found",
      value: {
        schema_version: 1,
        session_id: "s2",
        device_installation_id: "d1",
        user_public_id: "u2",
        driver_id: 42,
        role: "driver",
        active_context_id: "driver:42",
        refresh_generation: 1,
        last_authenticated_at: new Date().toISOString(),
      },
    });
    mockReadLease.mockResolvedValue(leaseFor(42, "driver:42:company:1", 11));
    await publishTrackingAuthSessionAvailable({
      driverId: 42,
      trackingIdentityId: "driver:42:company:1",
      sessionGenerationId: 11,
    });

    __resetTrackingAuthDecisionForTests();
    const authB = await ensureTrackingAuthAvailabilityForHeadless();
    expect(authB.kind).toBe("SESSION_AVAILABLE");
    if (authB.kind === "SESSION_AVAILABLE") {
      expect(authB.driverId).toBe(42);
    }

    const rejectA = validateNativeOwnerForHeadless({
      owner: {
        driverId: 19,
        sessionGenerationId: 10,
        trackingGenerationId: "gen-19",
        trackingIdentityId: "driver:19:company:1",
        missionContextVersion: 1,
        missionId: 26,
      },
      lease: leaseFor(42, "driver:42:company:1", 11),
      authUsable: true,
    });
    expect(rejectA.ok).toBe(false);

    const acceptB = validateNativeOwnerForHeadless({
      owner: {
        driverId: 42,
        sessionGenerationId: 11,
        trackingGenerationId: "gen-42",
        trackingIdentityId: "driver:42:company:1",
        missionContextVersion: 1,
        missionId: 26,
      },
      lease: leaseFor(42, "driver:42:company:1", 11),
      authUsable: true,
    });
    expect(acceptB).toEqual({ ok: true });
  });

  it("absence presence → auth_not_usable après hydrate", async () => {
    __resetTrackingAuthDecisionForTests();
    const auth = await ensureTrackingAuthAvailabilityForHeadless();
    expect(auth.kind).toBe("TRACKING_IDENTITY_UNAVAILABLE");
  });

  it("presence A + envelope B → reject hydrate", async () => {
    await publishTrackingAuthSessionAvailable({
      driverId: 19,
      trackingIdentityId: "driver:19:company:1",
    });
    mockReadEnvelope.mockResolvedValue({
      status: "found",
      value: {
        schema_version: 1,
        session_id: "s2",
        device_installation_id: "d1",
        user_public_id: "u2",
        driver_id: 42,
        role: "driver",
        active_context_id: "driver:42",
        refresh_generation: 1,
        last_authenticated_at: new Date().toISOString(),
      },
    });
    __resetTrackingAuthDecisionForTests();
    const auth = await hydrateTrackingAuthFromPersistedState();
    expect(auth.kind).toBe("TRACKING_IDENTITY_UNAVAILABLE");
  });

  it("AUTH_TEMPORARILY_UNAVAILABLE n'est pas écrasé par hydrate", async () => {
    await publishTrackingAuthSessionAvailable({
      driverId: 19,
      trackingIdentityId: "driver:19:company:1",
    });
    setTrackingAuthTemporarilyUnavailable("refreshing");
    const auth = await hydrateTrackingAuthFromPersistedState();
    expect(auth.kind).toBe("AUTH_TEMPORARILY_UNAVAILABLE");
  });
});
