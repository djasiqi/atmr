/**
 * P0-B — Presence tracking persistée (sans JWT) + hydrate headless.
 *
 * Invariants :
 * - Presence ≠ authentification HTTP ; pas de token stocké ici.
 * - SESSION_AVAILABLE mémoire = cache ; source de vérité = presence + SecureStore + lease.
 * - Logout / identity change invalident immédiatement la presence.
 */

import AsyncStorage from "@react-native-async-storage/async-storage";
import {
  getSessionGenerationId,
  readRefreshToken,
  readSessionEnvelope,
  type SessionGenerationId,
} from "./authCredentialStore";
import {
  getTrackingAuthAvailability,
  setTrackingAuthAvailability,
  setTrackingAuthTemporarilyUnavailable,
  type TrackingAuthAvailability,
  __resetTrackingAuthDecisionForTests,
} from "./sessionAuthDecision";
import { emitDriverTelemetry } from "../observability/driverTelemetry";
import {
  readTrackingContextLease,
  type TrackingContextLease,
} from "../../features/driver/services/trackingContextLease";

export const TRACKING_AUTH_PRESENCE_STORAGE_KEY = "@driver:tracking_auth_presence_v1";

export type TrackingAuthPresenceV1 = {
  schemaVersion: 1;
  driverId: number;
  trackingIdentityId: string;
  sessionGenerationId: SessionGenerationId;
  updatedAt: number;
  credentialsEpoch: number;
  logoutTombstoneAt: number | null;
};

const inMemoryStorage = new Map<string, string>();

async function readStorage(key: string): Promise<string | null> {
  const storage = AsyncStorage as unknown as {
    getItem?: (input: string) => Promise<string | null>;
  };
  if (typeof storage?.getItem === "function") {
    return storage.getItem(key);
  }
  return inMemoryStorage.get(key) ?? null;
}

async function writeStorage(key: string, value: string): Promise<void> {
  const storage = AsyncStorage as unknown as {
    setItem?: (k: string, v: string) => Promise<void>;
  };
  if (typeof storage?.setItem === "function") {
    await storage.setItem(key, value);
    return;
  }
  inMemoryStorage.set(key, value);
}

async function removeStorage(key: string): Promise<void> {
  const storage = AsyncStorage as unknown as {
    removeItem?: (k: string) => Promise<void>;
  };
  if (typeof storage?.removeItem === "function") {
    await storage.removeItem(key);
    return;
  }
  inMemoryStorage.delete(key);
}

function parsePresence(raw: string | null): TrackingAuthPresenceV1 | null {
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as Partial<TrackingAuthPresenceV1>;
    if (
      parsed.schemaVersion !== 1 ||
      typeof parsed.driverId !== "number" ||
      !Number.isFinite(parsed.driverId) ||
      typeof parsed.trackingIdentityId !== "string" ||
      typeof parsed.sessionGenerationId !== "number" ||
      typeof parsed.updatedAt !== "number" ||
      typeof parsed.credentialsEpoch !== "number"
    ) {
      return null;
    }
    return {
      schemaVersion: 1,
      driverId: parsed.driverId,
      trackingIdentityId: parsed.trackingIdentityId,
      sessionGenerationId: parsed.sessionGenerationId,
      updatedAt: parsed.updatedAt,
      credentialsEpoch: parsed.credentialsEpoch,
      logoutTombstoneAt:
        typeof parsed.logoutTombstoneAt === "number" ? parsed.logoutTombstoneAt : null,
    };
  } catch {
    return null;
  }
}

export async function readTrackingAuthPresence(): Promise<TrackingAuthPresenceV1 | null> {
  return parsePresence(await readStorage(TRACKING_AUTH_PRESENCE_STORAGE_KEY));
}

export async function persistTrackingAuthPresence(input: {
  driverId: number;
  trackingIdentityId: string;
  sessionGenerationId?: SessionGenerationId;
  credentialsEpoch?: number;
}): Promise<TrackingAuthPresenceV1> {
  const sessionGenerationId = input.sessionGenerationId ?? getSessionGenerationId();
  const credentialsEpoch = input.credentialsEpoch ?? sessionGenerationId;
  const next: TrackingAuthPresenceV1 = {
    schemaVersion: 1,
    driverId: input.driverId,
    trackingIdentityId: input.trackingIdentityId,
    sessionGenerationId,
    updatedAt: Date.now(),
    credentialsEpoch,
    logoutTombstoneAt: null,
  };
  await writeStorage(TRACKING_AUTH_PRESENCE_STORAGE_KEY, JSON.stringify(next));
  return next;
}

export async function clearTrackingAuthPresence(opts?: {
  reason?: "logout" | "revoke" | "identity_change";
}): Promise<void> {
  const reason = opts?.reason ?? "logout";
  if (reason === "identity_change") {
    // Écrasement par la nouvelle identity via persist — tombstone optionnelle
    await removeStorage(TRACKING_AUTH_PRESENCE_STORAGE_KEY);
    return;
  }
  const existing = await readTrackingAuthPresence();
  if (!existing) {
    await removeStorage(TRACKING_AUTH_PRESENCE_STORAGE_KEY);
    return;
  }
  const tombstoned: TrackingAuthPresenceV1 = {
    ...existing,
    logoutTombstoneAt: Date.now(),
    updatedAt: Date.now(),
  };
  await writeStorage(TRACKING_AUTH_PRESENCE_STORAGE_KEY, JSON.stringify(tombstoned));
}

/**
 * Publie SESSION_AVAILABLE (cache mémoire) + presence persistée.
 * À appeler au login / restore / entrée chauffeur.
 */
export async function publishTrackingAuthSessionAvailable(input: {
  driverId: number;
  trackingIdentityId: string;
  sessionGenerationId?: SessionGenerationId;
}): Promise<void> {
  const previous = await readTrackingAuthPresence();
  if (
    previous &&
    previous.logoutTombstoneAt == null &&
    previous.driverId !== input.driverId
  ) {
    await clearTrackingAuthPresence({ reason: "identity_change" });
  }
  const presence = await persistTrackingAuthPresence(input);
  setTrackingAuthAvailability({
    kind: "SESSION_AVAILABLE",
    trackingIdentityId: presence.trackingIdentityId,
    driverId: presence.driverId,
  });
  emitDriverTelemetry("tracking.auth.presence.published", {
    driver_id: presence.driverId,
    tracking_identity_id: presence.trackingIdentityId,
    session_generation_id: presence.sessionGenerationId,
    kind: "SESSION_AVAILABLE",
  });
}

/** Logout / revoke : efface presence + snapshot mémoire immédiatement. */
export async function clearTrackingAuthSession(opts?: {
  reason?: "logout" | "revoke" | "identity_change";
}): Promise<void> {
  const reason = opts?.reason ?? "logout";
  await clearTrackingAuthPresence({ reason });
  setTrackingAuthAvailability({ kind: "TRACKING_IDENTITY_UNAVAILABLE" });
  setTrackingAuthTemporarilyUnavailable(null);
  emitDriverTelemetry("tracking.auth.presence.cleared", {
    reason,
    kind: "TRACKING_IDENTITY_UNAVAILABLE",
  });
}

function leaseMatchesPresence(
  lease: TrackingContextLease | null,
  presence: TrackingAuthPresenceV1
): boolean {
  if (!lease || lease.state !== "driver_active") return false;
  return (
    lease.driverId === presence.driverId &&
    lease.trackingIdentityId === presence.trackingIdentityId &&
    lease.sessionGenerationId === presence.sessionGenerationId
  );
}

/**
 * Reconstruit le snapshot mémoire depuis presence + SecureStore + lease.
 * Ne lit jamais le JWT comme secret tracking — seulement le statut found/missing.
 */
export async function hydrateTrackingAuthFromPersistedState(): Promise<TrackingAuthAvailability> {
  // Refresh en cours : ne pas rétrograder en UNAVAILABLE
  const current = getTrackingAuthAvailability();
  if (current.kind === "AUTH_TEMPORARILY_UNAVAILABLE") {
    return current;
  }

  const presence = await readTrackingAuthPresence();
  if (!presence || presence.logoutTombstoneAt != null) {
    setTrackingAuthAvailability({ kind: "TRACKING_IDENTITY_UNAVAILABLE" });
    return getTrackingAuthAvailability();
  }

  const [refresh, envelope] = await Promise.all([readRefreshToken(), readSessionEnvelope()]);

  if (
    refresh.status === "temporarily_unavailable" ||
    envelope.status === "temporarily_unavailable"
  ) {
    setTrackingAuthTemporarilyUnavailable("credential_store_unavailable");
    return getTrackingAuthAvailability();
  }

  const credentialsPresent =
    refresh.status === "found" || envelope.status === "found";
  if (!credentialsPresent) {
    setTrackingAuthAvailability({ kind: "TRACKING_IDENTITY_UNAVAILABLE" });
    return getTrackingAuthAvailability();
  }

  if (envelope.status === "found" && envelope.value.driver_id != null) {
    if (Number(envelope.value.driver_id) !== presence.driverId) {
      setTrackingAuthAvailability({ kind: "TRACKING_IDENTITY_UNAVAILABLE" });
      return getTrackingAuthAvailability();
    }
  }

  const lease = await readTrackingContextLease();
  if (!leaseMatchesPresence(lease, presence)) {
    setTrackingAuthAvailability({ kind: "TRACKING_IDENTITY_UNAVAILABLE" });
    return getTrackingAuthAvailability();
  }

  // Clear temp store flag if we successfully hydrated
  setTrackingAuthTemporarilyUnavailable(null);
  setTrackingAuthAvailability({
    kind: "SESSION_AVAILABLE",
    trackingIdentityId: presence.trackingIdentityId,
    driverId: presence.driverId,
  });
  const next = getTrackingAuthAvailability();
  emitDriverTelemetry("tracking.auth.presence.hydrated", {
    driver_id: presence.driverId,
    tracking_identity_id: presence.trackingIdentityId,
    kind: next.kind,
    source: "persisted_presence",
  });
  return next;
}

/**
 * Point d'entrée headless : cache mémoire si déjà SESSION cohérente, sinon hydrate.
 */
export async function ensureTrackingAuthAvailabilityForHeadless(): Promise<TrackingAuthAvailability> {
  const current = getTrackingAuthAvailability();
  if (current.kind === "AUTH_TEMPORARILY_UNAVAILABLE") {
    emitDriverTelemetry("tracking.auth.presence.ensure_headless", {
      kind: current.kind,
      path: "memory_temp",
    });
    return current;
  }
  if (current.kind === "SESSION_AVAILABLE") {
    const presence = await readTrackingAuthPresence();
    if (
      presence &&
      presence.logoutTombstoneAt == null &&
      presence.driverId === current.driverId &&
      presence.trackingIdentityId === current.trackingIdentityId
    ) {
      emitDriverTelemetry("tracking.auth.presence.ensure_headless", {
        kind: current.kind,
        path: "memory_cache",
        driver_id: current.driverId,
      });
      return current;
    }
  }
  const hydrated = await hydrateTrackingAuthFromPersistedState();
  emitDriverTelemetry("tracking.auth.presence.ensure_headless", {
    kind: hydrated.kind,
    path: "hydrate",
    driver_id: "driverId" in hydrated ? hydrated.driverId : null,
  });
  return hydrated;
}

/**
 * Après refresh token : s'assurer que clear(temp) ne laisse pas UNAVAILABLE
 * si une presence valide existe (ré-hydratation mémoire).
 */
export async function reassertTrackingAuthSessionAfterRefresh(): Promise<void> {
  const presence = await readTrackingAuthPresence();
  if (!presence || presence.logoutTombstoneAt != null) {
    return;
  }
  setTrackingAuthAvailability({
    kind: "SESSION_AVAILABLE",
    trackingIdentityId: presence.trackingIdentityId,
    driverId: presence.driverId,
  });
}

/** Tests uniquement. */
export async function __resetTrackingAuthPresenceForTests(): Promise<void> {
  inMemoryStorage.clear();
  await removeStorage(TRACKING_AUTH_PRESENCE_STORAGE_KEY).catch(() => undefined);
  __resetTrackingAuthDecisionForTests();
}
