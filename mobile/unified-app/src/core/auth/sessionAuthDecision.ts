/**
 * Décisions d'auth typées pour le runtime tracking (Phase 1B).
 * Le tracking consomme uniquement cette API — jamais sessionProvider / logout / SecureStore.
 */
import {
  getSessionGenerationId,
  type SessionGenerationId,
} from "./authCredentialStore";

export type TrackingAuthAvailability =
  | {
      kind: "SESSION_AVAILABLE";
      sessionGenerationId: SessionGenerationId;
      trackingIdentityId: string;
      driverId: number;
    }
  | {
      kind: "AUTH_TEMPORARILY_UNAVAILABLE";
      sessionGenerationId: SessionGenerationId;
      reason: "refreshing" | "network" | "credential_store_unavailable";
    }
  | {
      kind: "TRACKING_IDENTITY_UNAVAILABLE";
      sessionGenerationId: SessionGenerationId;
    };

export type TrackingAuthTerminalEvent =
  | {
      kind: "EXPLICIT_LOGOUT";
      sourceSessionGenerationId: SessionGenerationId;
      operationId: string;
      trackingIdentityId: string | null;
    }
  | {
      kind: "ACCOUNT_REVOKED";
      sourceSessionGenerationId: SessionGenerationId;
      operationId: string;
      trackingIdentityId: string | null;
    }
  | {
      kind: "IDENTITY_CHANGED";
      previousSessionGenerationId: SessionGenerationId;
      nextSessionGenerationId: SessionGenerationId;
      previousTrackingIdentityId: string | null;
      nextTrackingIdentityId: string | null;
    };

/** Politique d'effets tracking — source de vérité testée (docs en dérivent). */
export const TRACKING_AUTH_EFFECT_POLICY = {
  explicit_logout: {
    stop: true,
    quarantine: true,
    restartAllowed: false,
  },
  temporary_refresh_failure: {
    stop: false,
    quarantine: false,
    restartAllowed: true,
  },
  account_revoked: {
    stop: true,
    quarantine: true,
    restartAllowed: false,
  },
  auth_exhausted_socket: {
    stop: false,
    quarantine: false,
    restartAllowed: true,
  },
} as const;

export type TrackingAuthEffectPolicyKey = keyof typeof TRACKING_AUTH_EFFECT_POLICY;

type TerminalListener = (event: TrackingAuthTerminalEvent) => void;

let availabilitySnapshot: TrackingAuthAvailability = {
  kind: "TRACKING_IDENTITY_UNAVAILABLE",
  sessionGenerationId: 0,
};

let temporaryUnavailableReason:
  | "refreshing"
  | "network"
  | "credential_store_unavailable"
  | null = null;

const terminalListeners = new Set<TerminalListener>();
const emittedTerminalOperationIds = new Set<string>();

export function getTrackingAuthAvailability(): TrackingAuthAvailability {
  if (temporaryUnavailableReason) {
    return {
      kind: "AUTH_TEMPORARILY_UNAVAILABLE",
      sessionGenerationId: getSessionGenerationId(),
      reason: temporaryUnavailableReason,
    };
  }
  if (availabilitySnapshot.kind === "SESSION_AVAILABLE") {
    return {
      ...availabilitySnapshot,
      sessionGenerationId: getSessionGenerationId(),
    };
  }
  return {
    kind: "TRACKING_IDENTITY_UNAVAILABLE",
    sessionGenerationId: getSessionGenerationId(),
  };
}

export function setTrackingAuthAvailability(
  next:
    | {
        kind: "SESSION_AVAILABLE";
        trackingIdentityId: string;
        driverId: number;
      }
    | { kind: "TRACKING_IDENTITY_UNAVAILABLE" }
): void {
  if (next.kind === "SESSION_AVAILABLE") {
    availabilitySnapshot = {
      kind: "SESSION_AVAILABLE",
      sessionGenerationId: getSessionGenerationId(),
      trackingIdentityId: next.trackingIdentityId,
      driverId: next.driverId,
    };
    return;
  }
  availabilitySnapshot = {
    kind: "TRACKING_IDENTITY_UNAVAILABLE",
    sessionGenerationId: getSessionGenerationId(),
  };
}

export function setTrackingAuthTemporarilyUnavailable(
  reason: "refreshing" | "network" | "credential_store_unavailable" | null
): void {
  temporaryUnavailableReason = reason;
}

export function subscribeToTrackingAuthTerminalEvents(
  listener: TerminalListener
): () => void {
  terminalListeners.add(listener);
  return () => {
    terminalListeners.delete(listener);
  };
}

/**
 * Émet un événement terminal une fois par operationId (quand présent).
 * IDENTITY_CHANGED n'a pas d'operationId unique — toujours émis.
 */
export function emitTrackingAuthTerminalEvent(
  event: TrackingAuthTerminalEvent
): void {
  if (event.kind === "EXPLICIT_LOGOUT" || event.kind === "ACCOUNT_REVOKED") {
    if (emittedTerminalOperationIds.has(event.operationId)) {
      return;
    }
    emittedTerminalOperationIds.add(event.operationId);
    if (emittedTerminalOperationIds.size > 200) {
      const first = emittedTerminalOperationIds.values().next().value;
      if (first) emittedTerminalOperationIds.delete(first);
    }
  }
  for (const listener of terminalListeners) {
    try {
      listener(event);
    } catch {
      /* listener isolé */
    }
  }
}

/** Tests uniquement. */
export function __resetTrackingAuthDecisionForTests(): void {
  availabilitySnapshot = {
    kind: "TRACKING_IDENTITY_UNAVAILABLE",
    sessionGenerationId: 0,
  };
  temporaryUnavailableReason = null;
  terminalListeners.clear();
  emittedTerminalOperationIds.clear();
}
