/**
 * État runtime partagé : erreurs de démarrage natif, dernier callback task, pending FGS.
 * Source unique pour bannière, QA panel et télémétrie P1.
 */

export type PendingFgsStartState = {
  active: boolean;
  reason?: string;
  missionId: number | null;
  deferredAt?: number;
};

export type TrackingRuntimeSnapshot = {
  lastNativeStartError: string | null;
  lastNativeStartErrorAt: number | null;
  lastTaskInvokedAt: number | null;
  pendingFgsStart: PendingFgsStartState;
};

type TrackingRuntimeListener = (snapshot: TrackingRuntimeSnapshot) => void;

const listeners = new Set<TrackingRuntimeListener>();

let lastNativeStartError: string | null = null;
let lastNativeStartErrorAt: number | null = null;
let lastTaskInvokedAt: number | null = null;
let pendingFgsStart: PendingFgsStartState = { active: false };

function buildSnapshot(): TrackingRuntimeSnapshot {
  return {
    lastNativeStartError,
    lastNativeStartErrorAt,
    lastTaskInvokedAt,
    pendingFgsStart: { ...pendingFgsStart },
  };
}

function notifyListeners(): void {
  const snapshot = buildSnapshot();
  for (const listener of listeners) {
    try {
      listener(snapshot);
    } catch {
      // noop
    }
  }
}

export function getTrackingRuntimeSnapshot(): TrackingRuntimeSnapshot {
  return buildSnapshot();
}

export function subscribeTrackingRuntime(listener: TrackingRuntimeListener): () => void {
  listeners.add(listener);
  listener(buildSnapshot());
  return () => {
    listeners.delete(listener);
  };
}

export function recordNativeStartFailure(payload: {
  reason: string;
  error: string;
}): void {
  lastNativeStartError = `${payload.reason}: ${payload.error}`.slice(0, 500);
  lastNativeStartErrorAt = Date.now();
  notifyListeners();
}

export function clearNativeStartFailure(): void {
  if (lastNativeStartError === null && lastNativeStartErrorAt === null) return;
  lastNativeStartError = null;
  lastNativeStartErrorAt = null;
  notifyListeners();
}

export function setLastTaskInvokedAt(timestampMs: number = Date.now()): void {
  lastTaskInvokedAt = timestampMs;
  notifyListeners();
}

export function getPendingFgsStart(): PendingFgsStartState {
  return { ...pendingFgsStart };
}

export function setPendingFgsStart(state: PendingFgsStartState): void {
  pendingFgsStart = { ...state };
  notifyListeners();
}

export function clearPendingFgsStart(): void {
  if (!pendingFgsStart.active) return;
  pendingFgsStart = { active: false };
  notifyListeners();
}

/** Appelé au boot (NativeCapabilitiesProvider) — réservé pour hydratation persistée future. */
export async function hydrateTrackingRuntimeState(): Promise<void> {
  // no-op Sprint 1
}

/** Test-only reset */
export function __resetTrackingRuntimeForTests(): void {
  lastNativeStartError = null;
  lastNativeStartErrorAt = null;
  lastTaskInvokedAt = null;
  pendingFgsStart = { active: false };
  listeners.clear();
}
