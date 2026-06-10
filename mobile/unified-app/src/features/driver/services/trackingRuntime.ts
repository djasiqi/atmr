/**
 * État runtime partagé : erreurs de démarrage natif, dernier callback task, pending FGS,
 * mission/mode courants (consommés par reconcileTrackingRuntime + telémetrie P1).
 * Source unique pour bannière, QA panel et observabilité.
 */

import type { DriverTrackingMode } from "../runtimeContracts";

export type PendingFgsStartState = {
  active: boolean;
  reason?: string;
  missionId: number | null;
  deferredAt?: number;
};

export type TrackingRuntimeState = {
  missionId: number | null;
  mode: DriverTrackingMode;
};

export type NativeStartDiagnostics = {
  native_start_phase: string | null;
  native_start_error: string | null;
  native_task_defined: boolean | null;
  native_started_before: boolean | null;
  native_started_after: boolean | null;
};

export type TrackingRuntimeSnapshot = {
  lastNativeStartError: string | null;
  lastNativeStartErrorAt: number | null;
  lastTaskInvokedAt: number | null;
  pendingFgsStart: PendingFgsStartState;
  missionId: number | null;
  mode: DriverTrackingMode;
  nativeStartDiagnostics: NativeStartDiagnostics;
};

type TrackingRuntimeListener = (snapshot: TrackingRuntimeSnapshot) => void;

const listeners = new Set<TrackingRuntimeListener>();

let lastNativeStartError: string | null = null;
let lastNativeStartErrorAt: number | null = null;
let lastTaskInvokedAt: number | null = null;
let pendingFgsStart: PendingFgsStartState = { active: false };
let currentState: TrackingRuntimeState = { missionId: null, mode: "off" };
let nativeStartDiagnostics: NativeStartDiagnostics = {
  native_start_phase: null,
  native_start_error: null,
  native_task_defined: null,
  native_started_before: null,
  native_started_after: null,
};

function buildSnapshot(): TrackingRuntimeSnapshot {
  return {
    lastNativeStartError,
    lastNativeStartErrorAt,
    lastTaskInvokedAt,
    pendingFgsStart: { ...pendingFgsStart },
    missionId: currentState.missionId,
    mode: currentState.mode,
    nativeStartDiagnostics: { ...nativeStartDiagnostics },
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

export function getNativeStartDiagnostics(): NativeStartDiagnostics {
  return { ...nativeStartDiagnostics };
}

export function recordNativeStartDiagnostics(
  partial: Partial<NativeStartDiagnostics>
): void {
  nativeStartDiagnostics = { ...nativeStartDiagnostics, ...partial };
  notifyListeners();
}

export function clearNativeStartDiagnostics(): void {
  nativeStartDiagnostics = {
    native_start_phase: null,
    native_start_error: null,
    native_task_defined: null,
    native_started_before: null,
    native_started_after: null,
  };
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

export function getTrackingRuntimeState(): TrackingRuntimeState {
  return { ...currentState };
}

export async function updateTrackingRuntimeState(
  next: Partial<TrackingRuntimeState>
): Promise<void> {
  const merged: TrackingRuntimeState = { ...currentState, ...next };
  if (merged.missionId === currentState.missionId && merged.mode === currentState.mode) {
    return;
  }
  currentState = merged;
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
  currentState = { missionId: null, mode: "off" };
  clearNativeStartDiagnostics();
  listeners.clear();
}
