/**
 * P1 — Runtime JS : intention (policy) vs réalité (applied execution).
 * Thread-safe côté JS : un seul thread ; mutex async dans trackingReconcile.
 */

import type { DegradedReason, TrackingDecisionInputsSnapshot, TrackingPolicy } from "./trackingPolicy";

export type FlushPathUsed = "socket" | "http" | "none" | "deferred";

/**
 * Ce qui a réellement été appliqué côté natif / foreground / flush.
 * `appliedExecutionGapReason` : écart intention vs natif (ne pas muter lastResolvedPolicy).
 */
export type AppliedExecutionState = {
  foregroundTrackerRunning: boolean;
  nativeBackgroundRunning: boolean;
  lastNativeStartError: string | null;
  flushPathUsed: FlushPathUsed;
  /** Aligné pending FGS Android (locationTracker). */
  pendingAndroidForegroundStart: boolean;
  /**
   * Si la policy exige le natif mais qu’il ne tourne pas (hors cas différé / web / dev).
   * @see DegradedReason — valeur `native_tracking_not_started` côté exécution uniquement.
   */
  appliedExecutionGapReason: DegradedReason | null;
};

let lastResolvedPolicy: TrackingPolicy | null = null;
let lastAppliedExecutionState: AppliedExecutionState = {
  foregroundTrackerRunning: false,
  nativeBackgroundRunning: false,
  lastNativeStartError: null,
  flushPathUsed: "none",
  pendingAndroidForegroundStart: false,
  appliedExecutionGapReason: null,
};

/** Entrées de décision alignées sur resolveTrackingPolicy (pas fusionné avec applied). */
let lastDecisionInputsSnapshot: TrackingDecisionInputsSnapshot | null = null;

export function getLastResolvedTrackingPolicy(): TrackingPolicy | null {
  return lastResolvedPolicy;
}

export function setLastResolvedPolicy(policy: TrackingPolicy | null): void {
  lastResolvedPolicy = policy;
}

export function getLastAppliedExecutionState(): AppliedExecutionState {
  return { ...lastAppliedExecutionState };
}

export function setLastAppliedExecutionState(state: AppliedExecutionState): void {
  lastAppliedExecutionState = { ...state };
}

export function patchLastAppliedExecutionState(
  partial: Partial<AppliedExecutionState>
): void {
  lastAppliedExecutionState = { ...lastAppliedExecutionState, ...partial };
}

export function setLastDecisionInputsSnapshot(s: TrackingDecisionInputsSnapshot | null): void {
  lastDecisionInputsSnapshot = s;
}

export function getLastDecisionInputsSnapshot(): TrackingDecisionInputsSnapshot | null {
  return lastDecisionInputsSnapshot ? { ...lastDecisionInputsSnapshot } : null;
}
