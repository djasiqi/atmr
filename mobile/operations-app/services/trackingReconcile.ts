/**
 * P1 — Réconciliation : mutex async, skip si policy inchangée, shadow P1.0.
 * L’exécution native reste dans locationTracker jusqu’à bascule EXPO_PUBLIC_TRACKING_RECONCILE_APPLY.
 * Si APPLY=true : retirer les appels dupliqués à reconcileBackgroundTrackingState dans _layout.tsx
 * (sinon double réconciliation native).
 */

import AsyncStorage from "@react-native-async-storage/async-storage";
import { Platform } from "react-native";
import { getLogger } from "@/utils/logger";
import { MissionStateManager } from "./missionState";
import type { BgTrackingInputs } from "./backgroundTrackingGating";
import {
  getAdaptiveLocationTracker,
  getPendingBackgroundTrackingStart,
  getPersistedLocationMode,
  getTrackingRuntimeState,
  reconcileBackgroundTrackingState,
} from "./locationTracker";
import { getNetworkStateSnapshot } from "./networkState";
import { rescheduleSyncEngineIntervalsFromPolicy } from "./syncEngine";
import {
  buildResolveTrackingPolicyInput,
  computeLegacyTrackingShadowSnapshot,
  computeTrackingPolicyShadowDiff,
  isTrackingPolicyStructurallyEqual,
  resolveTrackingPolicy,
  type DegradedReason,
  type ResolveTrackingPolicyInput,
  type TrackingDecisionInputsSnapshot,
  type TrackingPolicy,
} from "./trackingPolicy";
import {
  getLastResolvedTrackingPolicy,
  patchLastAppliedExecutionState,
  setLastDecisionInputsSnapshot,
  setLastResolvedPolicy,
} from "./trackingRuntime";

const log = getLogger("TrackingReconcile");
const trackLog = getLogger("TRACK");

/** P1.2 — quand true, reconcile applique aussi le chemin natif existant (sinon shadow + runtime uniquement). */
export function isTrackingReconcileApplyEnabled(): boolean {
  return process.env.EXPO_PUBLIC_TRACKING_RECONCILE_APPLY === "true";
}

/** Mutex async : pas de réconciliation concurrente ; les appels se sérialisent. */
let reconcileMutex: Promise<void> = Promise.resolve();

function networkReachableFromSnapshot(): boolean {
  const s = getNetworkStateSnapshot();
  if (!s) return true;
  const c = s.isConnected;
  const r = s.isInternetReachable;
  if (c === false) return false;
  if (r === false) return false;
  return true;
}

async function driverTokenAvailable(): Promise<boolean> {
  try {
    const id = await AsyncStorage.getItem("driver_id");
    return !!id && id.length > 0;
  } catch {
    return false;
  }
}

function buildDecisionSnapshot(
  resolveInput: ResolveTrackingPolicyInput,
  reason: string
): TrackingDecisionInputsSnapshot {
  return {
    ...resolveInput,
    reconcileReason: reason,
    capturedAtMs: Date.now(),
  };
}

function refreshAppliedExecutionFromRuntime(): void {
  const fg = getAdaptiveLocationTracker().getStats().isTracking;
  const bg = getTrackingRuntimeState().started;
  const pending = getPendingBackgroundTrackingStart();
  const policy = getLastResolvedTrackingPolicy();

  let appliedExecutionGapReason: DegradedReason | null = null;
  if (
    policy?.shouldRunNativeBackgroundTracking &&
    !bg &&
    !pending.active &&
    Platform.OS !== "web" &&
    !__DEV__
  ) {
    appliedExecutionGapReason = "native_tracking_not_started";
    trackLog.info("TRACKING_NATIVE_MISMATCH", {
      intentNativeBg: true,
      runtimeStarted: bg,
      pendingFgs: pending.active,
      policyMode: policy.mode,
    });
  }

  patchLastAppliedExecutionState({
    foregroundTrackerRunning: fg,
    nativeBackgroundRunning: bg,
    pendingAndroidForegroundStart: pending.active,
    appliedExecutionGapReason,
  });
}

async function reconcileTrackingStateInner(
  reason: string,
  inputs: BgTrackingInputs
): Promise<void> {
  const ms = MissionStateManager.getState();
  const persisted = await getPersistedLocationMode(inputs.hasActiveMission);
  const pendingBg = getPendingBackgroundTrackingStart();
  const tokenOk = await driverTokenAvailable();

  const resolveInput = buildResolveTrackingPolicyInput({
    bgInputs: inputs,
    pendingAndroidFgsDeferred: pendingBg.active,
    networkReachable: networkReachableFromSnapshot(),
    driverTokenAvailable: tokenOk,
    missionBarStatus: ms.activeMission ? ms.currentStatus : null,
    persistedLocationMode: persisted,
  });

  const resolved = resolveTrackingPolicy(resolveInput);
  const previous = getLastResolvedTrackingPolicy();

  if (isTrackingPolicyStructurallyEqual(previous, resolved)) {
    setLastDecisionInputsSnapshot(buildDecisionSnapshot(resolveInput, reason));
    refreshAppliedExecutionFromRuntime();
    return;
  }

  setLastResolvedPolicy(resolved);
  setLastDecisionInputsSnapshot(buildDecisionSnapshot(resolveInput, reason));

  const legacy = computeLegacyTrackingShadowSnapshot({
    bgInputs: inputs,
    appState: resolveInput.appState,
    currentLocationMode: inputs.locationMode,
  });
  const { diffDetected, diffReason } = computeTrackingPolicyShadowDiff({
    resolved,
    legacy,
  });

  if (diffDetected) {
    trackLog.info("TRACKING_POLICY_SHADOW_DIFF", {
      reason,
      diffDetected: true,
      diffReason,
      legacyMode: legacy.legacyMode,
      resolvedMode: resolved.mode,
      legacyBgShouldRun: legacy.legacyBgShouldRun,
      resolvedNativeBg: resolved.shouldRunNativeBackgroundTracking,
      legacyFgTracker: legacy.legacyForegroundTrackerAssumed,
      resolvedFgTracker: resolved.shouldRunForegroundTracker,
    });
  }

  trackLog.info("TRACKING_POLICY_TRANSITION", {
    reason,
    fromMode: previous?.mode ?? null,
    toMode: resolved.mode,
    degraded: resolved.degraded,
    degradedReason: resolved.degradedReason,
    policyReason: resolved.reason,
  });

  refreshAppliedExecutionFromRuntime();
  rescheduleSyncEngineIntervalsFromPolicy();

  if (isTrackingReconcileApplyEnabled()) {
    try {
      await reconcileBackgroundTrackingState(reason, inputs);
    } catch (e: unknown) {
      const err = e as { message?: string };
      patchLastAppliedExecutionState({
        lastNativeStartError: err?.message ?? String(e),
      });
    }
    refreshAppliedExecutionFromRuntime();
    rescheduleSyncEngineIntervalsFromPolicy();
  }
}

/**
 * Point d’entrée central : mutex async ; await avant reconcileBackgroundTrackingState pour policy à jour.
 */
export function reconcileTrackingState(
  reason: string,
  prebuiltInputs: BgTrackingInputs
): Promise<void> {
  const run = async (): Promise<void> => {
    try {
      await reconcileTrackingStateInner(reason, prebuiltInputs);
    } catch (e: unknown) {
      const err = e as { message?: string };
      log.warn("reconcileTrackingState failed", { message: err?.message ?? String(e) });
    }
  };
  const p = reconcileMutex.then(run, run);
  reconcileMutex = p.then(
    () => undefined,
    () => undefined
  );
  return p;
}

export function peekResolvedPolicyForTests(): TrackingPolicy | null {
  return getLastResolvedTrackingPolicy();
}
