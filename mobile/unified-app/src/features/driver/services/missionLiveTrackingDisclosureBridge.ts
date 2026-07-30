/**
 * Point de vérité unique pour la modale « Suivi de mission » (P2).
 *
 * Ordre Play verrouillé pour le readiness :
 * disclosure complète (si 1ʳᵉ fois) → FG → précision → BG (seulement si FG précis OK).
 */
import { Linking, Platform } from "react-native";
import * as Location from "expo-location";

import { isFeatureEnabled } from "../../../core/featureFlags/registry";
import {
  isExpoLocationPermissionGranted,
  resolveLocationAccuracy,
} from "../../../core/location/locationPermissionState";
import { emitDriverTelemetry } from "../../../core/observability/driverTelemetry";
import type { DriverTransitionStatus } from "../types";
import {
  evaluateMissionTrackingCapability,
  notifyMissionTrackingCapabilityRefresh,
  requiresLiveTrackingPermission,
} from "./missionLiveTrackingEligibility";
import {
  isLiveTrackingDisclosureAccepted,
  isPresenceDisclosureAccepted,
  markLiveTrackingDisclosureAccepted,
} from "./liveTrackingDisclosureSession";
import {
  markTrackingOnboarded,
  setTrackingNeedsAttention,
} from "./trackingReadinessPersistence";
import { setDriverMissionDisclosureVisible } from "./driverDisclosureOrchestrator";

export type MissionLiveTrackingDisclosureSnapshot = {
  visible: boolean;
  pending: boolean;
  showOpenSettings: boolean;
  compact: boolean;
};

export type ReadinessLocationAction =
  | "foreground"
  | "background"
  | "accuracy";

type PendingRequest = {
  missionId: number | null;
  target: DriverTransitionStatus | null;
  onProceed: (() => void) | null;
  onComplete: (() => void) | null;
  readinessAction: ReadinessLocationAction | null;
};

const INITIAL: MissionLiveTrackingDisclosureSnapshot = {
  visible: false,
  pending: false,
  showOpenSettings: false,
  compact: false,
};

let snapshot: MissionLiveTrackingDisclosureSnapshot = { ...INITIAL };
let pendingRequest: PendingRequest = {
  missionId: null,
  target: null,
  onProceed: null,
  onComplete: null,
  readinessAction: null,
};
let permissionRequestedThisAttempt = false;

const listeners = new Set<() => void>();

function notify(): void {
  listeners.forEach((listener) => listener());
}

function setSnapshot(next: MissionLiveTrackingDisclosureSnapshot): void {
  snapshot = next;
  setDriverMissionDisclosureVisible(next.visible);
  notify();
}

function closeDisclosure(): void {
  permissionRequestedThisAttempt = false;
  pendingRequest = {
    missionId: null,
    target: null,
    onProceed: null,
    onComplete: null,
    readinessAction: null,
  };
  setSnapshot({ ...INITIAL });
}

export function getMissionLiveTrackingDisclosureSnapshot(): MissionLiveTrackingDisclosureSnapshot {
  return snapshot;
}

export function subscribeMissionLiveTrackingDisclosure(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

function backgroundTrackingAvailable(): boolean {
  return (
    isFeatureEnabled("tracking_background_enabled") &&
    typeof Location.requestBackgroundPermissionsAsync === "function"
  );
}

/** Parcours mission : FG puis BG si FG accordée (après disclosure). */
async function requestMissionTrackingPermissions(): Promise<{
  fgGranted: boolean;
  bgGranted: boolean;
}> {
  const fg = await Location.requestForegroundPermissionsAsync().catch(() => ({
    granted: false,
  }));
  if (!isExpoLocationPermissionGranted(fg)) {
    return { fgGranted: false, bgGranted: false };
  }
  if (!backgroundTrackingAvailable()) {
    return { fgGranted: true, bgGranted: false };
  }
  const accuracy = resolveLocationAccuracy(fg);
  if (accuracy !== "precise") {
    return { fgGranted: true, bgGranted: false };
  }
  const bg = await Location.requestBackgroundPermissionsAsync().catch(() => ({
    granted: false,
  }));
  return { fgGranted: true, bgGranted: Boolean(bg.granted) };
}

/**
 * Parcours readiness : une anomalie à la fois ; BG seulement si FG + précision OK.
 */
async function requestPermissionsForReadinessAction(
  action: ReadinessLocationAction
): Promise<{ fgGranted: boolean; bgGranted: boolean }> {
  if (action === "accuracy") {
    const fg = await Location.requestForegroundPermissionsAsync().catch(() => ({
      granted: false,
    }));
    const fgGranted = isExpoLocationPermissionGranted(fg);
    return { fgGranted, bgGranted: false };
  }

  if (action === "background") {
    const currentFg = await Location.getForegroundPermissionsAsync().catch(() => ({
      granted: false,
    }));
    let fgGranted = isExpoLocationPermissionGranted(currentFg);
    let accuracy = resolveLocationAccuracy(currentFg);
    if (!fgGranted || accuracy !== "precise") {
      const fg = await Location.requestForegroundPermissionsAsync().catch(() => ({
        granted: false,
      }));
      fgGranted = isExpoLocationPermissionGranted(fg);
      accuracy = resolveLocationAccuracy(fg);
      if (!fgGranted || accuracy !== "precise") {
        return { fgGranted, bgGranted: false };
      }
    }
    if (!backgroundTrackingAvailable()) {
      return { fgGranted: true, bgGranted: false };
    }
    const bg = await Location.requestBackgroundPermissionsAsync().catch(() => ({
      granted: false,
    }));
    return { fgGranted: true, bgGranted: Boolean(bg.granted) };
  }

  // foreground : FG seule, puis BG seulement si précis
  const fg = await Location.requestForegroundPermissionsAsync().catch(() => ({
    granted: false,
  }));
  const fgGranted = isExpoLocationPermissionGranted(fg);
  if (!fgGranted) {
    return { fgGranted: false, bgGranted: false };
  }
  if (resolveLocationAccuracy(fg) !== "precise") {
    return { fgGranted: true, bgGranted: false };
  }
  if (!backgroundTrackingAvailable()) {
    return { fgGranted: true, bgGranted: false };
  }
  const bg = await Location.requestBackgroundPermissionsAsync().catch(() => ({
    granted: false,
  }));
  return { fgGranted: true, bgGranted: Boolean(bg.granted) };
}

function hasPriorLocationDisclosure(): boolean {
  return isPresenceDisclosureAccepted() || isLiveTrackingDisclosureAccepted();
}

async function runAfterCapability(
  missionId: number | null,
  target: DriverTransitionStatus | null,
  onProceed: (() => void) | null,
  onComplete: (() => void) | null
): Promise<void> {
  const capability = await evaluateMissionTrackingCapability({ forLiveTransition: true });
  if (capability.capable) {
    await setTrackingNeedsAttention(false);
    await markTrackingOnboarded().catch(() => undefined);
    onProceed?.();
    onComplete?.();
    closeDisclosure();
    return;
  }

  if (missionId != null && target != null) {
    emitDriverTelemetry("tracking.transition_blocked_permission", {
      source: "driver.mission_live_tracking_bridge",
      mission_id: missionId,
      target_status: target,
      constraint_reason: capability.constraintReason,
    });
  }

  await setTrackingNeedsAttention(true);

  setSnapshot({
    ...snapshot,
    pending: false,
    showOpenSettings: true,
    compact: hasPriorLocationDisclosure(),
  });
}

async function trySilentPermissionPath(
  missionId: number | null,
  target: DriverTransitionStatus | null,
  onProceed: (() => void) | null,
  onComplete: (() => void) | null,
  readinessAction: ReadinessLocationAction | null
): Promise<boolean> {
  if (!hasPriorLocationDisclosure()) return false;

  markLiveTrackingDisclosureAccepted();
  const perms =
    readinessAction != null
      ? await requestPermissionsForReadinessAction(readinessAction)
      : await requestMissionTrackingPermissions();
  permissionRequestedThisAttempt = true;

  if (missionId != null && target != null) {
    emitDriverTelemetry("tracking.mission_live_guard.permission_requested", {
      source: "driver.mission_live_tracking_bridge",
      mission_id: missionId,
      target_status: target,
      platform: Platform.OS,
      fg_granted: perms.fgGranted,
      bg_granted: perms.bgGranted,
      silent: true,
    });
  }

  if (readinessAction != null) {
    const accuracyOk =
      readinessAction !== "accuracy" ||
      (await Location.getForegroundPermissionsAsync()
        .then((fg) => resolveLocationAccuracy(fg) === "precise")
        .catch(() => false));
    const actionResolved =
      readinessAction === "foreground"
        ? perms.fgGranted
        : readinessAction === "background"
          ? perms.bgGranted
          : accuracyOk;
    if (actionResolved) {
      onComplete?.();
      closeDisclosure();
      return true;
    }
    // Disclosure déjà acceptée mais anomalie encore ouverte → modal compacte.
    return false;
  }

  const capability = await evaluateMissionTrackingCapability({ forLiveTransition: true });
  if (capability.capable) {
    await setTrackingNeedsAttention(false);
    await markTrackingOnboarded().catch(() => undefined);
    onProceed?.();
    onComplete?.();
    closeDisclosure();
    return true;
  }

  return false;
}

function openDisclosureModal(params: {
  missionId: number | null;
  target: DriverTransitionStatus | null;
  onProceed: (() => void) | null;
  onComplete: (() => void) | null;
  compact?: boolean;
  readinessAction?: ReadinessLocationAction | null;
}): void {
  permissionRequestedThisAttempt = false;
  pendingRequest = {
    missionId: params.missionId,
    target: params.target,
    onProceed: params.onProceed,
    onComplete: params.onComplete,
    readinessAction: params.readinessAction ?? null,
  };

  if (params.missionId != null && params.target != null) {
    emitDriverTelemetry("tracking.mission_live_guard.disclosure_shown", {
      source: "driver.mission_live_tracking_bridge",
      mission_id: params.missionId,
      target_status: params.target,
      platform: Platform.OS,
      compact: Boolean(params.compact),
    });
  }

  setSnapshot({
    visible: true,
    pending: false,
    showOpenSettings: false,
    compact: Boolean(params.compact),
  });
}

export function guardMissionLiveTransition(params: {
  missionId: number;
  target: DriverTransitionStatus;
  onProceed: () => void;
}): void {
  const { missionId, target, onProceed } = params;

  if (
    !isFeatureEnabled("driver_mission_live_tracking_guard_enabled") ||
    !isFeatureEnabled("tracking_background_enabled")
  ) {
    onProceed();
    return;
  }

  if (!requiresLiveTrackingPermission(target)) {
    onProceed();
    return;
  }

  void (async () => {
    const capability = await evaluateMissionTrackingCapability({ forLiveTransition: true });
    if (capability.capable) {
      await setTrackingNeedsAttention(false);
      onProceed();
      return;
    }

    const silentOk = await trySilentPermissionPath(missionId, target, onProceed, null, null);
    if (silentOk) return;

    openDisclosureModal({
      missionId,
      target,
      onProceed,
      onComplete: null,
      compact: hasPriorLocationDisclosure(),
    });
  })();
}

export function openMissionLiveTrackingDisclosureForBanner(): void {
  void (async () => {
    const silentOk = await trySilentPermissionPath(null, null, null, () => {
      notifyMissionTrackingCapabilityRefresh();
    }, null);
    if (silentOk) return;

    openDisclosureModal({
      missionId: null,
      target: null,
      onProceed: null,
      onComplete: () => notifyMissionTrackingCapabilityRefresh(),
    });
  })();
}

/**
 * Gate readiness : disclosure complète avant toute 1ʳᵉ demande FG/BG.
 * Compact uniquement si disclosure déjà acceptée et parcours silencieux insuffisant.
 */
export function openMissionLiveTrackingDisclosureForReadiness(
  onComplete: () => void,
  action: ReadinessLocationAction = "foreground"
): void {
  void (async () => {
    const silentOk = await trySilentPermissionPath(null, null, null, onComplete, action);
    if (silentOk) return;

    openDisclosureModal({
      missionId: null,
      target: null,
      onProceed: null,
      onComplete,
      compact: hasPriorLocationDisclosure(),
      readinessAction: action,
    });
  })();
}

export function cancelMissionLiveTrackingDisclosure(): void {
  closeDisclosure();
}

export function continueMissionLiveTrackingDisclosure(): void {
  const { missionId, target, onProceed, onComplete, readinessAction } = pendingRequest;
  if (missionId == null && !onComplete) return;

  if (permissionRequestedThisAttempt) {
    if (readinessAction != null) {
      onComplete?.();
      closeDisclosure();
      return;
    }
    void runAfterCapability(missionId, target, onProceed, onComplete);
    return;
  }

  setSnapshot({ ...snapshot, pending: true });
  markLiveTrackingDisclosureAccepted();

  void (async () => {
    const perms =
      readinessAction != null
        ? await requestPermissionsForReadinessAction(readinessAction)
        : await requestMissionTrackingPermissions();
    permissionRequestedThisAttempt = true;

    if (missionId != null && target != null) {
      emitDriverTelemetry("tracking.mission_live_guard.permission_requested", {
        source: "driver.mission_live_tracking_bridge",
        mission_id: missionId,
        target_status: target,
        platform: Platform.OS,
        fg_granted: perms.fgGranted,
        bg_granted: perms.bgGranted,
      });
    }

    if (readinessAction != null) {
      onComplete?.();
      closeDisclosure();
      return;
    }

    await runAfterCapability(missionId, target, onProceed, onComplete);
  })();
}

export function openMissionLiveTrackingSettings(): void {
  if (Platform.OS === "ios") {
    void Linking.openURL("app-settings:");
  } else {
    void Linking.openSettings();
  }
}

/** Test-only */
export function __resetMissionLiveTrackingDisclosureBridgeForTests(): void {
  closeDisclosure();
}
