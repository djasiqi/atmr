/**
 * Point de vérité unique pour la modale « Suivi de mission » (P2).
 */
import { Linking, Platform } from "react-native";
import * as Location from "expo-location";

import { isFeatureEnabled } from "../../../core/featureFlags/registry";
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

type PendingRequest = {
  missionId: number | null;
  target: DriverTransitionStatus | null;
  onProceed: (() => void) | null;
  onComplete: (() => void) | null;
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

async function requestMissionTrackingPermissions(): Promise<{
  fgGranted: boolean;
  bgGranted: boolean;
}> {
  const fg = await Location.requestForegroundPermissionsAsync().catch(() => ({
    granted: false,
  }));
  if (!fg.granted) {
    return { fgGranted: false, bgGranted: false };
  }
  if (
    !isFeatureEnabled("tracking_background_enabled") ||
    typeof Location.requestBackgroundPermissionsAsync !== "function"
  ) {
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
  onComplete: (() => void) | null
): Promise<boolean> {
  if (!hasPriorLocationDisclosure()) return false;

  markLiveTrackingDisclosureAccepted();
  const perms = await requestMissionTrackingPermissions();
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
}): void {
  permissionRequestedThisAttempt = false;
  pendingRequest = {
    missionId: params.missionId,
    target: params.target,
    onProceed: params.onProceed,
    onComplete: params.onComplete,
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

    const silentOk = await trySilentPermissionPath(missionId, target, onProceed, null);
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
    });
    if (silentOk) return;

    openDisclosureModal({
      missionId: null,
      target: null,
      onProceed: null,
      onComplete: () => notifyMissionTrackingCapabilityRefresh(),
    });
  })();
}

export function openMissionLiveTrackingDisclosureForReadiness(onComplete: () => void): void {
  void (async () => {
    const silentOk = await trySilentPermissionPath(null, null, null, onComplete);
    if (silentOk) return;

    openDisclosureModal({
      missionId: null,
      target: null,
      onProceed: null,
      onComplete,
      compact: hasPriorLocationDisclosure(),
    });
  })();
}

export function cancelMissionLiveTrackingDisclosure(): void {
  closeDisclosure();
}

export function continueMissionLiveTrackingDisclosure(): void {
  const { missionId, target, onProceed, onComplete } = pendingRequest;
  if (missionId == null && !onComplete) return;

  if (permissionRequestedThisAttempt) {
    void runAfterCapability(missionId, target, onProceed, onComplete);
    return;
  }

  setSnapshot({ ...snapshot, pending: true });
  markLiveTrackingDisclosureAccepted();

  void (async () => {
    const perms = await requestMissionTrackingPermissions();
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
