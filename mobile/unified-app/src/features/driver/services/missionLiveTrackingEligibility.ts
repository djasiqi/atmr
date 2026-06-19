/**
 * Éligibilité au suivi mission live — abstraction capability-oriented (store-safe).
 *
 * Découple la logique métier des mécanismes OS (iOS « Toujours », Android BG + FGS).
 * Le gate transition (EN_ROUTE / IN_PROGRESS) vérifie permissions + GPS ;
 * le monitoring en mission exige aussi le FGS Android s'il est attendu.
 */
import * as Location from "expo-location";
import { Platform } from "react-native";

import { isExpoLocationPermissionGranted } from "../../../core/location/locationPermissionState";
import type { DriverTransitionStatus } from "../types";
import { getNativeTaskLifecycleStatus } from "./backgroundLocationTask";

export const LIVE_TRACKING_TRANSITIONS = ["EN_ROUTE", "IN_PROGRESS"] as const;

export type LiveTrackingTransition = (typeof LIVE_TRACKING_TRANSITIONS)[number];

export type MissionTrackingCapabilitySnapshot = {
  fgGranted: boolean;
  bgGranted: boolean;
  gpsEnabled: boolean;
  foregroundServiceRunning: boolean;
  platform: "ios" | "android" | "web";
  constraintReason: string | null;
};

export type MissionTrackingCapabilityResult = MissionTrackingCapabilitySnapshot & {
  capable: boolean;
};

export function requiresLiveTrackingPermission(target: DriverTransitionStatus): boolean {
  return (LIVE_TRACKING_TRANSITIONS as readonly string[]).includes(target);
}

function resolveConstraintReason(snapshot: MissionTrackingCapabilitySnapshot): string | null {
  if (!snapshot.fgGranted) return "permission_fg_denied";
  if (!snapshot.bgGranted) return "permission_bg_denied";
  if (!snapshot.gpsEnabled) return "gps_provider_disabled";
  if (
    snapshot.platform === "android" &&
    !snapshot.foregroundServiceRunning
  ) {
    return "fgs_not_running";
  }
  return null;
}

/**
 * @param requireForegroundService — true pour monitoring en mission (bannières) ;
 *   false pour le gate transition (FGS pas encore démarré avant EN_ROUTE).
 */
export function hasMissionTrackingCapability(
  snapshot: MissionTrackingCapabilitySnapshot,
  options: { requireForegroundService?: boolean } = {}
): boolean {
  const { requireForegroundService = true } = options;
  if (snapshot.platform === "web") return false;
  if (!snapshot.fgGranted || !snapshot.bgGranted || !snapshot.gpsEnabled) {
    return false;
  }
  if (snapshot.platform === "android" && requireForegroundService) {
    return snapshot.foregroundServiceRunning;
  }
  return true;
}

export type EvaluateMissionTrackingCapabilityOptions = {
  /** Gate avant démarrage live : permissions + GPS sans exiger FGS déjà actif. */
  forLiveTransition?: boolean;
};

const capabilityRefreshListeners = new Set<() => void>();

export function subscribeMissionTrackingCapabilityRefresh(listener: () => void): () => void {
  capabilityRefreshListeners.add(listener);
  return () => {
    capabilityRefreshListeners.delete(listener);
  };
}

export function notifyMissionTrackingCapabilityRefresh(): void {
  capabilityRefreshListeners.forEach((listener) => listener());
}

export async function evaluateMissionTrackingCapability(
  options: EvaluateMissionTrackingCapabilityOptions = {}
): Promise<MissionTrackingCapabilityResult> {
  const platform =
    Platform.OS === "ios" ? "ios" : Platform.OS === "android" ? "android" : "web";

  if (platform === "web") {
    return {
      fgGranted: false,
      bgGranted: false,
      gpsEnabled: false,
      foregroundServiceRunning: false,
      platform,
      constraintReason: "web_unsupported",
      capable: false,
    };
  }

  const [fg, bg, gpsEnabled, lifecycle] = await Promise.all([
    Location.getForegroundPermissionsAsync().catch(() => ({ granted: false })),
    Location.getBackgroundPermissionsAsync().catch(() => ({ granted: false })),
    Location.hasServicesEnabledAsync().catch(() => false),
    getNativeTaskLifecycleStatus().catch(() => ({
      taskDefined: false,
      taskStarted: false,
    })),
  ]);

  const snapshot: MissionTrackingCapabilitySnapshot = {
    fgGranted: isExpoLocationPermissionGranted(fg),
    bgGranted: isExpoLocationPermissionGranted(bg),
    gpsEnabled: Boolean(gpsEnabled),
    foregroundServiceRunning: Boolean(lifecycle.taskStarted),
    platform,
    constraintReason: null,
  };

  snapshot.constraintReason = resolveConstraintReason(snapshot);

  const capable = hasMissionTrackingCapability(snapshot, {
    requireForegroundService: !options.forLiveTransition,
  });

  return { ...snapshot, capable };
}
