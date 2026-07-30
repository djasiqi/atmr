/**
 * Modèle pur readiness tracking — règles d'évaluation et d'actions conditionnelles.
 */
import { Platform } from "react-native";

import type { LocationAccuracyStatus } from "../../../core/location/locationPermissionState";

export type BatteryReadinessStatus =
  | "not_applicable"
  | "exempt"
  | "restricted"
  | "unknown";

export type LocationReadinessAction =
  | "foreground"
  | "enable_precise"
  | "verify_accuracy"
  | "background"
  | null;

export type TrackingReadinessSnapshot = {
  ready: boolean;
  bgPermissionGranted: boolean;
  fgPermissionGranted: boolean;
  notificationsGranted: boolean;
  /** @deprecated préférer batteryStatus — conservé pour compat tests / bannières. */
  batteryExempt: boolean;
  batteryStatus: BatteryReadinessStatus;
  locationAccuracy: LocationAccuracyStatus;
  gpsEnabled: boolean;
  oem: string | null;
  hasOemSettings: boolean;
  oemGuidanceAcknowledged: boolean;
};

export function resolveBatteryReadinessStatus(params: {
  platformOs: string;
  checked: boolean;
  isIgnoring: boolean | null;
}): BatteryReadinessStatus {
  if (params.platformOs !== "android") {
    return "not_applicable";
  }
  if (!params.checked) {
    return "unknown";
  }
  if (params.isIgnoring === true) {
    return "exempt";
  }
  if (params.isIgnoring === false) {
    return "restricted";
  }
  return "unknown";
}

export function computeTrackingReady(params: {
  fgPermissionGranted: boolean;
  bgPermissionGranted: boolean;
  locationAccuracy: LocationAccuracyStatus;
  gpsEnabled: boolean;
  notificationsGranted: boolean;
  batteryStatus: BatteryReadinessStatus;
}): boolean {
  const locationReady =
    params.fgPermissionGranted &&
    params.bgPermissionGranted &&
    params.locationAccuracy === "precise" &&
    params.gpsEnabled;

  const batteryReady =
    params.batteryStatus === "not_applicable" ||
    params.batteryStatus === "exempt" ||
    params.batteryStatus === "unknown";

  return locationReady && params.notificationsGranted && batteryReady;
}

export function resolveLocationReadinessAction(params: {
  fgPermissionGranted: boolean;
  bgPermissionGranted: boolean;
  locationAccuracy: LocationAccuracyStatus;
}): LocationReadinessAction {
  if (!params.fgPermissionGranted) {
    return "foreground";
  }
  if (params.locationAccuracy === "approximate") {
    return "enable_precise";
  }
  if (params.locationAccuracy === "unknown") {
    return "verify_accuracy";
  }
  if (!params.bgPermissionGranted) {
    return "background";
  }
  return null;
}

export function shouldShowOemGuidance(params: {
  platformOs?: string;
  hasOemSettings: boolean;
  oemGuidanceAcknowledged: boolean;
  batteryStatus: BatteryReadinessStatus;
}): boolean {
  const os = params.platformOs ?? Platform.OS;
  return (
    os === "android" &&
    params.hasOemSettings &&
    !params.oemGuidanceAcknowledged &&
    (params.batteryStatus === "restricted" || params.batteryStatus === "unknown")
  );
}

export function locationActionLabel(action: Exclude<LocationReadinessAction, null>): string {
  switch (action) {
    case "foreground":
      return "Autoriser la localisation";
    case "enable_precise":
      return "Activer la position précise";
    case "verify_accuracy":
      return "Vérifier la précision";
    case "background":
      return "Autoriser toujours";
  }
}

export function batteryActionLabel(status: BatteryReadinessStatus): string | null {
  if (status === "restricted") return "Batterie";
  if (status === "unknown") return "Vérifier la batterie";
  return null;
}

/** Ignore un résultat de refresh obsolète (concurrence mount / AppState / actions). */
export function shouldApplyRefreshSequence(sequence: number, latest: number): boolean {
  return sequence === latest;
}
