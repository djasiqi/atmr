import { useEffect, useState } from "react";
import { Platform } from "react-native";

/**
 * iOS New Architecture + react-native-maps : montage markers/polylines pendant
 * `connecting`/`reconnecting` + rafale Directions provoque NSInvalidArgumentException
 * (Sentry LIRIE-MOBILE, finalizeUpdates / AIRGoogleMap).
 */
const IOS_OVERLAY_GATE = Platform.OS === "ios";

/** Délai après socket `healthy` avant overlays natifs (évite la course reconnect). */
export const IOS_MAP_OVERLAY_STABILIZE_MS = 450;

export function isCompanyTransportStableForMapOverlays(transportStatus: string): boolean {
  const normalized = transportStatus.toLowerCase().trim();
  return normalized === "healthy";
}

/**
 * Retourne false sur iOS tant que le transport n'est pas stable (+ court délai).
 * Android / web : toujours true.
 */
export function useCompanyMapNativeOverlayGate(transportStatus: string): boolean {
  const [enabled, setEnabled] = useState(() => !IOS_OVERLAY_GATE);

  useEffect(() => {
    if (!IOS_OVERLAY_GATE) {
      setEnabled(true);
      return;
    }
    if (!isCompanyTransportStableForMapOverlays(transportStatus)) {
      setEnabled(false);
      return;
    }
    const timer = setTimeout(() => setEnabled(true), IOS_MAP_OVERLAY_STABILIZE_MS);
    return () => clearTimeout(timer);
  }, [transportStatus]);

  return enabled;
}
