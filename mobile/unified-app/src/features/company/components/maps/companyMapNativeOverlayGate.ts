import { useEffect, useRef, useState } from "react";
import { Platform } from "react-native";

/**
 * iOS New Architecture + react-native-maps : montage markers/polylines pendant
 * `connecting` ou démontage/remontage sur `reconnecting` provoque
 * NSInvalidArgumentException (Sentry finalizeUpdates / AIRGoogleMap).
 */
const isIosOverlayGate = (): boolean => Platform.OS === "ios";

/** Délai après socket `healthy` avant le premier montage des overlays natifs. */
export const IOS_MAP_OVERLAY_STABILIZE_MS = 900;

export function normalizeCompanyTransportStatus(transportStatus: string): string {
  return transportStatus.toLowerCase().trim();
}

export function isCompanyTransportStableForMapOverlays(transportStatus: string): boolean {
  return normalizeCompanyTransportStatus(transportStatus) === "healthy";
}

/**
 * iOS : une fois les overlays affichés, on les conserve pendant `reconnecting`
 * pour éviter un cycle mount/unmount qui fait crasher l'interop legacy.
 */
export function shouldHoldMapOverlaysDuringReconnect(
  transportStatus: string,
  overlaysWereEnabled: boolean
): boolean {
  if (!isIosOverlayGate() || !overlaysWereEnabled) return false;
  return normalizeCompanyTransportStatus(transportStatus) === "reconnecting";
}

export function shouldDisableMapOverlays(
  transportStatus: string,
  overlaysWereEnabled: boolean
): boolean {
  if (isCompanyTransportStableForMapOverlays(transportStatus)) return false;
  if (shouldHoldMapOverlaysDuringReconnect(transportStatus, overlaysWereEnabled)) return false;
  return true;
}

/**
 * Retourne false sur iOS tant que le transport n'est pas stable (+ court délai).
 * Android / web : toujours true.
 */
export function useCompanyMapNativeOverlayGate(transportStatus: string): boolean {
  const [enabled, setEnabled] = useState(() => !isIosOverlayGate());
  const enabledRef = useRef(enabled);
  enabledRef.current = enabled;

  useEffect(() => {
    if (!isIosOverlayGate()) {
      setEnabled(true);
      return;
    }

    if (isCompanyTransportStableForMapOverlays(transportStatus)) {
      const timer = setTimeout(() => setEnabled(true), IOS_MAP_OVERLAY_STABILIZE_MS);
      return () => clearTimeout(timer);
    }

    if (shouldHoldMapOverlaysDuringReconnect(transportStatus, enabledRef.current)) {
      return;
    }

    setEnabled(false);
  }, [transportStatus]);

  return enabled;
}
