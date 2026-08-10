import { Platform } from "react-native";

/**
 * Délai post-`mapReady` avant le premier montage des marqueurs chauffeurs iOS
 * (stabilisation New Arch MapView, indépendant du Socket).
 */
export const IOS_MAP_DRIVER_MARKERS_SETTLE_MS = 280;

export function isIosNativeMapPlatform(): boolean {
  return Platform.OS === "ios";
}

/**
 * Clustering flotte : désactivé sur iOS pour éviter ClusterMarker data-URI
 * (chemin `image` + URI fragile New Arch). Chauffeurs individuels PNG Metro uniquement.
 */
export function shouldEnableFleetClustering(simplifyClustering: boolean): boolean {
  if (simplifyClustering) return false;
  if (isIosNativeMapPlatform()) return false;
  return true;
}

export function resolveMountDriverMarkers(mapReady: boolean, iosNativeMapSettled: boolean): boolean {
  if (!mapReady) return false;
  if (!isIosNativeMapPlatform()) return true;
  return iosNativeMapSettled;
}

export function resolveMountDynamicOverlays(
  mapReady: boolean,
  nativeOverlaysEnabled: boolean
): boolean {
  return mapReady && nativeOverlaysEnabled;
}
