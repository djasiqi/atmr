import { Platform } from "react-native";
import type { LatLng } from "react-native-maps";

import { simplifyFleetDirectionsPathForNative, type FleetMapLatLng } from "./fleetMapDirections";
import type { FleetMissionRouteStyle } from "./fleetMapMissionVisual";

/** Épaisseurs adaptées à react-native-maps (≠ Google Maps JS web). */
export const FLEET_NATIVE_ROUTE_RENDER = {
  maxMainStrokePx: 4,
  maxGlowStrokePx: 8,
  minMainStrokePx: 2.5,
  glowExtraPx: 4,
  /** Halo discret — l’alpha principal est porté par `glowColor` (spec Operational Calm). */
  glowOpacityScale: 1,
  maxMainLineOpacity: 0.85,
} as const;

export function isNativeFleetMapPlatform(): boolean {
  return Platform.OS === "ios" || Platform.OS === "android";
}

export function isAndroidFleetMapPlatform(): boolean {
  return Platform.OS === "android";
}

function resolveNativeRouteRenderTuning() {
  if (isAndroidFleetMapPlatform()) {
    // Android Google renderer appears visually thicker than iOS/web for identical values.
    return {
      maxMainStrokePx: 2.6,
      maxGlowStrokePx: 3.8,
      minMainStrokePx: 1.4,
      glowExtraPx: 1.2,
    };
  }
  return {
    maxMainStrokePx: FLEET_NATIVE_ROUTE_RENDER.maxMainStrokePx,
    maxGlowStrokePx: FLEET_NATIVE_ROUTE_RENDER.maxGlowStrokePx,
    minMainStrokePx: FLEET_NATIVE_ROUTE_RENDER.minMainStrokePx,
    glowExtraPx: FLEET_NATIVE_ROUTE_RENDER.glowExtraPx,
  };
}

export function resolveNativeMissionRouteStrokes(routeStyle: FleetMissionRouteStyle): {
  mainStroke: number;
  glowStroke: number;
} {
  const tuning = resolveNativeRouteRenderTuning();
  const mainStroke = Math.min(
    tuning.maxMainStrokePx,
    Math.max(tuning.minMainStrokePx, routeStyle.strokeWidth)
  );
  const glowStroke = Math.min(
    tuning.maxGlowStrokePx,
    Math.max(mainStroke + tuning.glowExtraPx, routeStyle.glowWidth)
  );
  return { mainStroke, glowStroke };
}

export function prepareFleetRouteCoordsForNativeRender(coords: LatLng[]): LatLng[] {
  if (!isNativeFleetMapPlatform() || coords.length < 2) return coords;
  return simplifyFleetDirectionsPathForNative(coords as FleetMapLatLng[]).map((point) => ({
    latitude: point.latitude,
    longitude: point.longitude,
  }));
}
