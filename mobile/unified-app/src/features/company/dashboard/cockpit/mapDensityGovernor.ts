import { DENSITY_DRIVER_THRESHOLDS } from "./cockpitThresholds";

export type DensityLevel = "low" | "medium" | "high" | "extreme" | "aggregate";

export type MapDensityInput = {
  driverCount: number;
  zoomLatitudeDelta?: number | null;
};

export type MapDensityPolicy = {
  level: DensityLevel;
  hideSecondaryRoutes: boolean;
  simplifyMarkers: boolean;
  reduceGlow: boolean;
  reduceEtaLabels: boolean;
  disableNonCriticalAnimations: boolean;
  aggregateMode: boolean;
};

const ZOOM_LOW_THRESHOLD = 0.35;

export function resolveDensityLevel(driverCount: number, zoomLatitudeDelta?: number | null): DensityLevel {
  let level: DensityLevel = "low";
  if (driverCount > DENSITY_DRIVER_THRESHOLDS.aggregate) level = "aggregate";
  else if (driverCount > DENSITY_DRIVER_THRESHOLDS.extreme) level = "extreme";
  else if (driverCount > DENSITY_DRIVER_THRESHOLDS.high) level = "high";
  else if (driverCount > DENSITY_DRIVER_THRESHOLDS.medium) level = "medium";

  if (zoomLatitudeDelta != null && zoomLatitudeDelta >= ZOOM_LOW_THRESHOLD) {
    if (level === "low") return "medium";
    if (level === "medium") return "high";
  }
  return level;
}

export function computeMapDensityPolicy(input: MapDensityInput): MapDensityPolicy {
  const level = resolveDensityLevel(input.driverCount, input.zoomLatitudeDelta);
  const zoomLow =
    input.zoomLatitudeDelta != null && input.zoomLatitudeDelta >= ZOOM_LOW_THRESHOLD;

  return {
    level,
    hideSecondaryRoutes: level !== "low" || zoomLow,
    simplifyMarkers: level === "high" || level === "extreme" || level === "aggregate" || zoomLow,
    reduceGlow: level !== "low",
    reduceEtaLabels: level === "high" || level === "extreme" || level === "aggregate",
    disableNonCriticalAnimations:
      level === "extreme" || level === "aggregate" || zoomLow,
    aggregateMode: level === "aggregate",
  };
}
