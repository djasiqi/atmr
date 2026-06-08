import { computeFleetRegion } from "./fleetMapLogic";

export type MapEdgePadding = {
  top: number;
  right: number;
  bottom: number;
  left: number;
};

/** Padding pour `fitToCoordinates` — évite les insets UI énormes qui laissent trop de carte vide. */
export function computeFleetMapFitEdgePadding(
  cameraInsets: MapEdgePadding,
  options?: { immersive?: boolean }
): MapEdgePadding {
  const immersive = options?.immersive ?? false;
  if (!immersive) {
    return {
      top: Math.round(Math.max(56, cameraInsets.top * 0.5)),
      right: Math.round(Math.max(44, cameraInsets.right * 0.55)),
      bottom: Math.round(Math.max(56, cameraInsets.bottom * 0.5)),
      left: Math.round(Math.max(36, cameraInsets.left * 0.5)),
    };
  }
  // Cockpit : respecter la bande header + KPI (ne pas plafonner trop bas le top).
  return {
    top: Math.round(Math.max(cameraInsets.top, 176)),
    right: Math.round(Math.min(cameraInsets.right, 64)),
    bottom: Math.round(Math.min(cameraInsets.bottom, 220)),
    left: Math.round(Math.min(cameraInsets.left, 40)),
  };
}

export function applyFleetFitVerticalBias(
  padding: MapEdgePadding,
  verticalBias: number
): MapEdgePadding {
  if (verticalBias <= 0) return padding;
  const shift = Math.min(48, Math.round(padding.bottom * verticalBias * 0.35));
  return {
    top: Math.max(48, padding.top - Math.round(shift * 0.35)),
    right: padding.right,
    bottom: padding.bottom + shift,
    left: padding.left,
  };
}

export function buildDriversBoundsSignature(
  drivers: Array<{ driver_id: number; latitude: number; longitude: number }>
): string {
  return drivers
    .map((d) => `${d.driver_id}:${d.latitude.toFixed(5)},${d.longitude.toFixed(5)}`)
    .sort()
    .join("|");
}

/** Signature structurelle (IDs only) — auto-fit sans jitter GPS. */
export function buildDriversStructuralSignature(
  drivers: Array<{ driver_id: number }>
): string {
  return drivers
    .map((d) => String(d.driver_id))
    .sort()
    .join("|");
}

/** Limite le dézoom manuel au-delà du cadre flotte (évite de « descendre » dans le vide). */
export function computeFleetMaxRegionDelta(
  drivers: Array<{ latitude: number; longitude: number }>,
  paddingFactor = 2.1,
  slack = 1.3
): number | undefined {
  if (drivers.length === 0) return undefined;
  const region = computeFleetRegion(drivers, paddingFactor);
  return Math.max(region.latitudeDelta, region.longitudeDelta) * slack;
}
