import { missionCoordToMapPoint, normalizeMissionMapPoint, type MapLatLng } from "./missionMapCoordUtils";

/** Affichage carte ≠ fraîcheur tracking. Un snapshot API n’est jamais du GNSS. */
export type DriverMapDisplaySource = "gnss" | "api";

export type DriverMapDisplayPosition = {
  coord: MapLatLng | null;
  source: DriverMapDisplaySource | null;
  gnssCoord: MapLatLng | null;
};

export function resolveDriverMapDisplayPosition(
  livePosition: { lat: number; lng: number } | null,
  apiLat?: unknown,
  apiLng?: unknown
): DriverMapDisplayPosition {
  const gnssCoord = missionCoordToMapPoint(livePosition);
  if (gnssCoord) {
    return { coord: gnssCoord, source: "gnss", gnssCoord };
  }
  const apiCoord = normalizeMissionMapPoint(apiLat, apiLng);
  if (apiCoord) {
    return { coord: apiCoord, source: "api", gnssCoord: null };
  }
  return { coord: null, source: null, gnssCoord: null };
}
