/** Projection géographique légère (sans dépendance externe). */

export function projectCoordinate(
  latitude: number,
  longitude: number,
  headingDeg: number,
  distanceKm: number
): { latitude: number; longitude: number } {
  const rad = (headingDeg * Math.PI) / 180;
  const latRad = (latitude * Math.PI) / 180;
  const deltaLat = (distanceKm / 6371) * Math.cos(rad);
  const deltaLng = (distanceKm / (6371 * Math.cos(latRad || 1e-6))) * Math.sin(rad);
  return {
    latitude: latitude + (deltaLat * 180) / Math.PI,
    longitude: longitude + (deltaLng * 180) / Math.PI,
  };
}

export type FleetHeatmapPoint = {
  latitude: number;
  longitude: number;
  weight: number;
};

/** Points pondérés pour heatmap retards (données live réelles). */
export function buildFleetDelayHeatmapPoints(
  drivers: { latitude: number; longitude: number; enrichment: { operationalStatus: string; delayMinutes: number | null } }[]
): FleetHeatmapPoint[] {
  return drivers
    .filter(
      (d) =>
        d.enrichment.operationalStatus === "delayed" || d.enrichment.operationalStatus === "incident"
    )
    .map((d) => ({
      latitude: d.latitude,
      longitude: d.longitude,
      weight: Math.min(12, Math.max(2, d.enrichment.delayMinutes ?? 5)),
    }));
}
