import {
  buildStableDirectionsFetchPlan,
  decodeEncodedPolyline,
  dedupeFleetDirectionsPoints,
  fleetRoutePathKey,
  readCachedFleetDirectionsPath,
  simplifyFleetDirectionsPathForNative,
  writeCachedFleetDirectionsPath,
  type FleetDirectionsPlan,
  type FleetMapLatLng,
  type FleetRouteLegId,
} from "./fleetMapDirections";

type DirectionsJsonStep = {
  polyline?: { points?: string };
};

type DirectionsJsonLeg = {
  steps?: DirectionsJsonStep[];
};

type DirectionsJsonRoute = {
  legs?: DirectionsJsonLeg[];
  overview_polyline?: { points?: string };
};

type DirectionsJsonResponse = {
  status?: string;
  routes?: DirectionsJsonRoute[];
};

function extractNativeDirectionsPath(route: DirectionsJsonRoute | undefined): FleetMapLatLng[] {
  const overview = route?.overview_polyline?.points;
  if (overview) {
    return simplifyFleetDirectionsPathForNative(decodeEncodedPolyline(overview));
  }

  const detailed: FleetMapLatLng[] = [];
  for (const leg of route?.legs ?? []) {
    for (const step of leg.steps ?? []) {
      const encoded = step.polyline?.points;
      if (encoded) detailed.push(...decodeEncodedPolyline(encoded));
    }
  }
  if (detailed.length >= 2) return simplifyFleetDirectionsPathForNative(detailed);
  return [];
}

export async function fetchFleetDirectionsPathNative(
  plan: FleetDirectionsPlan,
  apiKey: string
): Promise<FleetMapLatLng[]> {
  const stablePlan = buildStableDirectionsFetchPlan(plan);
  const cached = readCachedFleetDirectionsPath(stablePlan);
  if (cached && cached.length >= 2) return cached;

  const params = new URLSearchParams({
    origin: `${stablePlan.origin.latitude},${stablePlan.origin.longitude}`,
    destination: `${stablePlan.destination.latitude},${stablePlan.destination.longitude}`,
    mode: "driving",
    region: "ch",
    key: apiKey,
  });

  const waypoints = (stablePlan.waypoints ?? [])
    .map((waypoint) => `${waypoint.latitude},${waypoint.longitude}`)
    .join("|");
  if (waypoints) params.set("waypoints", waypoints);

  try {
    const response = await fetch(
      `https://maps.googleapis.com/maps/api/directions/json?${params.toString()}`
    );
    const data = (await response.json()) as DirectionsJsonResponse;
    if (data.status !== "OK" || !data.routes?.[0]) return [];

    const points = dedupeFleetDirectionsPoints(extractNativeDirectionsPath(data.routes[0]), 8);
    if (points.length >= 2) {
      writeCachedFleetDirectionsPath(stablePlan, points);
    }
    return points;
  } catch {
    return [];
  }
}

type OverlayDirectionsInput = {
  missionId: number;
  directionsPlan: FleetDirectionsPlan | null;
  legDirectionsPlans?: Partial<Record<FleetRouteLegId, FleetDirectionsPlan>>;
};

export async function fetchFleetDirectionsPathsForOverlaysNative(
  overlays: OverlayDirectionsInput[],
  apiKey: string
): Promise<Map<string, FleetMapLatLng[]>> {
  const routed = new Map<string, FleetMapLatLng[]>();

  await Promise.all(
    overlays.flatMap((overlay) => {
      if (overlay.legDirectionsPlans) {
        return (["to_pickup", "to_dropoff"] as const).map(async (leg) => {
          const plan = overlay.legDirectionsPlans?.[leg];
          if (!plan) return;
          const path = await fetchFleetDirectionsPathNative(plan, apiKey);
          if (path.length >= 2) {
            routed.set(fleetRoutePathKey(overlay.missionId, leg), path);
          }
        });
      }
      if (!overlay.directionsPlan) return [];
      return [
        (async () => {
          const path = await fetchFleetDirectionsPathNative(overlay.directionsPlan!, apiKey);
          if (path.length >= 2) {
            routed.set(fleetRoutePathKey(overlay.missionId), path);
          }
        })(),
      ];
    })
  );

  return routed;
}
