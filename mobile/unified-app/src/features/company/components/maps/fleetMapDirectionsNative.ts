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
import { apiClient } from "../../../../core/api/client";
import { emitDriverTelemetry } from "../../../../core/observability/driverTelemetry";

// Le mobile n'appelle plus directement Google Directions : la clé est restée côté serveur,
// le proxy /api/v1/directions absorbe les renders répétés via cache Redis et renvoie la
// polyline encodée.
type DirectionsProxyResponse = {
  status?: string;
  overview_polyline?: string | null;
  cached?: boolean;
  source?: string;
  error_message?: string;
  http_status?: number;
};

function extractPathFromEncodedPolyline(encoded: string | null | undefined): FleetMapLatLng[] {
  if (!encoded) return [];
  return simplifyFleetDirectionsPathForNative(decodeEncodedPolyline(encoded));
}

export async function fetchFleetDirectionsPathNative(
  plan: FleetDirectionsPlan,
  _apiKey?: string
): Promise<FleetMapLatLng[]> {
  void _apiKey;
  const stablePlan = buildStableDirectionsFetchPlan(plan);
  const cached = readCachedFleetDirectionsPath(stablePlan);
  if (cached && cached.length >= 2) return cached;

  const body: Record<string, unknown> = {
    origin: {
      latitude: stablePlan.origin.latitude,
      longitude: stablePlan.origin.longitude,
    },
    destination: {
      latitude: stablePlan.destination.latitude,
      longitude: stablePlan.destination.longitude,
    },
    mode: "driving",
    region: "ch",
  };
  if (stablePlan.waypoints && stablePlan.waypoints.length > 0) {
    body.waypoints = stablePlan.waypoints.map((waypoint) => ({
      latitude: waypoint.latitude,
      longitude: waypoint.longitude,
    }));
  }

  try {
    const response = await apiClient.post<DirectionsProxyResponse>("/directions", body);
    const data = response.data ?? {};
    if (data.status !== "OK" || !data.overview_polyline) {
      emitDriverTelemetry("company.fleet.directions.failed", {
        source: "company.fleetMapDirectionsNative",
        status: data.status ?? "unknown",
        error_message: data.error_message ?? null,
        http_status: data.http_status ?? response.status ?? null,
        cached: Boolean(data.cached),
        has_waypoints: Boolean(stablePlan.waypoints && stablePlan.waypoints.length > 0),
      });
      return [];
    }

    const points = dedupeFleetDirectionsPoints(
      extractPathFromEncodedPolyline(data.overview_polyline),
      8
    );
    if (points.length >= 2) {
      writeCachedFleetDirectionsPath(stablePlan, points);
    }
    return points;
  } catch (error) {
    emitDriverTelemetry("company.fleet.directions.exception", {
      source: "company.fleetMapDirectionsNative",
      error: error instanceof Error ? error.message : String(error),
      has_waypoints: Boolean(stablePlan.waypoints && stablePlan.waypoints.length > 0),
    });
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
  apiKey?: string
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
