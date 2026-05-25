import {
  buildStableDirectionsFetchPlan,
  dedupeFleetDirectionsPoints,
  fleetRoutePathKey,
  readCachedFleetDirectionsPath,
  refineFleetDirectionsPath,
  writeCachedFleetDirectionsPath,
  type FleetDirectionsPlan,
  type FleetMapLatLng,
  type FleetRouteLegId,
} from "./fleetMapDirections";

type GoogleLatLng = { lat: () => number; lng: () => number };

type DirectionsRouteLeg = {
  steps?: Array<{
    path?: GoogleLatLng[];
  }>;
};

type DirectionsRouteResult = {
  routes?: Array<{
    overview_path?: GoogleLatLng[];
    legs?: DirectionsRouteLeg[];
  }>;
};

function mapGoogleLatLng(point: GoogleLatLng): FleetMapLatLng {
  return { latitude: point.lat(), longitude: point.lng() };
}

type DirectionsRoute = {
  overview_path?: GoogleLatLng[];
  legs?: DirectionsRouteLeg[];
};

export function extractDetailedDirectionsPath(route: DirectionsRoute | undefined): FleetMapLatLng[] {
  const detailed: FleetMapLatLng[] = [];
  for (const leg of route?.legs ?? []) {
    for (const step of leg.steps ?? []) {
      for (const point of step.path ?? []) {
        detailed.push(mapGoogleLatLng(point));
      }
    }
  }

  if (detailed.length >= 2) {
    return refineFleetDirectionsPath(detailed);
  }

  const overview = (route?.overview_path ?? []).map(mapGoogleLatLng);
  if (overview.length < 2) return [];
  return refineFleetDirectionsPath(overview);
}

export async function fetchFleetDirectionsPathWeb(
  gmaps: Record<string, unknown>,
  plan: FleetDirectionsPlan
): Promise<FleetMapLatLng[]> {
  const stablePlan = buildStableDirectionsFetchPlan(plan);
  const cached = readCachedFleetDirectionsPath(stablePlan);
  if (cached && cached.length >= 2) return cached;

  const DirectionsService = gmaps.DirectionsService as new () => {
    route: (
      req: Record<string, unknown>,
      cb: (result: DirectionsRouteResult | null, status: string) => void
    ) => void;
  };

  const service = new DirectionsService();

  return new Promise((resolve) => {
    service.route(
      {
        origin: { lat: stablePlan.origin.latitude, lng: stablePlan.origin.longitude },
        destination: {
          lat: stablePlan.destination.latitude,
          lng: stablePlan.destination.longitude,
        },
        waypoints: (stablePlan.waypoints ?? []).map((waypoint) => ({
          location: { lat: waypoint.latitude, lng: waypoint.longitude },
          stopover: true,
        })),
        travelMode: "DRIVING",
        region: "CH",
        provideRouteAlternatives: false,
      },
      (result, status) => {
        if (status !== "OK" || !result?.routes?.[0]) {
          resolve([]);
          return;
        }

        const points = extractDetailedDirectionsPath(result.routes[0]);
        const refined = dedupeFleetDirectionsPoints(points, 3);

        if (refined.length >= 2) {
          writeCachedFleetDirectionsPath(stablePlan, refined);
        }
        resolve(refined);
      }
    );
  });
}

type OverlayDirectionsInput = {
  missionId: number;
  directionsPlan: FleetDirectionsPlan | null;
  legDirectionsPlans?: Partial<Record<FleetRouteLegId, FleetDirectionsPlan>>;
};

export async function fetchFleetDirectionsPathsForOverlays(
  gmaps: Record<string, unknown>,
  overlays: OverlayDirectionsInput[]
): Promise<Map<string, FleetMapLatLng[]>> {
  const routed = new Map<string, FleetMapLatLng[]>();

  await Promise.all(
    overlays.flatMap((overlay) => {
      if (overlay.legDirectionsPlans) {
        return (["to_pickup", "to_dropoff"] as const).map(async (leg) => {
          const plan = overlay.legDirectionsPlans?.[leg];
          if (!plan) return;
          const path = await fetchFleetDirectionsPathWeb(gmaps, plan);
          if (path.length >= 2) {
            routed.set(fleetRoutePathKey(overlay.missionId, leg), path);
          }
        });
      }
      if (!overlay.directionsPlan) return [];
      return [
        (async () => {
          const path = await fetchFleetDirectionsPathWeb(gmaps, overlay.directionsPlan!);
          if (path.length >= 2) {
            routed.set(fleetRoutePathKey(overlay.missionId), path);
          }
        })(),
      ];
    })
  );

  return routed;
}
