export type FleetMapLatLng = { latitude: number; longitude: number };

type FleetMissionLifecyclePhase =
  | "assigned"
  | "en_route_pickup"
  | "patient_on_board"
  | "arrived"
  | "delayed";

export type FleetRouteLegId = "to_pickup" | "to_dropoff";

export type FleetRoutedPathsMap = Map<string, FleetMapLatLng[]>;
export type FleetRoutedStatesMap = Map<string, FleetRoutedPathState>;

export function fleetRoutePathKey(missionId: number, leg?: FleetRouteLegId): string {
  return leg ? `${missionId}:${leg}` : String(missionId);
}

export function readFleetRoutedPath(
  missionId: number,
  leg: FleetRouteLegId | undefined,
  paths: ReadonlyMap<string, FleetMapLatLng[]> | null | undefined
): FleetMapLatLng[] | undefined {
  if (!paths) return undefined;
  return paths.get(fleetRoutePathKey(missionId, leg));
}

export type FleetDirectionsPlan = {
  origin: FleetMapLatLng;
  destination: FleetMapLatLng;
  waypoints?: FleetMapLatLng[];
};

type DirectionsPlanInput = {
  driverPosition: FleetMapLatLng | null;
  pickup: FleetMapLatLng | null;
  dropoff: FleetMapLatLng | null;
  lifecyclePhase: FleetMissionLifecyclePhase;
};

const DIRECTIONS_CACHE_TTL_MS = 5 * 60 * 1000;
const STABLE_FETCH_COORD_DECIMALS = 3;
const directionsPathCache = new Map<string, { points: FleetMapLatLng[]; atMs: number }>();

function formatCoordKey(point: FleetMapLatLng, decimals = 4): string {
  return `${point.latitude.toFixed(decimals)},${point.longitude.toFixed(decimals)}`;
}

export function quantizeFleetMapCoord(point: FleetMapLatLng, decimals = STABLE_FETCH_COORD_DECIMALS): FleetMapLatLng {
  const factor = 10 ** decimals;
  return {
    latitude: Math.round(point.latitude * factor) / factor,
    longitude: Math.round(point.longitude * factor) / factor,
  };
}

/** Plan stabilisé pour éviter un refetch Directions à chaque tick GPS (~80 m). */
export function buildStableDirectionsFetchPlan(plan: FleetDirectionsPlan): FleetDirectionsPlan {
  return {
    origin: quantizeFleetMapCoord(plan.origin),
    destination: quantizeFleetMapCoord(plan.destination),
    waypoints: plan.waypoints?.map((waypoint) => quantizeFleetMapCoord(waypoint)),
  };
}

export function buildFleetDirectionsCacheKey(
  plan: FleetDirectionsPlan,
  options?: { stable?: boolean }
): string {
  const resolved = options?.stable ? buildStableDirectionsFetchPlan(plan) : plan;
  const waypointKey = (resolved.waypoints ?? []).map((waypoint) => formatCoordKey(waypoint)).join("|");
  return `${formatCoordKey(resolved.origin)}>${waypointKey}>${formatCoordKey(resolved.destination)}`;
}

export function buildMissionDirectionsPlanSignature(
  overlays: Array<{
    missionId: number;
    directionsPlan: FleetDirectionsPlan | null;
    legDirectionsPlans?: Partial<Record<FleetRouteLegId, FleetDirectionsPlan>>;
  }>
): string {
  return overlays
    .map((overlay) => {
      if (overlay.legDirectionsPlans) {
        const legParts = (["to_pickup", "to_dropoff"] as const)
          .map((leg) => {
            const plan = overlay.legDirectionsPlans?.[leg];
            if (!plan) return `${leg}:none`;
            return `${leg}:${buildFleetDirectionsCacheKey(plan, { stable: true })}`;
          })
          .join("+");
        return `${overlay.missionId}:${legParts}`;
      }
      if (!overlay.directionsPlan) return `${overlay.missionId}:none`;
      return `${overlay.missionId}:${buildFleetDirectionsCacheKey(overlay.directionsPlan, { stable: true })}`;
    })
    .join("|");
}

export function haversineMeters(a: FleetMapLatLng, b: FleetMapLatLng): number {
  const toRad = (deg: number) => (deg * Math.PI) / 180;
  const dLat = toRad(b.latitude - a.latitude);
  const dLng = toRad(b.longitude - a.longitude);
  const lat1 = toRad(a.latitude);
  const lat2 = toRad(b.latitude);
  const sinLat = Math.sin(dLat / 2);
  const sinLng = Math.sin(dLng / 2);
  const h = sinLat * sinLat + Math.cos(lat1) * Math.cos(lat2) * sinLng * sinLng;
  return 6371000 * 2 * Math.atan2(Math.sqrt(h), Math.sqrt(1 - h));
}

export function dedupeFleetDirectionsPoints(
  points: FleetMapLatLng[],
  minGapMeters = 4
): FleetMapLatLng[] {
  if (points.length < 2) return points;
  const deduped: FleetMapLatLng[] = [points[0]];
  for (let index = 1; index < points.length; index += 1) {
    const point = points[index];
    const previous = deduped[deduped.length - 1];
    if (haversineMeters(previous, point) >= minGapMeters) {
      deduped.push(point);
    }
  }
  const last = points[points.length - 1];
  const tail = deduped[deduped.length - 1];
  if (tail.latitude !== last.latitude || tail.longitude !== last.longitude) {
    deduped.push(last);
  }
  return deduped;
}

export function densifyFleetDirectionsPath(
  points: FleetMapLatLng[],
  maxSegmentMeters = 18
): FleetMapLatLng[] {
  if (points.length < 2) return points;
  const dense: FleetMapLatLng[] = [points[0]];
  for (let index = 1; index < points.length; index += 1) {
    const start = points[index - 1];
    const end = points[index];
    const distance = haversineMeters(start, end);
    const segments = Math.max(0, Math.floor(distance / maxSegmentMeters));
    for (let step = 1; step <= segments; step += 1) {
      const ratio = step / (segments + 1);
      dense.push({
        latitude: start.latitude + (end.latitude - start.latitude) * ratio,
        longitude: start.longitude + (end.longitude - start.longitude) * ratio,
      });
    }
    dense.push(end);
  }
  return dedupeFleetDirectionsPoints(dense, 2);
}

export function refineFleetDirectionsPath(points: FleetMapLatLng[]): FleetMapLatLng[] {
  const deduped = dedupeFleetDirectionsPoints(points);
  if (deduped.length < 2) return deduped;
  if (deduped.length >= 120) return deduped;
  return densifyFleetDirectionsPath(deduped);
}

/** Simplifie un tracé pour react-native-maps (évite le rendu « blob » sur mobile). */
export function simplifyFleetDirectionsPathForNative(
  points: FleetMapLatLng[],
  options?: { minGapMeters?: number; maxPoints?: number }
): FleetMapLatLng[] {
  if (points.length < 2) return points;
  const maxPoints = options?.maxPoints ?? 48;
  let gap = options?.minGapMeters ?? 16;
  let simplified = dedupeFleetDirectionsPoints(points, gap);
  while (simplified.length > maxPoints && gap < 96) {
    gap += 10;
    simplified = dedupeFleetDirectionsPoints(points, gap);
  }
  return simplified;
}

/** Raccorde le tracé routier à la position live du chauffeur sans refetch Directions. */
export function connectFleetRouteToLiveDriver(
  routed: FleetMapLatLng[],
  driverPosition: FleetMapLatLng | null
): FleetMapLatLng[] {
  if (!driverPosition || routed.length < 2) return routed;
  const start = routed[0];
  if (haversineMeters(driverPosition, start) < 6) {
    return [driverPosition, ...routed.slice(1)];
  }
  return [driverPosition, ...routed];
}

export function readCachedFleetDirectionsPath(plan: FleetDirectionsPlan): FleetMapLatLng[] | null {
  const stablePlan = buildStableDirectionsFetchPlan(plan);
  const cached = directionsPathCache.get(buildFleetDirectionsCacheKey(stablePlan));
  if (!cached) return null;
  if (Date.now() - cached.atMs > DIRECTIONS_CACHE_TTL_MS) {
    directionsPathCache.delete(buildFleetDirectionsCacheKey(stablePlan));
    return null;
  }
  return cached.points;
}

export function writeCachedFleetDirectionsPath(
  plan: FleetDirectionsPlan,
  points: FleetMapLatLng[]
): void {
  const stablePlan = buildStableDirectionsFetchPlan(plan);
  directionsPathCache.set(buildFleetDirectionsCacheKey(stablePlan), {
    points,
    atMs: Date.now(),
  });
}

/** Helper test-only : vide le cache mémoire des trajets pré-calculés. */
export function resetFleetDirectionsCacheForTests(): void {
  directionsPathCache.clear();
}

/** Segment mis en avant quand la mission est sélectionnée sur la carte. */
export function resolveFleetMissionRouteFocusLeg(
  lifecyclePhase: FleetMissionLifecyclePhase
): FleetRouteLegId {
  if (lifecyclePhase === "patient_on_board" || lifecyclePhase === "arrived") {
    return "to_dropoff";
  }
  return "to_pickup";
}

export function resolveFleetMissionLegPlans(
  input: DirectionsPlanInput & { isSelected: boolean }
): {
  focusLeg: FleetRouteLegId;
  legs: Partial<Record<FleetRouteLegId, FleetDirectionsPlan>>;
  combined: FleetDirectionsPlan | null;
} {
  const focusLeg = resolveFleetMissionRouteFocusLeg(input.lifecyclePhase);
  const combined = resolveFleetMissionDirectionsPlan(input);

  if (!input.isSelected) {
    return { focusLeg, legs: {}, combined };
  }

  const legs: Partial<Record<FleetRouteLegId, FleetDirectionsPlan>> = {};
  const { driverPosition: driver, pickup, dropoff } = input;
  if (driver && pickup) {
    legs.to_pickup = { origin: driver, destination: pickup };
  }
  if (pickup && dropoff) {
    legs.to_dropoff = { origin: pickup, destination: dropoff };
  }

  return {
    focusLeg,
    legs,
    combined: legs[focusLeg] ?? combined,
  };
}

/** Plan Directions API : itinéraire routier (pas de segments à vol d'oiseau). */
export function resolveFleetMissionDirectionsPlan(
  input: DirectionsPlanInput
): FleetDirectionsPlan | null {
  const { driverPosition: driver, pickup, dropoff, lifecyclePhase } = input;

  if (lifecyclePhase === "patient_on_board" || lifecyclePhase === "arrived") {
    if (driver && dropoff) {
      return { origin: driver, destination: dropoff };
    }
    if (pickup && dropoff) {
      return { origin: pickup, destination: dropoff };
    }
    return null;
  }

  if (driver && pickup && dropoff) {
    return { origin: driver, destination: dropoff, waypoints: [pickup] };
  }
  if (driver && pickup) {
    return { origin: driver, destination: pickup };
  }
  if (driver && dropoff) {
    return { origin: driver, destination: dropoff };
  }
  if (pickup && dropoff) {
    return { origin: pickup, destination: dropoff };
  }
  return null;
}

export type FleetRoutedPathState = "loading" | "ready" | "failed";

export function resolveFleetOverlayRouteDrawPoints(
  overlay: {
    missionId: number;
    points: FleetMapLatLng[];
    directionsPlan: FleetDirectionsPlan | null;
    legDirectionsPlans?: Partial<Record<FleetRouteLegId, FleetDirectionsPlan>>;
    driverPosition?: FleetMapLatLng | null;
    pickup?: FleetMapLatLng | null;
    dropoff?: FleetMapLatLng | null;
  },
  routedPathsByKey: ReadonlyMap<string, FleetMapLatLng[]> | null | undefined,
  routedStateByKey?: ReadonlyMap<string, FleetRoutedPathState> | null,
  leg?: FleetRouteLegId
): FleetMapLatLng[] {
  const routed = readFleetRoutedPath(overlay.missionId, leg, routedPathsByKey);
  if (routed && routed.length >= 2) {
    const connected =
      leg === "to_pickup"
        ? connectFleetRouteToLiveDriver(routed, overlay.driverPosition ?? null)
        : routed;
    return connected;
  }

  const legPlan = leg ? overlay.legDirectionsPlans?.[leg] : null;
  const plan = legPlan ?? (leg ? null : overlay.directionsPlan);
  const pathKey = fleetRoutePathKey(overlay.missionId, leg);
  if (plan) {
    const state = routedStateByKey?.get(pathKey);
    if (state === "failed") {
      return [plan.origin, plan.destination];
    }
    return [];
  }

  return overlay.points;
}

export function resolveFleetOverlayRoutePoints(
  overlay: {
    missionId: number;
    points: FleetMapLatLng[];
    directionsPlan: FleetDirectionsPlan | null;
    legDirectionsPlans?: Partial<Record<FleetRouteLegId, FleetDirectionsPlan>>;
    driverPosition?: FleetMapLatLng | null;
    pickup?: FleetMapLatLng | null;
    dropoff?: FleetMapLatLng | null;
  },
  routedPathsByKey: ReadonlyMap<string, FleetMapLatLng[]> | null | undefined,
  routedStateByKey?: ReadonlyMap<string, FleetRoutedPathState> | null,
  leg?: FleetRouteLegId
): FleetMapLatLng[] {
  return resolveFleetOverlayRouteDrawPoints(overlay, routedPathsByKey, routedStateByKey, leg);
}

export function decodeEncodedPolyline(encoded: string): FleetMapLatLng[] {
  const points: FleetMapLatLng[] = [];
  let index = 0;
  let lat = 0;
  let lng = 0;

  while (index < encoded.length) {
    let shift = 0;
    let result = 0;
    let byte = 0;
    do {
      byte = encoded.charCodeAt(index++) - 63;
      result |= (byte & 0x1f) << shift;
      shift += 5;
    } while (byte >= 0x20);
    lat += (result & 1) !== 0 ? ~(result >> 1) : result >> 1;

    shift = 0;
    result = 0;
    do {
      byte = encoded.charCodeAt(index++) - 63;
      result |= (byte & 0x1f) << shift;
      shift += 5;
    } while (byte >= 0x20);
    lng += (result & 1) !== 0 ? ~(result >> 1) : result >> 1;

    points.push({ latitude: lat / 1e5, longitude: lng / 1e5 });
  }

  return points;
}

export function resolveFleetMissionEtaBadgeOverlay<T extends { missionId: number; showEtaBadge: boolean }>(
  overlays: T[],
  selectedMissionId: number | null | undefined
): T | null {
  return (
    overlays.find((overlay) => overlay.showEtaBadge && overlay.missionId === selectedMissionId) ??
    overlays.find((overlay) => overlay.showEtaBadge) ??
    null
  );
}

export function resolveFleetMissionEtaAnchor(
  overlay: {
    missionId: number;
    points: FleetMapLatLng[];
    directionsPlan: FleetDirectionsPlan | null;
    legDirectionsPlans?: Partial<Record<FleetRouteLegId, FleetDirectionsPlan>>;
    driverPosition?: FleetMapLatLng | null;
    pickup?: FleetMapLatLng | null;
    dropoff?: FleetMapLatLng | null;
  },
  routedPathsByKey: ReadonlyMap<string, FleetMapLatLng[]> | null | undefined,
  routedStateByKey: ReadonlyMap<string, FleetRoutedPathState> | null | undefined,
  stableEtaAnchors?: ReadonlyMap<number, FleetMapLatLng> | null
): FleetMapLatLng | null {
  const stable = stableEtaAnchors?.get(overlay.missionId);
  if (stable) return stable;

  const routePoints = resolveFleetOverlayRouteDrawPoints(
    overlay,
    routedPathsByKey,
    routedStateByKey
  );
  if (routePoints.length < 2) return null;
  return routePoints[Math.floor(routePoints.length / 2)] ?? routePoints[0];
}
