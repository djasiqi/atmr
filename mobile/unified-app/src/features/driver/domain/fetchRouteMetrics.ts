import { Platform } from "react-native";
import {
  resolveGoogleMapsNativeApiKey,
  resolveGoogleMapsWebApiKey,
} from "../../../config/googleMapsKeys";
import type { MissionCoord, MissionRouteMetrics } from "./missionRouteMetrics";

const CACHE_TTL_MS = 5 * 60 * 1000;
const routeCache = new Map<string, { metrics: MissionRouteMetrics; atMs: number }>();

function resolveDirectionsApiKey(): string | undefined {
  if (Platform.OS === "web") return resolveGoogleMapsWebApiKey();
  return resolveGoogleMapsNativeApiKey();
}

function buildCacheKey(origin: string, destination: string): string {
  return `${origin}|${destination}`;
}

function coordToParam(coord: MissionCoord): string {
  return `${coord.lat},${coord.lng}`;
}

function addressToParam(address: string | null | undefined): string | null {
  const t = address?.trim();
  return t && t.length > 0 ? t : null;
}

export async function fetchRouteMetrics(options: {
  origin?: MissionCoord | null;
  destination?: MissionCoord | null;
  originAddress?: string | null;
  destinationAddress?: string | null;
}): Promise<MissionRouteMetrics | null> {
  const origin =
    options.origin != null
      ? coordToParam(options.origin)
      : addressToParam(options.originAddress);
  const destination =
    options.destination != null
      ? coordToParam(options.destination)
      : addressToParam(options.destinationAddress);

  if (!origin || !destination) return null;

  const cacheKey = buildCacheKey(origin, destination);
  const cached = routeCache.get(cacheKey);
  const now = Date.now();
  if (cached && now - cached.atMs < CACHE_TTL_MS) {
    return cached.metrics;
  }

  const apiKey = resolveDirectionsApiKey();
  if (!apiKey) return null;

  const params = new URLSearchParams({
    origin,
    destination,
    mode: "driving",
    region: "ch",
    key: apiKey,
  });

  try {
    const response = await fetch(
      `https://maps.googleapis.com/maps/api/directions/json?${params.toString()}`
    );
    const data = (await response.json()) as {
      status?: string;
      routes?: Array<{
        legs?: Array<{
          distance?: { value?: number };
          duration?: { value?: number };
          duration_in_traffic?: { value?: number };
        }>;
      }>;
    };
    if (data.status !== "OK") return null;

    const legs = data.routes?.[0]?.legs ?? [];
    if (legs.length === 0) return null;

    const totals = legs.reduce(
      (acc, leg) => {
        const traffic = Number(leg.duration_in_traffic?.value);
        const normal = Number(leg.duration?.value);
        const seconds = Number.isFinite(traffic) && traffic > 0 ? traffic : normal;
        const meters = Number(leg.distance?.value);
        return {
          seconds: acc.seconds + (Number.isFinite(seconds) && seconds > 0 ? seconds : 0),
          meters: acc.meters + (Number.isFinite(meters) && meters > 0 ? meters : 0),
        };
      },
      { seconds: 0, meters: 0 }
    );

    if (totals.meters <= 0 && totals.seconds <= 0) return null;

    const metrics: MissionRouteMetrics = {
      distanceKm: totals.meters > 0 ? totals.meters / 1000 : null,
      durationMinutes:
        totals.seconds > 0 ? Math.max(1, Math.round(totals.seconds / 60)) : null,
    };

    routeCache.set(cacheKey, { metrics, atMs: now });
    return metrics;
  } catch {
    return null;
  }
}

/** Trajet planifié prise en charge → destination. */
export async function fetchPickupDropoffRouteMetrics(options: {
  pickup?: MissionCoord | null;
  dropoff?: MissionCoord | null;
  pickupAddress?: string | null;
  dropoffAddress?: string | null;
}): Promise<MissionRouteMetrics | null> {
  return fetchRouteMetrics({
    origin: options.pickup,
    destination: options.dropoff,
    originAddress: options.pickupAddress,
    destinationAddress: options.dropoffAddress,
  });
}
