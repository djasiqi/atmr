import AsyncStorage from "@react-native-async-storage/async-storage";
import { apiClient } from "../../../core/api/client";
import type { MissionCoord } from "./missionRouteMetrics";

const CACHE_TTL_MS = 30 * 60 * 1000;
const PERSISTENT_GEOCODE_PREFIX = "mission_geocode_v1:";
const geocodeCache = new Map<string, { coord: MissionCoord; atMs: number }>();

type PersistedGeocodeEntry = { coord: MissionCoord; atMs: number };

async function readPersistentGeocode(cacheKey: string): Promise<MissionCoord | null> {
  try {
    const raw = await AsyncStorage.getItem(`${PERSISTENT_GEOCODE_PREFIX}${cacheKey}`);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as PersistedGeocodeEntry;
    if (!parsed?.coord || Date.now() - parsed.atMs >= CACHE_TTL_MS) {
      void AsyncStorage.removeItem(`${PERSISTENT_GEOCODE_PREFIX}${cacheKey}`);
      return null;
    }
    geocodeCache.set(cacheKey, { coord: parsed.coord, atMs: parsed.atMs });
    return parsed.coord;
  } catch {
    return null;
  }
}

function writePersistentGeocode(cacheKey: string, coord: MissionCoord, atMs: number): void {
  const payload: PersistedGeocodeEntry = { coord, atMs };
  geocodeCache.set(cacheKey, { coord, atMs });
  void AsyncStorage.setItem(
    `${PERSISTENT_GEOCODE_PREFIX}${cacheKey}`,
    JSON.stringify(payload)
  ).catch(() => undefined);
}

/**
 * Géocode une adresse mission uniquement via le backend.
 * L'app ne doit jamais appeler Google Geocoding elle-même.
 */
export async function geocodeMissionAddress(address: string): Promise<MissionCoord | null> {
  const trimmed = address.trim();
  if (!trimmed) return null;

  const cacheKey = trimmed.toLowerCase();
  const now = Date.now();
  const cached = geocodeCache.get(cacheKey);
  if (cached && now - cached.atMs < CACHE_TTL_MS) {
    return cached.coord;
  }
  const persisted = await readPersistentGeocode(cacheKey);
  if (persisted) return persisted;

  try {
    const { data } = await apiClient.get("/geocode/geocode", {
      params: { address: trimmed, country: "CH" },
    });
    const payload = (data ?? {}) as Record<string, unknown>;
    const lat = Number(payload.lat);
    const lng = Number(payload.lon ?? payload.lng);
    if (Number.isFinite(lat) && Number.isFinite(lng)) {
      const coord = { lat, lng };
      writePersistentGeocode(cacheKey, coord, now);
      return coord;
    }
  } catch {
    return null;
  }

  return null;
}
