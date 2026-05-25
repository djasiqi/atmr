import AsyncStorage from "@react-native-async-storage/async-storage";
import { Platform } from "react-native";
import { apiClient } from "../../../core/api/client";
import {
  resolveGoogleMapsNativeApiKey,
  resolveGoogleMapsWebApiKey,
} from "../../../config/googleMapsKeys";
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

function resolveGeocodeApiKey(): string | undefined {
  if (Platform.OS === "web") return resolveGoogleMapsWebApiKey();
  return resolveGoogleMapsNativeApiKey();
}

async function geocodeMissionAddressViaGoogleClient(address: string): Promise<MissionCoord | null> {
  const apiKey = resolveGeocodeApiKey();
  if (!apiKey) return null;

  const params = new URLSearchParams({
    address: address.trim(),
    region: "ch",
    key: apiKey,
  });

  try {
    const response = await fetch(
      `https://maps.googleapis.com/maps/api/geocode/json?${params.toString()}`
    );
    const data = (await response.json()) as {
      status?: string;
      results?: Array<{ geometry?: { location?: { lat?: number; lng?: number } } }>;
    };
    if (data.status !== "OK") return null;
    const loc = data.results?.[0]?.geometry?.location;
    const lat = Number(loc?.lat);
    const lng = Number(loc?.lng);
    if (!Number.isFinite(lat) || !Number.isFinite(lng)) return null;
    return { lat, lng };
  } catch {
    return null;
  }
}

/** Géocode une adresse mission (API backend, repli Google client). */
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
    // repli client ci-dessous
  }

  const clientCoord = await geocodeMissionAddressViaGoogleClient(trimmed);
  if (clientCoord) {
    writePersistentGeocode(cacheKey, clientCoord, now);
  }
  return clientCoord;
}
