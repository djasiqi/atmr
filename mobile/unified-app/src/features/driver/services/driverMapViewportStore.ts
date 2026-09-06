import { getItem, setItem } from "../../../core/storage/typedStorage";
import { STORAGE_KEYS } from "../../../core/storage/storageKeys";
import {
  isUsableMapRegion,
  type DriverMapRegion,
} from "../domain/driverMapCameraPolicy";

export type DriverMapViewport = {
  center: { latitude: number; longitude: number };
  latitudeDelta: number;
  longitudeDelta: number;
  saved_at: string;
};

const VIEWPORT_MAX_AGE_MS = 7 * 24 * 60 * 60 * 1000;

let memory: DriverMapViewport | null = null;
let hydratePromise: Promise<DriverMapViewport | null> | null = null;

export function viewportToRegion(viewport: DriverMapViewport): DriverMapRegion {
  return {
    latitude: viewport.center.latitude,
    longitude: viewport.center.longitude,
    latitudeDelta: viewport.latitudeDelta,
    longitudeDelta: viewport.longitudeDelta,
  };
}

export function parseDriverMapViewport(raw: unknown, nowMs = Date.now()): DriverMapViewport | null {
  if (!raw || typeof raw !== "object") return null;
  const rec = raw as Record<string, unknown>;
  const center = rec.center;
  if (!center || typeof center !== "object") return null;
  const c = center as Record<string, unknown>;
  const latitude = typeof c.latitude === "number" ? c.latitude : Number.NaN;
  const longitude = typeof c.longitude === "number" ? c.longitude : Number.NaN;
  const latitudeDelta = typeof rec.latitudeDelta === "number" ? rec.latitudeDelta : Number.NaN;
  const longitudeDelta = typeof rec.longitudeDelta === "number" ? rec.longitudeDelta : Number.NaN;
  const savedAt = typeof rec.saved_at === "string" ? rec.saved_at : "";
  const savedMs = Date.parse(savedAt);
  if (!Number.isFinite(savedMs) || nowMs - savedMs > VIEWPORT_MAX_AGE_MS) return null;
  const region = { latitude, longitude, latitudeDelta, longitudeDelta };
  if (!isUsableMapRegion(region)) return null;
  return {
    center: { latitude, longitude },
    latitudeDelta,
    longitudeDelta,
    saved_at: savedAt,
  };
}

export function peekDriverMapViewport(): DriverMapViewport | null {
  return memory;
}

export async function hydrateDriverMapViewport(): Promise<DriverMapViewport | null> {
  if (hydratePromise) return hydratePromise;
  hydratePromise = (async () => {
    const raw = await getItem<unknown>(STORAGE_KEYS.DRIVER_LAST_MAP_VIEWPORT);
    const parsed = parseDriverMapViewport(raw);
    memory = parsed;
    return parsed;
  })();
  try {
    return await hydratePromise;
  } catch {
    memory = null;
    return null;
  }
}

export async function writeDriverMapViewport(region: DriverMapRegion, nowIso = new Date().toISOString()): Promise<void> {
  if (!isUsableMapRegion(region)) return;
  const next: DriverMapViewport = {
    center: { latitude: region.latitude, longitude: region.longitude },
    latitudeDelta: region.latitudeDelta,
    longitudeDelta: region.longitudeDelta,
    saved_at: nowIso,
  };
  memory = next;
  await setItem(STORAGE_KEYS.DRIVER_LAST_MAP_VIEWPORT, next);
}

export function __resetDriverMapViewportStoreForTests(): void {
  memory = null;
  hydratePromise = null;
}
