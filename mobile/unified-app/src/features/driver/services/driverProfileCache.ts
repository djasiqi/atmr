import { STORAGE_KEYS } from "../../../core/storage/storageKeys";
import {
  getItem,
  migrateIfNeeded,
  removeItem,
  setItem,
} from "../../../core/storage/typedStorage";
import type { DriverProfile } from "../api";

const DRIVER_PROFILE_CACHE_SCHEMA_VERSION = 1;
const DRIVER_PROFILE_CACHE_TTL_MS = 24 * 60 * 60 * 1000;

type DriverProfileCacheEnvelope = {
  schemaVersion: number;
  cachedAtMs: number;
  profile: DriverProfile;
};

export type DriverProfileCacheStatus =
  | "hit"
  | "miss"
  | "expired"
  | "invalid"
  | "schema_mismatch"
  | "error";

export type ReadDriverProfileCacheResult = {
  status: DriverProfileCacheStatus;
  profile?: DriverProfile;
};

function isValidEnvelope(value: unknown): value is DriverProfileCacheEnvelope {
  if (!value || typeof value !== "object") return false;
  const envelope = value as Record<string, unknown>;
  return (
    typeof envelope.schemaVersion === "number" &&
    typeof envelope.cachedAtMs === "number" &&
    !!envelope.profile &&
    typeof envelope.profile === "object"
  );
}

export async function readDriverProfileCache(options: {
  allowStale: boolean;
}): Promise<ReadDriverProfileCacheResult> {
  try {
    await migrateIfNeeded();
    const envelope = await getItem<DriverProfileCacheEnvelope>(
      STORAGE_KEYS.DRIVER_PROFILE
    );
    if (!envelope) return { status: "miss" };
    if (!isValidEnvelope(envelope)) return { status: "invalid" };
    if (envelope.schemaVersion !== DRIVER_PROFILE_CACHE_SCHEMA_VERSION) {
      return { status: "schema_mismatch" };
    }
    const ageMs = Date.now() - envelope.cachedAtMs;
    if (ageMs > DRIVER_PROFILE_CACHE_TTL_MS) {
      if (options.allowStale) {
        return { status: "expired", profile: envelope.profile };
      }
      return { status: "expired" };
    }
    return { status: "hit", profile: envelope.profile };
  } catch {
    return { status: "error" };
  }
}

export async function writeDriverProfileCache(
  profile: DriverProfile
): Promise<void> {
  const envelope: DriverProfileCacheEnvelope = {
    schemaVersion: DRIVER_PROFILE_CACHE_SCHEMA_VERSION,
    cachedAtMs: Date.now(),
    profile,
  };
  await setItem(STORAGE_KEYS.DRIVER_PROFILE, envelope);
}

export async function purgeDriverProfileCache(): Promise<void> {
  await removeItem(STORAGE_KEYS.DRIVER_PROFILE);
}

