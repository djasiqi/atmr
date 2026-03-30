/**
 * Cache disque du profil chauffeur (P2) — TTL 24h, schema_version, garde-fous taille / identité.
 */
import AsyncStorage from "@react-native-async-storage/async-storage";
import type { Driver } from "@/services/api";
import { asyncStorage, secureStorage } from "@/services/storage";
import { getLogger } from "@/utils/logger";
import { sendIngestEvent } from "@/src/config/telemetry";

const log = getLogger("DriverProfileCache");

/** TTL cache profil chauffeur — ne pas modifier sans validation produit (nom, téléphone, photo changent rarement). */
export const DRIVER_PROFILE_CACHE_TTL_MS = 24 * 60 * 60 * 1000;

/** Incrémenter si l'enveloppe ou le modèle Driver côté cache change de façon incompatible. */
export const DRIVER_PROFILE_CACHE_SCHEMA_VERSION = 1;

export const DRIVER_PROFILE_CACHE_STORAGE_KEY = "driver_profile_cache_v1";

/** Garde-fou taille sérialisée (préférences / métadonnées futures). */
export const MAX_PROFILE_CACHE_SIZE_BYTES = 256 * 1024;

export type DriverProfileCacheStatus =
  | "hit"
  | "miss"
  | "expired"
  | "driver_mismatch"
  | "schema_mismatch"
  | "invalid"
  | "error"
  | "oversized_skip";

export type DriverProfileCacheEnvelope = {
  schema_version: number;
  profile: Driver;
  driver_id: number;
  company_id: number | null;
  cached_at_ms: number;
};

let writeGeneration = 0;

function emitCacheStatus(status: DriverProfileCacheStatus): void {
  log.debug("driver_profile_cache_status", { status });
  if (__DEV__) {
    try {
      sendIngestEvent({
        event: "driver_profile_cache_status",
        status,
        ts: Date.now(),
      });
    } catch {
      // ignore
    }
  }
}

function isEnvelope(x: unknown): x is DriverProfileCacheEnvelope {
  if (!x || typeof x !== "object") return false;
  const o = x as Record<string, unknown>;
  return (
    typeof o.schema_version === "number" &&
    o.profile !== null &&
    typeof o.profile === "object" &&
    typeof o.driver_id === "number" &&
    (o.company_id === null || typeof o.company_id === "number") &&
    typeof o.cached_at_ms === "number"
  );
}

/**
 * Valide l'enveloppe par rapport à la session courante (AsyncStorage driver_id + SecureStore public_id).
 * Retourne "ok" si cohérent, sinon "driver_mismatch".
 */
export async function validateCachedProfileAgainstSession(
  env: DriverProfileCacheEnvelope
): Promise<"ok" | "driver_mismatch"> {
  const [storedDriverId, userPublicId] = await Promise.all([
    asyncStorage.getDriverId(),
    secureStorage.getUserPublicId(),
  ]);
  const p = env.profile;
  if (userPublicId && p.user?.public_id && userPublicId !== p.user.public_id) {
    return "driver_mismatch";
  }
  if (storedDriverId !== null && storedDriverId !== env.driver_id) {
    return "driver_mismatch";
  }
  if (storedDriverId !== null && storedDriverId !== p.id) {
    return "driver_mismatch";
  }
  return "ok";
}

export type ReadDriverProfileCacheResult = {
  status: DriverProfileCacheStatus;
  profile?: Driver;
};

/**
 * Lecture cache pour fetchDriverProfile — ne fait pas de GET réseau.
 */
export async function readDriverProfileCache(options: {
  allowStale: boolean;
}): Promise<ReadDriverProfileCacheResult> {
  try {
    const raw = await AsyncStorage.getItem(DRIVER_PROFILE_CACHE_STORAGE_KEY);
    if (raw === null) {
      emitCacheStatus("miss");
      return { status: "miss" };
    }
    let parsed: unknown;
    try {
      parsed = JSON.parse(raw);
    } catch {
      emitCacheStatus("invalid");
      return { status: "invalid" };
    }
    if (!isEnvelope(parsed)) {
      emitCacheStatus("invalid");
      return { status: "invalid" };
    }
    if (parsed.schema_version !== DRIVER_PROFILE_CACHE_SCHEMA_VERSION) {
      emitCacheStatus("schema_mismatch");
      return { status: "schema_mismatch" };
    }
    const now = Date.now();
    const age = now - parsed.cached_at_ms;
    const identity = await validateCachedProfileAgainstSession(parsed);
    if (identity !== "ok") {
      emitCacheStatus("driver_mismatch");
      return { status: "driver_mismatch" };
    }
    if (age > DRIVER_PROFILE_CACHE_TTL_MS) {
      if (options.allowStale) {
        emitCacheStatus("expired");
        return { status: "expired", profile: parsed.profile };
      }
      emitCacheStatus("expired");
      return { status: "expired" };
    }
    emitCacheStatus("hit");
    return { status: "hit", profile: parsed.profile };
  } catch (e) {
    log.warn("readDriverProfileCache error", { error: e });
    emitCacheStatus("error");
    return { status: "error" };
  }
}

/**
 * Écrit le profil en cache (génération pour éviter qu'une écriture lente n'écrase une plus récente).
 */
export async function writeDriverProfileCache(profile: Driver): Promise<void> {
  const gen = ++writeGeneration;
  const driver_id = profile.id;
  const company_id =
    typeof profile.company_id === "number" ? profile.company_id : null;
  const envelope: DriverProfileCacheEnvelope = {
    schema_version: DRIVER_PROFILE_CACHE_SCHEMA_VERSION,
    profile,
    driver_id,
    company_id,
    cached_at_ms: Date.now(),
  };
  let serialized: string;
  try {
    serialized = JSON.stringify(envelope);
  } catch (e) {
    log.warn("writeDriverProfileCache stringify failed", { error: e });
    emitCacheStatus("error");
    return;
  }
  if (serialized.length > MAX_PROFILE_CACHE_SIZE_BYTES) {
    log.warn("writeDriverProfileCache skipped: payload too large", {
      length: serialized.length,
    });
    emitCacheStatus("oversized_skip");
    return;
  }
  try {
    await AsyncStorage.setItem(DRIVER_PROFILE_CACHE_STORAGE_KEY, serialized);
  } catch (e) {
    log.warn("writeDriverProfileCache setItem failed", { error: e });
    emitCacheStatus("error");
    return;
  }
  if (gen !== writeGeneration) {
    return;
  }
}

export async function purgeDriverProfileCache(): Promise<void> {
  try {
    await AsyncStorage.removeItem(DRIVER_PROFILE_CACHE_STORAGE_KEY);
  } catch (e) {
    log.warn("purgeDriverProfileCache failed", { error: e });
  }
}

/**
 * Après GET /auth/me : si l'identité utilisateur ne correspond pas au profil caché, purge.
 */
export async function invalidateDriverProfileCacheIfUserMismatch(
  authUserId: number
): Promise<void> {
  try {
    const raw = await AsyncStorage.getItem(DRIVER_PROFILE_CACHE_STORAGE_KEY);
    if (raw === null) return;
    let parsed: unknown;
    try {
      parsed = JSON.parse(raw);
    } catch {
      await purgeDriverProfileCache();
      return;
    }
    if (!isEnvelope(parsed)) {
      await purgeDriverProfileCache();
      return;
    }
    if (parsed.profile.user_id !== authUserId) {
      log.info("invalidateDriverProfileCacheIfUserMismatch: user_id mismatch", {
        authUserId,
        cachedUserId: parsed.profile.user_id,
      });
      await purgeDriverProfileCache();
    }
  } catch (e) {
    log.warn("invalidateDriverProfileCacheIfUserMismatch error", { error: e });
  }
}
