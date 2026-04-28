import AsyncStorage from "@react-native-async-storage/async-storage";
import { STORAGE_KEYS } from "./storageKeys";

const CURRENT_STORAGE_SCHEMA_VERSION = 1;
const LEGACY_DRIVER_PROFILE_KEY = "driver_profile_cache_v1";

async function parseStoredVersion(raw: string | null): Promise<number> {
  if (!raw) return 0;
  const parsed = Number.parseInt(raw, 10);
  if (!Number.isFinite(parsed) || parsed < 0) return 0;
  return parsed;
}

async function applyMigrationV1(): Promise<void> {
  const [legacyRaw, newRaw] = await Promise.all([
    AsyncStorage.getItem(LEGACY_DRIVER_PROFILE_KEY),
    AsyncStorage.getItem(STORAGE_KEYS.DRIVER_PROFILE),
  ]);

  if (!legacyRaw) return;
  if (!newRaw) {
    await AsyncStorage.setItem(STORAGE_KEYS.DRIVER_PROFILE, legacyRaw);
  }
  await AsyncStorage.removeItem(LEGACY_DRIVER_PROFILE_KEY);
}

export async function runStorageMigrations(): Promise<void> {
  const rawVersion = await AsyncStorage.getItem(STORAGE_KEYS.STORAGE_MIGRATION_VERSION);
  let version = await parseStoredVersion(rawVersion);
  if (version >= CURRENT_STORAGE_SCHEMA_VERSION) return;

  if (version < 1) {
    await applyMigrationV1();
    version = 1;
    await AsyncStorage.setItem(STORAGE_KEYS.STORAGE_MIGRATION_VERSION, String(version));
  }
}

