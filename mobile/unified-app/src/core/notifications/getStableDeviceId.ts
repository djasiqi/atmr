import * as Application from "expo-application";
import * as SecureStore from "expo-secure-store";

const DEVICE_ID_KEY = "@atmr/device_id";

let cachedDeviceId: string | null = null;

/**
 * Identifiant d'installation stable (survit aux redémarrages, change après réinstallation).
 */
export async function getStableDeviceId(): Promise<string> {
  if (cachedDeviceId) return cachedDeviceId;

  try {
    const installationId = Application.getInstallationIdAsync
      ? await Application.getInstallationIdAsync()
      : null;
    if (installationId && installationId.length > 0) {
      cachedDeviceId = installationId;
      return installationId;
    }
  } catch {
    // fallback SecureStore
  }

  const stored = await SecureStore.getItemAsync(DEVICE_ID_KEY).catch(() => null);
  if (stored && stored.length > 0) {
    cachedDeviceId = stored;
    return stored;
  }

  const generated = `atmr-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`;
  await SecureStore.setItemAsync(DEVICE_ID_KEY, generated).catch(() => undefined);
  cachedDeviceId = generated;
  return generated;
}

/** Réinitialise le cache (tests uniquement). */
export function resetStableDeviceIdCacheForTests(): void {
  cachedDeviceId = null;
}
