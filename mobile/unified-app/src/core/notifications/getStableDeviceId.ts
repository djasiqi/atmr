import {
  createAndPersistInstallationId,
  readInstallationId,
} from "../auth/authCredentialStore";

let cachedDeviceId: string | null = null;

/**
 * Identifiant d'installation stable (survit aux redémarrages, change après réinstallation).
 *
 * Source de vérité : SecureStore via authCredentialStore.
 * Ne pas utiliser de méthodes non documentées d'expo-application (ex. getInstallationIdAsync).
 * Échec de persistance → erreur device_identity_storage_unavailable (pas d'ID mémoire silencieux).
 */
export async function getStableDeviceId(): Promise<string> {
  if (cachedDeviceId) return cachedDeviceId;

  const stored = await readInstallationId();
  if (stored.status === "found") {
    cachedDeviceId = stored.value;
    return stored.value;
  }
  if (stored.status === "temporarily_unavailable") {
    throw new Error("device_identity_storage_unavailable");
  }

  const created = await createAndPersistInstallationId();
  if (created.status !== "found") {
    throw new Error("device_identity_storage_unavailable");
  }
  cachedDeviceId = created.value;
  return created.value;
}

/** Réinitialise le cache (tests uniquement). */
export function resetStableDeviceIdCacheForTests(): void {
  cachedDeviceId = null;
}
