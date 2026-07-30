import * as Application from "expo-application";
import {
  createAndPersistInstallationId,
  readInstallationId,
} from "../auth/authCredentialStore";

let cachedDeviceId: string | null = null;

type ApplicationWithInstallationId = typeof Application & {
  getInstallationIdAsync?: () => Promise<string | null>;
};

/**
 * Identifiant d'installation stable (survit aux redémarrages, change après réinstallation).
 * Échec de persistance → erreur device_identity_storage_unavailable (pas d'ID mémoire silencieux).
 */
export async function getStableDeviceId(): Promise<string> {
  if (cachedDeviceId) return cachedDeviceId;

  try {
    const application = Application as ApplicationWithInstallationId;
    const installationId = application.getInstallationIdAsync
      ? await application.getInstallationIdAsync()
      : null;
    if (installationId && installationId.length > 0) {
      cachedDeviceId = installationId;
      return installationId;
    }
  } catch {
    // fallback SecureStore via authCredentialStore
  }

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
