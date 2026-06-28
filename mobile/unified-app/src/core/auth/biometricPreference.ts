import { getItem, removeItem, setItem } from "../storage/typedStorage";
import { STORAGE_KEYS } from "../storage/storageKeys";

async function readLegacyDriverBiometricEnabled(): Promise<boolean> {
  const legacy = await getItem<boolean>(STORAGE_KEYS.DRIVER_BIOMETRIC_ENABLED);
  return legacy === true;
}

export async function readAuthBiometricEnabled(): Promise<boolean> {
  const value = await getItem<boolean>(STORAGE_KEYS.AUTH_BIOMETRIC_ENABLED);
  if (value === true) return true;
  return readLegacyDriverBiometricEnabled();
}

export async function writeAuthBiometricEnabled(enabled: boolean): Promise<void> {
  if (enabled) {
    await setItem(STORAGE_KEYS.AUTH_BIOMETRIC_ENABLED, true);
    await setItem(STORAGE_KEYS.DRIVER_BIOMETRIC_ENABLED, true);
    return;
  }
  await removeItem(STORAGE_KEYS.AUTH_BIOMETRIC_ENABLED);
  await removeItem(STORAGE_KEYS.DRIVER_BIOMETRIC_ENABLED);
}
