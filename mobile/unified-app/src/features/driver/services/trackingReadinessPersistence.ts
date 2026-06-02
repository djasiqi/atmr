/**
 * Persistance "tracking onboarding terminé" — empêche la Readiness Gate de
 * réapparaître à chaque ouverture une fois que tous les prérequis ont été
 * validés au moins une fois sur cet appareil.
 */
import { getItem, setItem, removeItem } from "../../../core/storage/typedStorage";
import { STORAGE_KEYS } from "../../../core/storage/storageKeys";

type Payload = {
  onboarded: boolean;
  at: string;
};

export async function readTrackingOnboarded(): Promise<boolean> {
  const value = await getItem<Payload | boolean>(STORAGE_KEYS.DRIVER_TRACKING_ONBOARDED);
  if (value == null) return false;
  if (typeof value === "boolean") return value;
  return Boolean(value.onboarded);
}

export async function markTrackingOnboarded(): Promise<void> {
  await setItem<Payload>(STORAGE_KEYS.DRIVER_TRACKING_ONBOARDED, {
    onboarded: true,
    at: new Date().toISOString(),
  });
}

export async function resetTrackingOnboarded(): Promise<void> {
  await removeItem(STORAGE_KEYS.DRIVER_TRACKING_ONBOARDED);
}
