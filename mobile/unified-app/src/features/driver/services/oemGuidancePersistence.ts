/**
 * Acquittement du guide fabricant (OEM) — pas une validation technique des réglages.
 */
import Constants from "expo-constants";

import { getItem, removeItem, setItem } from "../../../core/storage/typedStorage";
import { STORAGE_KEYS } from "../../../core/storage/storageKeys";

export type OemGuidanceAcknowledgement = {
  acknowledgedAt: string;
  oem: string;
  appVersion?: string;
};

function readAppVersion(): string | undefined {
  try {
    const version = Constants.expoConfig?.version ?? Constants.nativeAppVersion;
    return version ? String(version) : undefined;
  } catch {
    return undefined;
  }
}

export async function readOemGuidanceAcknowledgement(): Promise<OemGuidanceAcknowledgement | null> {
  const value = await getItem<OemGuidanceAcknowledgement>(
    STORAGE_KEYS.DRIVER_OEM_GUIDANCE_ACKNOWLEDGED
  );
  if (!value || typeof value !== "object") return null;
  if (typeof value.acknowledgedAt !== "string" || typeof value.oem !== "string") {
    return null;
  }
  return value;
}

/** true si un acquittement existe pour le même OEM détecté. */
export async function isOemGuidanceAcknowledgedFor(oem: string | null): Promise<boolean> {
  if (!oem) return false;
  const stored = await readOemGuidanceAcknowledgement();
  if (!stored) return false;
  return stored.oem === oem;
}

export async function markOemGuidanceAcknowledged(oem: string): Promise<void> {
  const payload: OemGuidanceAcknowledgement = {
    acknowledgedAt: new Date().toISOString(),
    oem,
    appVersion: readAppVersion(),
  };
  await setItem(STORAGE_KEYS.DRIVER_OEM_GUIDANCE_ACKNOWLEDGED, payload);
}

export async function clearOemGuidanceAcknowledgement(): Promise<void> {
  await removeItem(STORAGE_KEYS.DRIVER_OEM_GUIDANCE_ACKNOWLEDGED);
}
