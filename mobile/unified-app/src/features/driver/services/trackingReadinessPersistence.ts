/**
 * Persistance tracking chauffeur :
 * - trackingOnboarded : panneau pédagogique vu au moins une fois (jamais effacé auto)
 * - trackingNeedsAttention : réglages incomplets ou révoqués (réaffichage informatif)
 */
import { getItem, setItem, removeItem } from "../../../core/storage/typedStorage";
import { STORAGE_KEYS } from "../../../core/storage/storageKeys";

type OnboardedPayload = {
  onboarded: boolean;
  at: string;
};

type NeedsAttentionPayload = {
  needsAttention: boolean;
  at: string;
};

export async function readTrackingOnboarded(): Promise<boolean> {
  const value = await getItem<OnboardedPayload | boolean>(STORAGE_KEYS.DRIVER_TRACKING_ONBOARDED);
  if (value == null) return false;
  if (typeof value === "boolean") return value;
  return Boolean(value.onboarded);
}

export async function markTrackingOnboarded(): Promise<void> {
  await setItem<OnboardedPayload>(STORAGE_KEYS.DRIVER_TRACKING_ONBOARDED, {
    onboarded: true,
    at: new Date().toISOString(),
  });
}

export async function resetTrackingOnboarded(): Promise<void> {
  await removeItem(STORAGE_KEYS.DRIVER_TRACKING_ONBOARDED);
}

export async function readTrackingNeedsAttention(): Promise<boolean> {
  const value = await getItem<NeedsAttentionPayload | boolean>(
    STORAGE_KEYS.DRIVER_TRACKING_NEEDS_ATTENTION
  );
  if (value == null) return false;
  if (typeof value === "boolean") return value;
  return Boolean(value.needsAttention);
}

export async function setTrackingNeedsAttention(needsAttention: boolean): Promise<void> {
  if (!needsAttention) {
    await removeItem(STORAGE_KEYS.DRIVER_TRACKING_NEEDS_ATTENTION);
    return;
  }
  await setItem<NeedsAttentionPayload>(STORAGE_KEYS.DRIVER_TRACKING_NEEDS_ATTENTION, {
    needsAttention: true,
    at: new Date().toISOString(),
  });
}
