import { getItem, setItem, removeItem } from "../storage/typedStorage";
import { STORAGE_KEYS } from "../storage/storageKeys";

type DisclosurePayload = {
  accepted: boolean;
  at: string;
};

export async function readNotificationDisclosureAccepted(): Promise<boolean> {
  const value = await getItem<DisclosurePayload | boolean>(
    STORAGE_KEYS.DRIVER_NOTIFICATION_DISCLOSURE_ACCEPTED
  );
  if (value == null) return false;
  if (typeof value === "boolean") return value;
  return Boolean(value.accepted);
}

const acceptanceListeners = new Set<() => void>();

export function subscribeNotificationDisclosureAccepted(listener: () => void): () => void {
  acceptanceListeners.add(listener);
  return () => {
    acceptanceListeners.delete(listener);
  };
}

export async function markNotificationDisclosureAccepted(): Promise<void> {
  await setItem<DisclosurePayload>(STORAGE_KEYS.DRIVER_NOTIFICATION_DISCLOSURE_ACCEPTED, {
    accepted: true,
    at: new Date().toISOString(),
  });
  acceptanceListeners.forEach((listener) => listener());
}

export async function resetNotificationDisclosureAccepted(): Promise<void> {
  await removeItem(STORAGE_KEYS.DRIVER_NOTIFICATION_DISCLOSURE_ACCEPTED);
}
