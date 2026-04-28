import * as SecureStore from "../storage/secureStoreCompat";

const KEY = "guest_saferpay_pending_v1";

export type GuestSaferpayPending = {
  status_token: string;
  guest_booking_id: string;
  draft_id?: string;
};

export async function setGuestSaferpayPending(value: GuestSaferpayPending | null): Promise<void> {
  if (!value) {
    await SecureStore.deleteItemAsync(KEY);
    return;
  }
  await SecureStore.setItemAsync(KEY, JSON.stringify(value));
}

export async function getGuestSaferpayPending(): Promise<GuestSaferpayPending | null> {
  const raw = await SecureStore.getItemAsync(KEY);
  if (!raw) return null;
  try {
    const p = JSON.parse(raw) as GuestSaferpayPending;
    if (p?.status_token && p?.guest_booking_id) {
      return p;
    }
  } catch {
    /* ignore */
  }
  return null;
}
