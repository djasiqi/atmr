import * as SecureStore from "../storage/secureStoreCompat";

const GUEST_SAFERPAY_PENDING_SLOT = "guest-saferpay-pending";

export type GuestSaferpayPending = {
  status_token: string;
  guest_booking_id: string;
  draft_id?: string;
};

export async function setGuestSaferpayPending(value: GuestSaferpayPending | null): Promise<void> {
  if (!value) {
    await SecureStore.deleteItemAsync(GUEST_SAFERPAY_PENDING_SLOT);
    return;
  }
  await SecureStore.setItemAsync(GUEST_SAFERPAY_PENDING_SLOT, JSON.stringify(value));
}

export async function getGuestSaferpayPending(): Promise<GuestSaferpayPending | null> {
  const raw = await SecureStore.getItemAsync(GUEST_SAFERPAY_PENDING_SLOT);
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
