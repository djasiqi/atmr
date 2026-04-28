import * as SecureStore from 'expo-secure-store';

const PENDING_PAYMENT_KEY = 'clientapp.pending_payment';

type PendingPayment = {
  bookingId: number;
  paymentId: number;
};

export async function setPendingPayment(bookingId: number, paymentId: number): Promise<void> {
  const value = JSON.stringify({ bookingId, paymentId } satisfies PendingPayment);
  await SecureStore.setItemAsync(PENDING_PAYMENT_KEY, value);
}

export async function getPendingPayment(): Promise<PendingPayment | null> {
  const raw = await SecureStore.getItemAsync(PENDING_PAYMENT_KEY);
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as PendingPayment;
    if (!parsed?.bookingId || !parsed?.paymentId) return null;
    return parsed;
  } catch {
    return null;
  }
}

export async function clearPendingPayment(): Promise<void> {
  await SecureStore.deleteItemAsync(PENDING_PAYMENT_KEY);
}
