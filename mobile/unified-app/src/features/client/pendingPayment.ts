import * as SecureStore from "../../core/storage/secureStoreCompat";

const PENDING_PAYMENT_KEY = "unified.pending_payment";
export const PENDING_PAYMENT_TTL_MS = 30 * 60 * 1000;

export type PendingPayment = {
  bookingId: number;
  paymentId: number | null;
  contextId: string;
  createdAt: number;
};

type PendingPaymentLookup = {
  pending: PendingPayment | null;
  expired: boolean;
};

export async function setPendingPayment(payload: PendingPayment): Promise<void> {
  await SecureStore.setItemAsync(PENDING_PAYMENT_KEY, JSON.stringify(payload));
}

function normalizePendingPayment(raw: string): PendingPayment | null {
  const parsed = JSON.parse(raw) as PendingPayment;
  if (!parsed || !parsed.bookingId || !parsed.contextId) return null;
  return {
    bookingId: Number(parsed.bookingId),
    paymentId:
      parsed.paymentId === null || parsed.paymentId === undefined
        ? null
        : Number(parsed.paymentId),
    contextId: String(parsed.contextId),
    createdAt: Number(parsed.createdAt || Date.now()),
  };
}

async function readPendingPayment(): Promise<PendingPaymentLookup> {
  const raw = await SecureStore.getItemAsync(PENDING_PAYMENT_KEY);
  if (!raw) return { pending: null, expired: false };
  try {
    const normalized = normalizePendingPayment(raw);
    if (!normalized) return { pending: null, expired: false };
    if (Date.now() - normalized.createdAt > PENDING_PAYMENT_TTL_MS) {
      await clearPendingPayment();
      return { pending: null, expired: true };
    }
    return { pending: normalized, expired: false };
  } catch {
    return { pending: null, expired: false };
  }
}

export async function getPendingPayment(): Promise<PendingPayment | null> {
  const { pending } = await readPendingPayment();
  return pending;
}

export async function getPendingPaymentLookup(): Promise<PendingPaymentLookup> {
  return readPendingPayment();
}

export async function clearPendingPayment(): Promise<void> {
  await SecureStore.deleteItemAsync(PENDING_PAYMENT_KEY);
}
