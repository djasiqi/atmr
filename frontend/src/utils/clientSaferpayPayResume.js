/** Reprise du bandeau paiement sur le dashboard après échec sur la page dédiée. */

export const SAFERPAY_PAY_RESUME_STORAGE_KEY = 'atmr:saferpayPayResume';

/**
 * @param {{ bookingId: number, finalAmount?: number, payerLabel?: string, lifecycleLabel?: string }} payload
 */
export function writeSaferpayPayResume(payload) {
  try {
    sessionStorage.setItem(SAFERPAY_PAY_RESUME_STORAGE_KEY, JSON.stringify(payload));
  } catch {
    /* ignore */
  }
}

export function clearSaferpayPayResume() {
  try {
    sessionStorage.removeItem(SAFERPAY_PAY_RESUME_STORAGE_KEY);
  } catch {
    /* ignore */
  }
}

/**
 * @returns {{ bookingId: number, finalAmount?: number, payerLabel?: string, lifecycleLabel?: string } | null}
 */
export function readAndConsumeSaferpayPayResume() {
  try {
    const raw = sessionStorage.getItem(SAFERPAY_PAY_RESUME_STORAGE_KEY);
    sessionStorage.removeItem(SAFERPAY_PAY_RESUME_STORAGE_KEY);
    if (!raw) return null;
    const obj = JSON.parse(raw);
    if (!obj || typeof obj !== 'object' || typeof obj.bookingId !== 'number') return null;
    return obj;
  } catch {
    return null;
  }
}
