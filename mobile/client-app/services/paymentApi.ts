import { api } from '@/services/api';
import type { SaferpayAssertResponse, SaferpayInitializeResponse } from '@/types/api';

export const SAFERPAY_ASSERT_CLIENT_TIMEOUT_MS = 150000;

export async function initializeSaferpayCheckout(
  bookingId: number,
  returnUrl: string,
): Promise<SaferpayInitializeResponse> {
  const res = await api.post<{ data?: SaferpayInitializeResponse } | SaferpayInitializeResponse>(
    `/bookings/${bookingId}/saferpay/initialize`,
    { return_url: returnUrl },
  );
  if ('data' in (res.data as { data?: SaferpayInitializeResponse })) {
    return ((res.data as { data?: SaferpayInitializeResponse }).data ?? {}) as SaferpayInitializeResponse;
  }
  return res.data as SaferpayInitializeResponse;
}

export async function assertSaferpayCheckout(
  bookingId: number,
  paymentId: number,
): Promise<SaferpayAssertResponse> {
  const res = await api.post<{ data?: SaferpayAssertResponse } | SaferpayAssertResponse>(
    `/bookings/${bookingId}/saferpay/assert`,
    { payment_id: paymentId },
    { timeout: SAFERPAY_ASSERT_CLIENT_TIMEOUT_MS },
  );
  if ('data' in (res.data as { data?: SaferpayAssertResponse })) {
    return ((res.data as { data?: SaferpayAssertResponse }).data ?? {}) as SaferpayAssertResponse;
  }
  return res.data as SaferpayAssertResponse;
}
