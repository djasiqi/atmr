type PaymentEventName =
  | "payment_initialize_started"
  | "payment_initialize_failed"
  | "payment_redirect_opened"
  | "payment_return_received"
  | "payment_assert_started"
  | "payment_assert_succeeded"
  | "payment_assert_failed"
  | "payment_status_refetched";

type PaymentEventPayload = {
  bookingId: number;
  paymentId?: number | null;
  contextId?: string | null;
  requestId?: string | null;
  reason?: string;
};

export function logPaymentEvent(
  event: PaymentEventName,
  payload: PaymentEventPayload
) {
  // Placeholder analytics pipe for Phase 2B. Replace with analytics SDK later.
  console.info(`[payment-event] ${event}`, payload);
}
