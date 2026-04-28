import { Booking, PaymentStatus } from "./types";
import { trackClientKpiEvent } from "./statusEvents";

export function derivePaymentStatusFromBooking(booking: Booking): PaymentStatus {
  const bookingStatus = String(booking.status ?? "")
    .trim()
    .toLowerCase();
  const onlineStatus = String(booking.online_payment?.status ?? "")
    .trim()
    .toLowerCase();

  if (onlineStatus === "completed") return "paid";
  if (onlineStatus === "failed") return "failed";
  if (onlineStatus === "cancelled" || onlineStatus === "canceled") return "cancelled";
  if (onlineStatus === "pending" && booking.online_payment?.has_pending_session) {
    return "pending_verification";
  }
  if (bookingStatus === "awaiting_client_payment") return "required";
  if (!bookingStatus && !onlineStatus) {
    trackClientKpiEvent("status_dictionary_mismatch_event", {
      surface: "mobile",
      status: null,
      reason: "missing_payment_status",
    });
    return "unknown";
  }
  return "not_required";
}
