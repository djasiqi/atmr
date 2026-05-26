import type { Router } from "expo-router";

export type CompanyPushPayload = {
  type: string;
  booking_id?: number;
  company_id?: number;
  event_id?: string;
};

export function parseCompanyPushPayload(data: unknown): CompanyPushPayload | null {
  if (!data || typeof data !== "object") return null;
  const record = data as Record<string, unknown>;
  const type = typeof record.type === "string" ? record.type : null;
  if (!type) return null;
  const bookingRaw = record.booking_id ?? record.bookingId;
  const bookingId = bookingRaw != null ? Number(bookingRaw) : undefined;
  return {
    type,
    booking_id: Number.isFinite(bookingId) ? bookingId : undefined,
    company_id:
      record.company_id != null ? Number(record.company_id) : undefined,
    event_id: typeof record.event_id === "string" ? record.event_id : undefined,
  };
}

export function navigateFromCompanyPush(
  router: Router,
  payload: CompanyPushPayload
): void {
  if (payload.booking_id != null) {
    router.push({
      pathname: "/(app)/(company)/ride-details",
      params: { rideId: String(payload.booking_id) },
    });
    return;
  }
  router.push("/(app)/(company)/dashboard");
}
