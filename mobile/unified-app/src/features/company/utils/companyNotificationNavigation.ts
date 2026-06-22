import type { CompanyInboxNotification } from "../api/companyInboxApi";

export type CompanyInboxNavigationTarget = {
  pathname: "/(app)/(company)/offers/[offerId]" | "/(app)/(company)/ride-details";
  params: Record<string, string>;
};

function readMetaNumber(meta: Record<string, unknown>, key: string): number | null {
  const raw = meta[key];
  const value = raw != null ? Number(raw) : NaN;
  return Number.isFinite(value) ? value : null;
}

/**
 * Résout la cible de navigation pour une notification entreprise (miroir web).
 */
export function resolveCompanyInboxNavigation(
  notif: CompanyInboxNotification
): CompanyInboxNavigationTarget | null {
  const meta = (notif.metadata ?? {}) as Record<string, unknown>;
  const offerId = readMetaNumber(meta, "offer_id");
  const requestId = readMetaNumber(meta, "request_id");
  const bookingId = readMetaNumber(meta, "booking_id");

  if (
    notif.event_type === "request_updated" ||
    notif.event_type === "new_request" ||
    notif.event_type === "offer_unavailable"
  ) {
    if (offerId == null) return null;
    const params: Record<string, string> = { offerId: String(offerId) };
    if (requestId != null) params.request = String(requestId);
    if (typeof meta.mission_date === "string" && meta.mission_date) {
      params.missionDate = meta.mission_date;
    }
    if (typeof meta.institution_name === "string" && meta.institution_name) {
      params.institutionName = meta.institution_name;
    }
    return {
      pathname: "/(app)/(company)/offers/[offerId]",
      params,
    };
  }

  if (bookingId != null) {
    return {
      pathname: "/(app)/(company)/ride-details",
      params: { rideId: String(bookingId) },
    };
  }

  return null;
}
