const PENDING_STATUS = "PENDING";

/** Durée d'affichage d'une offre expirée avant masquage automatique (1h). */
export const EXPIRED_OFFER_VISIBLE_MS = 60 * 60 * 1000;

export type InstitutionOfferLike = {
  status?: string | null;
  can_respond?: boolean | null;
  expires_at?: string | null;
};

export function isInstitutionOfferExpired(
  offer: InstitutionOfferLike | null | undefined,
  now: Date = new Date()
): boolean {
  if (!offer?.expires_at) return false;
  const expiresAt = new Date(offer.expires_at);
  if (Number.isNaN(expiresAt.getTime())) return false;
  return expiresAt <= now;
}

/** Fallback client si can_respond est absent ou incohérent avec expires_at. */
export function canRespondToInstitutionOffer(
  offer: InstitutionOfferLike | null | undefined,
  now: Date = new Date()
): boolean {
  if (typeof offer?.can_respond === "boolean") {
    if (!offer.can_respond) return false;
  } else if (offer?.status && String(offer.status).toUpperCase() !== PENDING_STATUS) {
    return false;
  }

  if (offer?.expires_at) {
    const expiresAt = new Date(offer.expires_at);
    if (!Number.isNaN(expiresAt.getTime())) {
      return expiresAt > now;
    }
  }

  return offer?.can_respond !== false;
}

export function isInstitutionOfferVisible(
  offer: InstitutionOfferLike | null | undefined,
  nowMs: number = Date.now()
): boolean {
  if (!offer?.expires_at) return true;
  const expiresAt = new Date(offer.expires_at).getTime();
  if (Number.isNaN(expiresAt)) return true;
  if (expiresAt > nowMs) return true;
  return nowMs - expiresAt <= EXPIRED_OFFER_VISIBLE_MS;
}

export function filterVisibleInstitutionOffers<T extends InstitutionOfferLike>(
  offers: T[] | null | undefined,
  nowMs: number = Date.now()
): T[] {
  return (offers ?? []).filter((offer) => isInstitutionOfferVisible(offer, nowMs));
}

export type InstitutionOfferSegment = "urgent" | "today" | "upcoming";

export function segmentInstitutionOffer(
  offer: InstitutionOfferLike & {
    transport_request?: { scheduled_time?: string | null; mission_date?: string | null } | null;
  },
  now: Date = new Date()
): InstitutionOfferSegment {
  const scheduledRaw =
    offer.transport_request?.scheduled_time ??
    (offer.transport_request?.mission_date
      ? `${offer.transport_request.mission_date}T12:00:00`
      : null);
  if (!scheduledRaw) return "upcoming";
  const scheduled = new Date(scheduledRaw);
  if (Number.isNaN(scheduled.getTime())) return "upcoming";

  const diffMs = scheduled.getTime() - now.getTime();
  if (diffMs <= 2 * 60 * 60 * 1000) return "urgent";

  const sameDay =
    scheduled.getFullYear() === now.getFullYear() &&
    scheduled.getMonth() === now.getMonth() &&
    scheduled.getDate() === now.getDate();
  if (sameDay) return "today";
  return "upcoming";
}

export const INSTITUTION_OFFER_ERROR_CODES = [
  "OFFER_ALREADY_ACCEPTED",
  "OFFER_UNAVAILABLE",
  "OFFER_REJECTED",
  "OFFER_EXPIRED",
  "REQUEST_CANCELLED",
  "REQUEST_CONVERTED",
  "REQUEST_NOT_SENT",
  "CARRIER_EXTERNAL",
] as const;

export type InstitutionOfferErrorCode = (typeof INSTITUTION_OFFER_ERROR_CODES)[number];

export function isInstitutionOfferErrorCode(value: unknown): value is InstitutionOfferErrorCode {
  return (
    typeof value === "string" &&
    (INSTITUTION_OFFER_ERROR_CODES as readonly string[]).includes(value)
  );
}
