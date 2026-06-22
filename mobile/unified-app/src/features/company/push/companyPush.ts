import type { Router } from "expo-router";
import { resolveCompanyDeepLink } from "../../../core/navigation/deepLinkHandler";

export type InstitutionOfferPushPreview = {
  institution_name?: string;
  patient_name?: string;
  trip_type?: string;
  scheduled_time_label?: string;
  mission_date?: string;
  expires_at?: string;
  deep_link?: string;
};

export type CompanyPushPayload = {
  type: string;
  offer_id?: number;
  request_id?: number;
  booking_id?: number;
  company_id?: number;
  event_id?: string;
  dedupe_key?: string;
  deep_link?: string;
  preview?: InstitutionOfferPushPreview;
};

const offerPreviewById = new Map<number, InstitutionOfferPushPreview>();
const openedAtByOfferId = new Map<number, number>();

export function resetOfferPushPreviewStoreForTests(): void {
  offerPreviewById.clear();
  openedAtByOfferId.clear();
}

export function setOfferPushPreview(
  offerId: number,
  preview: InstitutionOfferPushPreview
): void {
  offerPreviewById.set(offerId, preview);
}

export function getOfferPushPreview(
  offerId: number
): InstitutionOfferPushPreview | undefined {
  return offerPreviewById.get(offerId);
}

export function markOfferPushOpened(offerId: number): void {
  openedAtByOfferId.set(offerId, Date.now());
}

export function consumeOfferOpenToAcceptSeconds(offerId: number): number | null {
  const openedAt = openedAtByOfferId.get(offerId);
  if (openedAt == null) return null;
  openedAtByOfferId.delete(offerId);
  return Math.max(0, (Date.now() - openedAt) / 1000);
}

function readNumber(value: unknown): number | undefined {
  if (value == null) return undefined;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function readString(value: unknown): string | undefined {
  return typeof value === "string" && value.trim().length > 0 ? value.trim() : undefined;
}

export function buildOfferPreviewFromPushData(
  data: Record<string, unknown>
): InstitutionOfferPushPreview {
  return {
    institution_name: readString(data.institution_name),
    patient_name: readString(data.patient_name),
    trip_type: readString(data.trip_type),
    scheduled_time_label: readString(data.scheduled_time_label),
    mission_date: readString(data.mission_date),
    expires_at: readString(data.expires_at),
    deep_link: readString(data.deep_link ?? data.deepLink),
  };
}

export function parseCompanyPushPayload(data: unknown): CompanyPushPayload | null {
  if (!data || typeof data !== "object") return null;
  const record = data as Record<string, unknown>;
  const type = typeof record.type === "string" ? record.type : null;
  if (!type) return null;

  const offerId = readNumber(record.offer_id ?? record.offerId);
  const requestId = readNumber(record.request_id ?? record.requestId);
  const bookingId = readNumber(record.booking_id ?? record.bookingId);
  const preview = buildOfferPreviewFromPushData(record);

  if (offerId != null) {
    setOfferPushPreview(offerId, preview);
  }

  return {
    type,
    offer_id: offerId,
    request_id: requestId,
    booking_id: bookingId,
    company_id: readNumber(record.company_id ?? record.companyId),
    event_id: readString(record.event_id),
    dedupe_key: readString(record.dedupe_key),
    deep_link: readString(record.deep_link ?? record.deepLink),
    preview,
  };
}

export function resolveCompanyPushTitleBody(
  data: Record<string, unknown>,
  fallbackTitle = "Liri Entreprise",
  fallbackBody = "Nouvelle activité"
): { title: string; body: string } {
  const notification = data.notification;
  if (notification && typeof notification === "object") {
    const n = notification as Record<string, unknown>;
    const title = readString(n.title);
    const body = readString(n.body);
    if (title && body) return { title, body };
  }
  const title = readString(data.title) ?? fallbackTitle;
  const preview = buildOfferPreviewFromPushData(data);
  const institution = preview.institution_name;
  const patient = preview.patient_name;
  const schedule = preview.scheduled_time_label;
  if (institution && patient) {
    const parts = [institution, patient];
    if (schedule) parts.push(schedule);
    return { title: title || "Nouvelle demande institution", body: parts.join(" — ") };
  }
  const body = readString(data.body) ?? readString(data.message) ?? fallbackBody;
  return { title, body };
}

const INSTITUTION_OFFER_TYPES = new Set([
  "new_request",
  "request_updated",
  "offer_unavailable",
]);

export function navigateFromCompanyPush(router: Router, payload: CompanyPushPayload): void {
  if (payload.deep_link) {
    const resolved = resolveCompanyDeepLink(payload.deep_link);
    if (resolved?.route) {
      router.push(resolved.route as never);
      return;
    }
  }

  if (
    payload.offer_id != null &&
    (INSTITUTION_OFFER_TYPES.has(payload.type) || payload.request_id != null)
  ) {
    router.push({
      pathname: "/(app)/(company)/offers/[offerId]",
      params: {
        offerId: String(payload.offer_id),
        ...(payload.request_id != null ? { request: String(payload.request_id) } : {}),
      },
    });
    return;
  }

  if (payload.booking_id != null) {
    router.push({
      pathname: "/(app)/(company)/ride-details",
      params: { rideId: String(payload.booking_id) },
    });
    return;
  }

  router.push("/(app)/(company)/dashboard");
}
