import type { InstitutionTransportRequestSummary } from "../api/institutionOffersApi";

const BUSINESS_TZ = "Europe/Zurich";

function sortedLegs(req: InstitutionTransportRequestSummary | null | undefined) {
  return [...(req?.legs ?? [])].sort(
    (a, b) => (a.sequence_index ?? 0) - (b.sequence_index ?? 0)
  );
}

export function formatOutboundRouteLabel(
  req: InstitutionTransportRequestSummary | null | undefined
): string {
  const legs = sortedLegs(req);
  if (legs.length > 0) {
    const first = legs[0];
    const pickup = first.pickup_location ?? req?.pickup_location ?? "—";
    const dropoff = first.dropoff_location ?? req?.dropoff_location ?? "—";
    return `${pickup} → ${dropoff}`;
  }
  const pickup = req?.pickup_location ?? "—";
  const dropoff = req?.dropoff_location ?? "—";
  return `${pickup} → ${dropoff}`;
}

function genevaPartsFromDate(date: Date) {
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone: BUSINESS_TZ,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).formatToParts(date);
  const pick = (type: string) => parts.find((p) => p.type === type)?.value ?? "";
  return {
    year: pick("year"),
    month: pick("month"),
    day: pick("day"),
    hour: pick("hour"),
    minute: pick("minute"),
  };
}

/** Valeur `datetime-local` (heure murale Genève) depuis un Date. */
export function datetimeLocalValueFromDate(date: Date): string {
  const p = genevaPartsFromDate(date);
  return `${p.year}-${p.month}-${p.day}T${p.hour}:${p.minute}`;
}

/** ISO naïf Genève depuis une valeur `datetime-local`. */
export function isoFromDatetimeLocalValue(value: string): string | null {
  const m = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})/.exec(String(value).trim());
  if (!m) return null;
  return `${m[1]}-${m[2]}-${m[3]}T${m[4]}:${m[5]}:00`;
}

/** ISO naïf Genève pour POST accept (proposed_pickup_time). */
export function formatProposedPickupIso(date: Date): string {
  const p = genevaPartsFromDate(date);
  return `${p.year}-${p.month}-${p.day}T${p.hour}:${p.minute}:00`;
}

export function formatProposedPickupDisplay(date: Date): string {
  return new Intl.DateTimeFormat("fr-CH", {
    timeZone: BUSINESS_TZ,
    day: "2-digit",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(date);
}

export const INSTITUTION_ACCEPT_NOW_OFFSET_MINUTES = 15;

/** Départ immédiat : now + délai opérationnel (Genève). */
export function computeAcceptNowPickupIso(
  offsetMinutes: number = INSTITUTION_ACCEPT_NOW_OFFSET_MINUTES
): string {
  const next = new Date();
  next.setMinutes(next.getMinutes() + offsetMinutes);
  return formatProposedPickupIso(next);
}

function shiftDateMinutes(date: Date, deltaMinutes: number): Date {
  const next = new Date(date.getTime());
  next.setMinutes(next.getMinutes() + deltaMinutes);
  return next;
}

/**
 * Horaire de prise en charge proposé par défaut (RDV − trajet si arrivée).
 */
export function computeDefaultProposedDate(
  req: InstitutionTransportRequestSummary | null | undefined,
  travelMinutes: number | null
): Date | null {
  if (!req) return null;

  const legs = sortedLegs(req);
  let sourceIso: string | null = null;
  let isArrival = req.scheduled_time_type === "arrival";

  for (const leg of legs) {
    if (leg.time_confirmed !== false && leg.scheduled_time) {
      sourceIso = leg.scheduled_time;
      isArrival = true;
      break;
    }
  }

  if (!sourceIso) {
    sourceIso = req.next_confirmed_time ?? req.scheduled_time ?? null;
  }

  if (!sourceIso) return null;
  const base = new Date(sourceIso);
  if (Number.isNaN(base.getTime())) return null;

  if (isArrival && travelMinutes != null && travelMinutes > 0) {
    return shiftDateMinutes(base, -travelMinutes);
  }
  return base;
}
