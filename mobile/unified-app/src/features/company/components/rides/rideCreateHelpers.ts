import { normalizeScheduledTimeIso } from "../../useRideForms";

/** Jour ISO 8601 côté backend : 0 = lundi … 6 = dimanche. */
export function backendWeekdayFromScheduledIso(raw: string): number | null {
  const n = normalizeScheduledTimeIso(raw);
  if (!n) return null;
  const d = new Date(n);
  if (Number.isNaN(d.getTime())) return null;
  return (d.getDay() + 6) % 7;
}

export function parseSimulationAmount(payload: unknown): number | null {
  if (!payload || typeof payload !== "object") return null;
  const raw = payload as Record<string, unknown>;
  const candidates = [raw.amount, (raw.pricing as Record<string, unknown> | undefined)?.amount];
  for (const candidate of candidates) {
    if (typeof candidate === "number" && Number.isFinite(candidate) && candidate > 0) return candidate;
    if (typeof candidate === "string") {
      const parsed = Number.parseFloat(candidate);
      if (Number.isFinite(parsed) && parsed > 0) return parsed;
    }
  }
  return null;
}

export function parseSimulationDistanceKm(payload: unknown): number | null {
  if (!payload || typeof payload !== "object") return null;
  const raw = payload as Record<string, unknown>;
  const pricing =
    raw.pricing && typeof raw.pricing === "object"
      ? (raw.pricing as Record<string, unknown>)
      : undefined;
  const breakdown =
    raw.breakdown && typeof raw.breakdown === "object"
      ? (raw.breakdown as Record<string, unknown>)
      : undefined;
  const candidates = [
    raw.distance_km,
    raw.distance,
    raw.distance_meters,
    pricing?.distance_km,
    pricing?.distance,
    pricing?.distance_meters,
    pricing?.distance_m,
    breakdown?.distance_km,
    breakdown?.distance,
    breakdown?.distance_meters,
    breakdown?.distance_m,
  ];
  for (const candidate of candidates) {
    if (typeof candidate === "number" && Number.isFinite(candidate) && candidate > 0) {
      // Heuristique : au-delà de 1000, on suppose des mètres.
      return candidate > 1000 ? candidate / 1000 : candidate;
    }
    if (typeof candidate === "string") {
      const parsed = Number.parseFloat(candidate);
      if (Number.isFinite(parsed) && parsed > 0) {
        return parsed > 1000 ? parsed / 1000 : parsed;
      }
    }
  }
  return null;
}

export type PricingSimulationAnalysis = {
  amount: number | null;
  warningMessage: string | null;
  blocked: boolean;
};

export function analyzePricingSimulation(payload: unknown): PricingSimulationAnalysis {
  const amount = parseSimulationAmount(payload);
  if (!payload || typeof payload !== "object") {
    return { amount, warningMessage: "Calcul auto indisponible: saisissez un montant.", blocked: true };
  }

  const raw = payload as Record<string, unknown>;
  const warnings = Array.isArray(raw.warnings)
    ? raw.warnings
    : Array.isArray((raw.breakdown as Record<string, unknown> | undefined)?.warnings)
      ? ((raw.breakdown as Record<string, unknown>).warnings as unknown[])
      : [];
  const warningList = warnings.filter((v): v is string => typeof v === "string");

  const blockingReasons = Array.isArray(raw.blocking_reasons)
    ? raw.blocking_reasons.filter((v): v is string => typeof v === "string")
    : [];
  const confidence = String(raw.confidence ?? "").toLowerCase();
  const modelUsed = String(
    (raw.breakdown as Record<string, unknown> | undefined)?.model_used ?? ""
  ).toLowerCase();
  const isDistanceDependentModel = modelUsed === "distance" || modelUsed === "hybrid_stack";

  const hasDistanceUnavailable = warningList.includes("distance_unavailable");
  const hasZoneUnresolved = warningList.includes("zone_unresolved");

  if ((confidence === "blocked" || blockingReasons.length > 0) && amount == null) {
    if (blockingReasons.includes("zone_unresolved")) {
      return {
        amount: null,
        warningMessage: "Calcul indisponible: zonage introuvable pour ce trajet. Saisissez un montant.",
        blocked: true,
      };
    }
    if (blockingReasons.includes("zone_unresolved_timeout")) {
      return {
        amount: null,
        warningMessage: "Calcul indisponible: délai de calcul zonage dépassé. Réessayez.",
        blocked: true,
      };
    }
    if (blockingReasons.includes("distance_unavailable")) {
      return {
        amount: null,
        warningMessage: "Calcul indisponible (distance). Saisissez un montant ou réessayez.",
        blocked: true,
      };
    }
    return {
      amount: null,
      warningMessage: "Calcul auto indisponible: saisissez un montant.",
      blocked: true,
    };
  }

  if (hasDistanceUnavailable && isDistanceDependentModel && amount == null) {
    return {
      amount: null,
      warningMessage: "Calcul indisponible (distance). Saisissez un montant ou réessayez.",
      blocked: true,
    };
  }

  if (hasZoneUnresolved && amount == null) {
    return {
      amount: null,
      warningMessage: "Calcul indisponible: zonage introuvable pour ce trajet. Saisissez un montant.",
      blocked: true,
    };
  }

  if (warningList.includes("zone_unresolved_fallback")) {
    return {
      amount,
      warningMessage: "Zonage partiellement résolu: calcul appliqué avec fallback conservateur.",
      blocked: false,
    };
  }

  return { amount, warningMessage: null, blocked: false };
}

export function parseMedicalHintsFromAddress(label: string): {
  establishment?: string;
  doctorName?: string;
  hospitalService?: string;
  notesMedical?: string;
} {
  const clean = label.trim();
  const lower = clean.toLowerCase();
  const firstSegment = clean.split(",")[0]?.trim() ?? clean;
  const out: {
    establishment?: string;
    doctorName?: string;
    hospitalService?: string;
    notesMedical?: string;
  } = {};

  if (/\bdr\.?\b|\bdocteur\b/.test(lower)) {
    out.doctorName = firstSegment;
  }
  if (
    /\bhôpital\b|\bhopital\b|\bclinique\b|\bmaternit|\burgences\b|\bimagerie\b|\bhug\b/.test(lower)
  ) {
    out.establishment = firstSegment;
  }
  if (/\burgences\b/.test(lower)) out.hospitalService = "Urgences";
  else if (/\bcardio/.test(lower)) out.hospitalService = "Cardiologie";
  else if (/\bchir/.test(lower)) out.hospitalService = "Chirurgie";

  const floor = clean.match(/(\d{1,2}\s*(?:e|eme|ème|er)\s*étage|étage\s*\d{1,2})/i)?.[0] ?? "";
  if (floor) out.notesMedical = `Étage: ${floor}`;

  return out;
}

type AddressLike = {
  label: string;
  placeId: string | null;
  latitude: number | null;
  longitude: number | null;
};

export type RecurrenceLimitMode = "count" | "until" | "open";

export type RecurrencePreviewInput = {
  /** ISO date/heure du premier trajet (scheduledAt). */
  scheduledAt: string;
  /** Type de récurrence. `none` = aucun trajet calculé. */
  recurrence: "none" | "daily" | "weekly" | "custom";
  /** Jours 0–6 (lun–dim) — utilisé pour `weekly` (multi-jours) et `custom`. */
  days?: number[];
  /** Date de fin YYYY-MM-DD optionnelle. Si absente, on borne sur `defaultDailyDays`/`defaultWeeklyWeeks`. */
  endDate?: string;
  /** Garde-fou : nombre max de dates à générer (défaut 100). */
  maxDates?: number;
  /** Borne par défaut quand pas de date de fin (daily/custom : nb de jours, weekly : nb de semaines). */
  defaultDailyDays?: number;
  defaultWeeklyWeeks?: number;
  /** Intervalle en semaines entre deux blocs (utilisé en `custom`, défaut 1). */
  intervalWeeks?: number;
};

export type RecurrencePreview = {
  /** Total de trajets prévus dans la fenêtre calculée. */
  total: number;
  /** Liste des dates générées (Date locale, normalisée à 00:00). */
  dates: Date[];
};

function startOfDay(d: Date): Date {
  const out = new Date(d.getTime());
  out.setHours(0, 0, 0, 0);
  return out;
}

function parseEndDateInclusive(endDate: string | undefined): Date | null {
  if (!endDate) return null;
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(endDate.trim());
  if (!m) return null;
  const y = Number(m[1]);
  const mo = Number(m[2]) - 1;
  const d = Number(m[3]);
  const out = new Date(y, mo, d, 23, 59, 59, 999);
  if (Number.isNaN(out.getTime())) return null;
  return out;
}

/**
 * Génère un aperçu des prochaines dates d'une récurrence (côté UI uniquement, pas envoyé au backend).
 * Sert à afficher au manager combien de trajets seront créés et un échantillon des prochaines dates.
 */
export function computeRecurrencePreview(input: RecurrencePreviewInput): RecurrencePreview {
  const empty: RecurrencePreview = { total: 0, dates: [] };
  if (input.recurrence === "none") return empty;

  const normalizedIso = normalizeScheduledTimeIso(input.scheduledAt);
  if (!normalizedIso) return empty;
  const startDate = startOfDay(new Date(normalizedIso));
  if (Number.isNaN(startDate.getTime())) return empty;

  const maxDates = Math.max(1, Math.min(1000, Math.floor(input.maxDates ?? 100)));
  const endInclusive = parseEndDateInclusive(input.endDate);
  const defaultDailyDays = Math.max(1, Math.floor(input.defaultDailyDays ?? 60));
  const defaultWeeklyWeeks = Math.max(1, Math.floor(input.defaultWeeklyWeeks ?? 12));

  const dates: Date[] = [];

  if (input.recurrence === "daily") {
    const hardStop = endInclusive ?? new Date(startDate.getTime() + defaultDailyDays * 86_400_000);
    let i = 0;
    while (i < maxDates) {
      const next = new Date(startDate.getTime() + i * 86_400_000);
      if (next.getTime() > hardStop.getTime()) break;
      dates.push(next);
      i += 1;
    }
    return { total: dates.length, dates };
  }

  if (input.recurrence === "weekly") {
    const wantedDaysWeekly = Array.from(
      new Set((input.days ?? []).map((x) => Math.max(0, Math.min(6, Math.floor(x))))),
    );
    const hardStop =
      endInclusive ?? new Date(startDate.getTime() + defaultWeeklyWeeks * 7 * 86_400_000);
    // Cas legacy : pas de jours sélectionnés → même jour de la semaine que le départ.
    if (wantedDaysWeekly.length === 0) {
      let i = 0;
      while (i < maxDates) {
        const next = new Date(startDate.getTime() + i * 7 * 86_400_000);
        if (next.getTime() > hardStop.getTime()) break;
        dates.push(next);
        i += 1;
      }
      return { total: dates.length, dates };
    }
    let cursor = startDate.getTime();
    let safety = 0;
    while (dates.length < maxDates && cursor <= hardStop.getTime() && safety < 1000) {
      const day = new Date(cursor);
      const wd = (day.getDay() + 6) % 7;
      if (wantedDaysWeekly.includes(wd)) {
        dates.push(day);
      }
      cursor += 86_400_000;
      safety += 1;
    }
    return { total: dates.length, dates };
  }

  const wantedDays = Array.from(
    new Set((input.days ?? []).map((x) => Math.max(0, Math.min(6, Math.floor(x))))),
  );
  if (wantedDays.length === 0) return empty;

  const intervalWeeks = Math.max(1, Math.min(12, Math.floor(input.intervalWeeks ?? 1)));
  const hardStop = endInclusive ?? new Date(startDate.getTime() + defaultDailyDays * 86_400_000);
  const startMs = startDate.getTime();
  const weekMs = 7 * 86_400_000;
  let cursor = startMs;
  let safety = 0;
  while (dates.length < maxDates && cursor <= hardStop.getTime() && safety < 1000) {
    const day = new Date(cursor);
    const wd = (day.getDay() + 6) % 7;
    if (wantedDays.includes(wd)) {
      // On ne garde la date que si elle tombe dans un bloc « actif » selon l'intervalle de semaines.
      // Bloc actif = nombre entier de blocs de `intervalWeeks` depuis le lundi de la semaine de départ.
      const startOfWeekMs = startMs - ((startDate.getDay() + 6) % 7) * 86_400_000;
      const startOfCursorWeekMs = cursor - ((day.getDay() + 6) % 7) * 86_400_000;
      const deltaWeeks = Math.round((startOfCursorWeekMs - startOfWeekMs) / weekMs);
      if (deltaWeeks >= 0 && deltaWeeks % intervalWeeks === 0) {
        dates.push(day);
      }
    }
    cursor += 86_400_000;
    safety += 1;
  }
  return { total: dates.length, dates };
}

type BuildRidePayloadInput = {
  structuredPayloadEnabled: boolean;
  clientId: number | null;
  pickup: string;
  dropoff: string;
  pickupAddress: AddressLike | null;
  dropoffAddress: AddressLike | null;
  scheduledTime: string;
  isRoundTrip: boolean;
  recurrence: "none" | "daily" | "weekly" | "custom";
  recurrenceLimitMode?: RecurrenceLimitMode;
  /** Nombre total de trajets (mode « nombre »). */
  recurrenceOccurrences?: number;
  /** Date de fin YYYY-MM-DD (mode « jusqu’au »). */
  recurrenceEndDate?: string;
  /** Jours 0–6 (lun–dim), utilisé si recurrence === custom. */
  recurrenceDays?: number[];
  /** Intervalle en semaines (utilisé si recurrence === custom). */
  recurrenceIntervalWeeks?: number;
  notesMedical: string;
  establishment: string;
  hospitalService: string;
  doctorName: string;
  pickupAccessNotes: string;
  dropoffAccessNotes: string;
  wheelchairClient: boolean;
  wheelchairProvide: boolean;
  internalNotes: string;
  notesMax: number;
  amountInput: string;
  amountSource: "preferential" | "simulated" | "manual" | null;
  pricingProfileId: number | null;
  pricingProfileVersionId: number | null;
  isMaterialDelivery: boolean;
  deliveryDescription: string;
  returnScheduledAt: string;
  billToPatient: boolean;
  hasActiveStay: boolean;
  clinicBillingPartyId: number | null;
};

function parseOptionalAmount(raw: string): number | null {
  const t = raw.trim().replace(",", ".");
  if (!t) return null;
  const n = Number.parseFloat(t);
  return Number.isFinite(n) && n >= 0 ? n : null;
}

function extractIsoDatePart(raw: string): string | null {
  const t = raw.trim();
  if (!t) return null;
  const directDate = /^(\d{4}-\d{2}-\d{2})$/.exec(t);
  if (directDate) return directDate[1];
  const dateTime = /^(\d{4}-\d{2}-\d{2})T/.exec(t);
  if (dateTime) return dateTime[1];
  return null;
}

const RECURRENCE_MAX_LOOP = 999;
const RECURRENCE_OPEN_DAILY = 365;
const RECURRENCE_OPEN_WEEKLY = 104;

/** Champs API alignés sur `ManualBookingCreateSchema` / `create_manual_booking`. */
export function buildRecurrenceApiFields(
  recurrence: "daily" | "weekly" | "custom",
  limitMode: RecurrenceLimitMode,
  occurrencesInput: number,
  endDate: string,
  recurrenceDays: number[],
): { occurrences: number; recurrence_end_date?: string; recurrence_days?: number[] } {
  let occ = Math.max(1, Math.min(RECURRENCE_MAX_LOOP, Math.floor(Number(occurrencesInput)) || 1));
  let recurrence_end_date: string | undefined;

  if (limitMode === "until") {
    const d = endDate.trim();
    if (/^\d{4}-\d{2}-\d{2}$/.test(d)) {
      recurrence_end_date = d;
      // Alignement web: conserver une longueur de série bornée
      // même quand une date de fin est renseignée.
      occ = Math.max(1, Math.min(52, Math.floor(Number(occurrencesInput)) || 10));
    }
  } else if (limitMode === "open") {
    if (recurrence === "weekly") occ = RECURRENCE_OPEN_WEEKLY;
    else if (recurrence === "daily") occ = RECURRENCE_OPEN_DAILY;
    else occ = RECURRENCE_MAX_LOOP;
  }

  const recurrence_days =
    (recurrence === "custom" || recurrence === "weekly") && recurrenceDays.length > 0
      ? Array.from(new Set(recurrenceDays.map((x) => Math.max(0, Math.min(6, Math.floor(x)))))).sort(
          (a, b) => a - b,
        )
      : undefined;

  return { occurrences: occ, recurrence_end_date, recurrence_days };
}

export function buildRideCreatePayload(input: BuildRidePayloadInput): Record<string, unknown> {
  const pickupPayload =
    input.structuredPayloadEnabled && input.pickupAddress
      ? {
          label: input.pickupAddress.label,
          place_id: input.pickupAddress.placeId,
          lat: input.pickupAddress.latitude,
          lon: input.pickupAddress.longitude,
        }
      : input.pickup.trim();
  const dropoffPayload =
    input.structuredPayloadEnabled && input.dropoffAddress
      ? {
          label: input.dropoffAddress.label,
          place_id: input.dropoffAddress.placeId,
          lat: input.dropoffAddress.latitude,
          lon: input.dropoffAddress.longitude,
        }
      : input.dropoff.trim();

  const payload: Record<string, unknown> = {
    client_id: input.clientId,
    pickup_address: pickupPayload,
    dropoff_address: dropoffPayload,
    pickup_location: input.pickup.trim(),
    dropoff_location: input.dropoff.trim(),
    scheduled_time: input.scheduledTime,
    is_return: input.isRoundTrip,
    is_round_trip: input.isRoundTrip,
    notes_medical: input.notesMedical.trim() || null,
  };

  if (typeof input.pickupAddress?.latitude === "number" && Number.isFinite(input.pickupAddress.latitude)) {
    payload.pickup_lat = input.pickupAddress.latitude;
  }
  if (typeof input.pickupAddress?.longitude === "number" && Number.isFinite(input.pickupAddress.longitude)) {
    payload.pickup_lon = input.pickupAddress.longitude;
  }
  if (typeof input.dropoffAddress?.latitude === "number" && Number.isFinite(input.dropoffAddress.latitude)) {
    payload.dropoff_lat = input.dropoffAddress.latitude;
  }
  if (typeof input.dropoffAddress?.longitude === "number" && Number.isFinite(input.dropoffAddress.longitude)) {
    payload.dropoff_lon = input.dropoffAddress.longitude;
  }
  const scheduledDatePart = extractIsoDatePart(input.scheduledTime);
  if (scheduledDatePart) {
    payload.scheduled_date = scheduledDatePart;
  }

  if (input.establishment.trim()) payload.medical_facility = input.establishment.trim();
  if (input.hospitalService.trim()) payload.hospital_service = input.hospitalService.trim();
  if (input.doctorName.trim()) payload.doctor_name = input.doctorName.trim();
  if (input.pickupAccessNotes.trim()) payload.pickup_access_notes = input.pickupAccessNotes.trim();
  if (input.dropoffAccessNotes.trim()) payload.dropoff_access_notes = input.dropoffAccessNotes.trim();
  if (input.wheelchairClient) payload.wheelchair_client_has = true;
  if (input.wheelchairProvide) payload.wheelchair_need = true;
  if (input.internalNotes.trim()) payload.notes = input.internalNotes.trim().slice(0, input.notesMax);

  const amount = parseOptionalAmount(input.amountInput);
  if (amount != null) payload.amount = amount;
  if (input.amountSource) payload.amount_source = input.amountSource;
  if (input.pricingProfileId) payload.pricing_profile_id = input.pricingProfileId;
  if (input.pricingProfileVersionId) payload.pricing_profile_version_id = input.pricingProfileVersionId;

  if (input.isMaterialDelivery) {
    payload.mission_type = "material_delivery";
    payload.delivery_description = input.deliveryDescription.trim() || null;
  }

  if (input.isRoundTrip) {
    const returnDatePart =
      extractIsoDatePart(input.returnScheduledAt) ?? extractIsoDatePart(input.scheduledTime);
    if (returnDatePart) {
      payload.return_date = returnDatePart;
    }
    if (input.returnScheduledAt.includes("T")) {
      payload.return_time = input.returnScheduledAt;
    }
  }

  if (input.recurrence !== "none") {
    payload.is_recurring = true;
    payload.recurrence_type = input.recurrence;
    const rb = buildRecurrenceApiFields(
      input.recurrence,
      input.recurrenceLimitMode ?? "count",
      input.recurrenceOccurrences ?? 10,
      input.recurrenceEndDate ?? "",
      input.recurrenceDays ?? [],
    );
    payload.occurrences = rb.occurrences;
    // Compatibilité endpoint web/client: certains chemins lisent encore `recurrence_series_length`.
    payload.recurrence_series_length = rb.occurrences;
    if (rb.recurrence_end_date) payload.recurrence_end_date = rb.recurrence_end_date;
    if (rb.recurrence_days?.length) payload.recurrence_days = rb.recurrence_days;
    if (input.recurrence === "custom") {
      const iv = Math.max(1, Math.min(12, Math.floor(Number(input.recurrenceIntervalWeeks) || 1)));
      if (iv > 1) {
        payload.recurrence_interval_weeks = iv;
        payload.recurrence_interval = iv;
      }
    }
  }

  if (input.billToPatient) {
    payload.bill_to_patient = true;
  } else if (input.hasActiveStay && input.clinicBillingPartyId) {
    payload.billing_party_id = input.clinicBillingPartyId;
  }

  return payload;
}
