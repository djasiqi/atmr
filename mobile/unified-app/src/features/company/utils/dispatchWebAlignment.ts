/**
 * Alignement tableau dispatch mobile ↔ web (`DispatchTable` + `companyService.fetchDispatchDelays`).
 */

/** Identique au web : `frontend/src/pages/company/Dashboard/components/DispatchTable.jsx`. */
export function dispatchDelayLevel(minutes: number): "critical" | "moderate" | "light" | null {
  if (!minutes || minutes <= 0) return null;
  if (minutes <= 5) return "light";
  if (minutes <= 15) return "moderate";
  return "critical";
}

/**
 * Couleurs heure + liseré carte + pastille chauffeur (même recette que assignation : fond pastel,
 * bordure `rgba(..., 0.25)`, texte accent).
 */
export const DISPATCH_DELAY_SEVERITY_UI = {
  light: {
    time: "#b45309",
    stripe: "#f59e0b",
    badgeBg: "#fffbeb",
    badgeBorder: "rgba(217, 119, 6, 0.25)",
  },
  moderate: {
    time: "#c2410c",
    stripe: "#ea580c",
    badgeBg: "#fff7ed",
    badgeBorder: "rgba(234, 88, 12, 0.25)",
  },
  critical: {
    time: "#b91c1c",
    stripe: "#ef4444",
    badgeBg: "#fef2f2",
    badgeBorder: "rgba(220, 38, 38, 0.25)",
  },
} as const;

export type DispatchDelaySeverity = keyof typeof DISPATCH_DELAY_SEVERITY_UI;

export function uiForDispatchDelayMinutes(delayMinutes: number | null | undefined): {
  severity: DispatchDelaySeverity;
  timeColor: string;
  stripeColor: string;
  badgeBg: string;
  badgeBorder: string;
} | null {
  if (delayMinutes == null || delayMinutes <= 0) return null;
  const level = dispatchDelayLevel(delayMinutes);
  if (!level) return null;
  const ui = DISPATCH_DELAY_SEVERITY_UI[level];
  return {
    severity: level,
    timeColor: ui.time,
    stripeColor: ui.stripe,
    badgeBg: ui.badgeBg,
    badgeBorder: ui.badgeBorder,
  };
}

/** Même sous-tendance que `DispatchTable.jsx` (`formatTime` + `toLocaleTimeString`). */
export function formatDispatchScheduledTime(isoLike: string | null | undefined): string {
  if (!isoLike?.trim()) return "\u2014";
  const date = new Date(isoLike.trim());
  if (Number.isNaN(date.getTime())) return "\u2014";
  return date.toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit" });
}

export type DispatchDelayFlattenRow = {
  booking_id: number;
  delay_minutes: number;
  is_pickup: boolean;
};

/**
 * Minutes de retard affichables : aligné sur `DispatchTable.jsx` (`getDelayLevel`) — tout retard **> 0**
 * (palier léger 1–5 min inclus), pas seulement ≥ 5 comme l’ancien `fetchDispatchDelays`.
 * Agrégation type delayMap : `delay_minutes || pickup || dropoff`.
 */
function finiteNum(v: unknown, fallback = 0): number {
  const n = typeof v === "number" ? v : Number.parseFloat(String(v ?? ""));
  return Number.isFinite(n) ? n : fallback;
}

const EXCLUDED_DELAY_BOOKING_STATUSES = new Set([
  "completed",
  "return_completed",
  "canceled",
  "cancelled",
  "in_progress",
  "awaiting_client_payment",
]);

function nestedBookingStatusLower(d: Record<string, unknown>): string {
  const b = d.booking;
  if (b && typeof b === "object") {
    return String((b as Record<string, unknown>).status ?? "")
      .trim()
      .toLowerCase();
  }
  return "";
}

function shouldSkipDelayRow(d: Record<string, unknown>): boolean {
  const st = nestedBookingStatusLower(d);
  return Boolean(st && EXCLUDED_DELAY_BOOKING_STATUSES.has(st));
}

/** booking_id tel que `/delays*` + liste courses (booking.id). Tolère objet `booking`, camelCase rare. */
function delayRowBookingId(d: Record<string, unknown>): number | null {
  const nestedBook = d.booking;
  const fromNested =
    nestedBook && typeof nestedBook === "object"
      ? finiteNum((nestedBook as Record<string, unknown>).id, NaN)
      : NaN;
  const direct = finiteNum(
    d.booking_id ?? (d.bookingId as unknown) ?? (Number.isFinite(fromNested) ? fromNested : NaN),
    NaN,
  );
  const rounded = Number.isFinite(direct) ? Math.round(direct) : NaN;
  return Number.isFinite(rounded) ? rounded : null;
}

/** Fusion `/delays/live` + `/delays` (aligné UnifiedDispatchRefactored côté web). */
export function mergeCompanyDispatchDelaySources(live: unknown[], snapshot: unknown[]): unknown[] {
  const map = new Map<number, Record<string, unknown>>();

  const bump = (row: unknown) => {
    if (!row || typeof row !== "object") return;
    const d = row as Record<string, unknown>;
    if (shouldSkipDelayRow(d)) return;
    const bookingId = delayRowBookingId(d);
    if (bookingId == null) return;

    const minutesAgg = Math.round(
      Math.max(
        finiteNum(d.delay_minutes ?? d.delayMinutes, 0),
        finiteNum(d.pickup_delay_minutes, 0),
        finiteNum(d.dropoff_delay_minutes, 0),
      ),
    );

    const pickupEtaCandidate =
      (typeof d.pickup_eta === "string" && d.pickup_eta ? d.pickup_eta : "") ||
      (typeof d.current_eta === "string" && d.current_eta ? d.current_eta : "") ||
      null;

    const prev = map.get(bookingId);
    if (!prev) {
      map.set(bookingId, {
        booking_id: bookingId,
        delay_minutes: minutesAgg,
        pickup_delay_minutes: d.pickup_delay_minutes ?? null,
        dropoff_delay_minutes: d.dropoff_delay_minutes ?? null,
        pickup_eta: pickupEtaCandidate ?? (d.pickup_eta ?? null),
        current_eta: d.current_eta ?? null,
      });
      return;
    }

    const prevMin = Math.round(
      Math.max(
        finiteNum(prev.delay_minutes, 0),
        finiteNum(prev.pickup_delay_minutes, 0),
        finiteNum(prev.dropoff_delay_minutes, 0),
      ),
    );
    const nextMin = Math.max(prevMin, minutesAgg);

    map.set(bookingId, {
      ...prev,
      ...d,
      booking_id: bookingId,
      delay_minutes: nextMin,
      pickup_eta: pickupEtaCandidate ?? prev.pickup_eta ?? null,
      current_eta:
        typeof d.current_eta === "string" && d.current_eta
          ? d.current_eta
          : (prev.current_eta as string | null) ?? null,
    });
  };

  const liveArr = Array.isArray(live) ? live : [];
  const snapArr = Array.isArray(snapshot) ? snapshot : [];
  for (const r of liveArr) bump(r);
  for (const r of snapArr) bump(r);
  return Array.from(map.values());
}

export function flattenCompanyDispatchDelays(apiRows: unknown[]): DispatchDelayFlattenRow[] {
  const input = Array.isArray(apiRows) ? apiRows : [];
  const out: DispatchDelayFlattenRow[] = [];
  for (const entry of input) {
    if (!entry || typeof entry !== "object") continue;
    const d = entry as Record<string, unknown>;
    if (shouldSkipDelayRow(d)) continue;
    const bookingId = delayRowBookingId(d);
    if (bookingId == null) continue;
    const pickupRaw = finiteNum(d.pickup_delay_minutes, 0);
    const dropRaw = finiteNum(d.dropoff_delay_minutes, 0);
    const declared = finiteNum(d.delay_minutes ?? d.delayMinutes, 0);
    const minutes = Math.round(Math.max(declared, pickupRaw, dropRaw));
    if (minutes > 0) {
      out.push({ booking_id: bookingId, delay_minutes: minutes, is_pickup: true });
    }
  }
  return out;
}

/**
 * Pour coller au `dispatchMap` interne du web : même clé plusieurs lignes ⇒ la **dernière** gagne,
 * comme `delays.forEach` dans `DispatchTable.jsx`.
 * (Pickup et dropoff produisent deux lignes : la dernière écrase l’autre — voir {@link pickupDelaysByBookingLastWins}.)
 */
export function delaysByBookingLastWins(rows: DispatchDelayFlattenRow[]): Map<number, number> {
  const map = new Map<number, number>();
  for (const r of rows) {
    map.set(r.booking_id, r.delay_minutes);
  }
  return map;
}

/**
 * Texte sous l’heure : « arrive dans ~N min », aligné `frontend/src/utils/formatPickupEta.js`.
 */
export function pickupArrivalHintFr(eta: string): { text: string; accessibility?: string } | null {
  if (!eta?.trim()) return null;
  const t = new Date(eta.trim());
  if (Number.isNaN(t.getTime())) return null;
  const diffMin = Math.round((t.getTime() - Date.now()) / 60_000);
  const clock = t.toLocaleTimeString("fr-CH", { hour: "2-digit", minute: "2-digit" });

  if (diffMin < -5) {
    return { text: "Pickup dépassé", accessibility: `ETA ${clock}` };
  }
  if (diffMin <= 5) {
    return { text: "Arrivée imminente", accessibility: `Vers ${clock}` };
  }
  return { text: `~${diffMin} min`, accessibility: `Pickup estimé vers ${clock}` };
}

/** Dernière ETA connue par booking — entrées pré-fusion OK. */
export function pickupEtaIsoByBookingId(rows: unknown[]): Map<number, string> {
  const m = new Map<number, string>();
  const input = Array.isArray(rows) ? rows : [];
  for (const entry of input) {
    if (!entry || typeof entry !== "object") continue;
    const d = entry as Record<string, unknown>;
    if (shouldSkipDelayRow(d)) continue;
    const id = delayRowBookingId(d);
    if (id == null) continue;
    const cand =
      (typeof d.pickup_eta === "string" && d.pickup_eta ? d.pickup_eta : "") ||
      (typeof d.current_eta === "string" && d.current_eta ? d.current_eta : "");
    if (cand) m.set(id, cand);
  }
  return m;
}

/** Retard pickup uniquement (liste mobile : même pastille que l’heure de prise en charge). */
export function pickupDelaysByBookingLastWins(rows: DispatchDelayFlattenRow[]): Map<number, number> {
  const map = new Map<number, number>();
  for (const r of rows) {
    if (!r.is_pickup) continue;
    map.set(r.booking_id, r.delay_minutes);
  }
  return map;
}
