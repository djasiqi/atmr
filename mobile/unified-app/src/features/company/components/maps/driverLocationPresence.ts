/**
 * Présence GPS flotte — machine d’état canonique (5 états, pas d’offline).
 * Axes séparés : métier / GPS / device (tracking_display_status = diagnostic).
 */

import {
  LOCAL_LIVE_MAX_SECONDS,
  LOCAL_RECENT_MAX_SECONDS,
  localAgeSecondsFromRecordedAt,
} from "../../utils/localDriverLocationFreshness";

export type DriverLocationPresence =
  | "live"
  | "recent"
  | "stale"
  | "last_known"
  | "offline_unknown";

/** Entrée avant couche spatiale — lat/lon optionnels (roster sans GPS). */
export type FleetDriverPresenceInput = {
  driver_id: number;
  latitude?: number | null;
  longitude?: number | null;
  recorded_at?: string | null;
  timestamp?: string | null;
  last_seen_seconds?: number | null;
  location_status?: string | null;
  tracking_display_status?: string | null;
  position_source?: string | null;
  status?: string | null;
  presence_status?: string | null;
  device_health?: {
    constraint_reason?: string | null;
    battery_optimized?: boolean | null;
    tracking_active?: boolean | null;
  } | null;
};

export type DriverLocationPresenceView = {
  presence: DriverLocationPresence;
  countedAsLocated: boolean;
  isVisuallyStale: boolean;
  showMarker: boolean;
  ageSeconds: number | null;
};

const FALLBACK_SOURCES = new Set(["db_fallback", "company_fallback"]);

function hasFiniteCoords(driver: FleetDriverPresenceInput): boolean {
  if (driver.latitude == null || driver.longitude == null) return false;
  const lat = Number(driver.latitude);
  const lon = Number(driver.longitude);
  return Number.isFinite(lat) && Number.isFinite(lon);
}


/**
 * Priorité âge : recorded_at → timestamp → last_seen_seconds → inconnu.
 */
export function resolvePresenceAgeSeconds(
  driver: FleetDriverPresenceInput,
  nowMs: number = Date.now()
): number | null {
  const fromRecorded = localAgeSecondsFromRecordedAt(driver.recorded_at, nowMs);
  if (fromRecorded != null) return fromRecorded;
  const fromTimestamp = localAgeSecondsFromRecordedAt(driver.timestamp, nowMs);
  if (fromTimestamp != null) return fromTimestamp;
  const lastSeen = driver.last_seen_seconds;
  if (typeof lastSeen === "number" && Number.isFinite(lastSeen) && lastSeen >= 0) {
    return Math.floor(lastSeen);
  }
  return null;
}

function ageToPresence(ageSeconds: number): DriverLocationPresence {
  if (ageSeconds <= LOCAL_LIVE_MAX_SECONDS) return "live";
  if (ageSeconds <= LOCAL_RECENT_MAX_SECONDS) return "recent";
  return "stale";
}

/**
 * Dégrade live/recent selon l’âge ; ne promeut jamais un statut déjà dégradé.
 */
function degradeByAge(
  serverFresh: "live" | "recent",
  ageSeconds: number | null
): DriverLocationPresence {
  if (ageSeconds == null) return serverFresh;
  const fromAge = ageToPresence(ageSeconds);
  const rank = (p: DriverLocationPresence): number => {
    if (p === "live") return 0;
    if (p === "recent") return 1;
    return 2;
  };
  return rank(fromAge) > rank(serverFresh) ? fromAge : serverFresh;
}

function normalizeLabel(value: string | null | undefined): string {
  return String(value ?? "")
    .trim()
    .toLowerCase();
}

function viewFor(presence: DriverLocationPresence, ageSeconds: number | null): DriverLocationPresenceView {
  const countedAsLocated = presence === "live" || presence === "recent";
  const isVisuallyStale = presence === "stale" || presence === "last_known";
  const showMarker = presence !== "offline_unknown";
  return { presence, countedAsLocated, isVisuallyStale, showMarker, ageSeconds };
}

/**
 * Fallback déterministe quand location_status est absent.
 * degraded_constrained n’est jamais un état de fraîcheur.
 */
function fromTrackingDisplayFallback(
  tracking: string,
  ageSeconds: number | null,
  hasCoords: boolean
): DriverLocationPresence {
  if (tracking === "stale") return "stale";
  if (tracking === "offline_unknown") {
    return hasCoords ? "last_known" : "offline_unknown";
  }
  if (tracking === "degraded_constrained") {
    if (ageSeconds != null) return ageToPresence(ageSeconds);
    return hasCoords ? "last_known" : "offline_unknown";
  }
  if (tracking === "live" || tracking === "recent") {
    if (ageSeconds != null) return ageToPresence(ageSeconds);
    return tracking === "recent" ? "recent" : "live";
  }
  if (ageSeconds != null) return ageToPresence(ageSeconds);
  return hasCoords ? "last_known" : "offline_unknown";
}

export function resolveDriverLocationPresence(
  driver: FleetDriverPresenceInput,
  nowMs: number = Date.now()
): DriverLocationPresenceView {
  const hasCoords = hasFiniteCoords(driver);
  const ageSeconds = resolvePresenceAgeSeconds(driver, nowMs);
  const source = normalizeLabel(driver.position_source);
  const locationStatus = normalizeLabel(driver.location_status);
  const tracking = normalizeLabel(driver.tracking_display_status);

  if (!hasCoords) {
    return viewFor("offline_unknown", ageSeconds);
  }

  if (FALLBACK_SOURCES.has(source)) {
    return viewFor("last_known", ageSeconds);
  }

  if (locationStatus === "offline" || locationStatus === "last_known") {
    return viewFor("last_known", ageSeconds);
  }

  if (locationStatus === "stale") {
    return viewFor("stale", ageSeconds);
  }

  if (locationStatus === "live" || locationStatus === "recent") {
    return viewFor(degradeByAge(locationStatus, ageSeconds), ageSeconds);
  }

  // location_status absent / inconnu → fallback tracking_display
  if (!locationStatus) {
    return viewFor(fromTrackingDisplayFallback(tracking, ageSeconds, hasCoords), ageSeconds);
  }

  // Autres labels location_status inattendus : âge si possible, sinon last_known avec coords
  if (ageSeconds != null) {
    return viewFor(ageToPresence(ageSeconds), ageSeconds);
  }
  return viewFor("last_known", ageSeconds);
}

function formatRelativeAge(ageSeconds: number | null): string | null {
  if (ageSeconds == null || !Number.isFinite(ageSeconds)) return null;
  if (ageSeconds < 60) return `il y a ${Math.max(0, Math.floor(ageSeconds))} s`;
  const minutes = Math.max(1, Math.round(ageSeconds / 60));
  return `il y a ${minutes} min`;
}

/** Libellé GPS unique pour toutes les surfaces flotte. */
export function formatDriverLocationPresenceLabel(
  view: Pick<DriverLocationPresenceView, "presence" | "ageSeconds">
): string {
  const relative = formatRelativeAge(view.ageSeconds);
  switch (view.presence) {
    case "live":
      return relative ? `En direct · ${relative}` : "En direct";
    case "recent":
      return relative ? `Position récente · ${relative}` : "Position récente";
    case "stale":
      return relative ? `Position périmée · ${relative}` : "Position périmée";
    case "last_known":
      return "Dernière position connue";
    case "offline_unknown":
      return "Aucune position disponible";
    default:
      return "Aucune position disponible";
  }
}

/** Filtre GPS distinct du statut métier. */
export type FleetGpsFilter = "all" | "live" | "not_recent";

export function matchesFleetGpsFilter(
  presence: DriverLocationPresence,
  filter: FleetGpsFilter
): boolean {
  if (filter === "all") return true;
  if (filter === "live") return presence === "live" || presence === "recent";
  return presence === "stale" || presence === "last_known" || presence === "offline_unknown";
}
