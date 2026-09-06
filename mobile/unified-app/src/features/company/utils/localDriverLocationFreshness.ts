/**
 * Vieillissement local des positions flotte.
 * Barème : live 0–30s, recent 30–120s, stale >120s.
 * Dégrade location_status sans promotion ; ne touche pas tracking_display_status.
 */

export const LOCAL_LIVE_MAX_SECONDS = 30;
export const LOCAL_RECENT_MAX_SECONDS = 120;

export type LocalLocationFreshnessStatus =
  | "live"
  | "recent"
  | "stale"
  | "offline"
  | "offline_unknown"
  | "last_known";

export function recordedAtEpochMs(recordedAt: string | null | undefined): number | null {
  if (!recordedAt) return null;
  const parsed = Date.parse(recordedAt);
  return Number.isFinite(parsed) ? parsed : null;
}

export function localAgeSecondsFromRecordedAt(
  recordedAt: string | null | undefined,
  nowMs: number = Date.now()
): number | null {
  const epoch = recordedAtEpochMs(recordedAt);
  if (epoch == null) return null;
  return Math.max(0, Math.floor((nowMs - epoch) / 1000));
}

export function resolveLocalLocationFreshnessStatus(
  recordedAt: string | null | undefined,
  nowMs: number = Date.now()
): LocalLocationFreshnessStatus {
  const age = localAgeSecondsFromRecordedAt(recordedAt, nowMs);
  if (age == null) return "offline_unknown";
  if (age <= LOCAL_LIVE_MAX_SECONDS) return "live";
  if (age <= LOCAL_RECENT_MAX_SECONDS) return "recent";
  return "stale";
}

function normalizeStatus(value: string | null | undefined): string {
  return String(value ?? "")
    .trim()
    .toLowerCase();
}

function ageToFreshStatus(age: number): "live" | "recent" | "stale" {
  if (age <= LOCAL_LIVE_MAX_SECONDS) return "live";
  if (age <= LOCAL_RECENT_MAX_SECONDS) return "recent";
  return "stale";
}

export function lastSeenAtOf(driver: {
  recorded_at?: string | null;
  timestamp?: string | null;
}): string | null {
  return driver.recorded_at ?? driver.timestamp ?? null;
}

/**
 * État fonctionnel live → recent → stale à partir de last_seen_at.
 * Ne promeut jamais stale/last_known/offline → live/recent.
 * N’écrit pas last_seen_seconds.
 */
export function resolveFunctionalLocationStatus(
  driver: {
    recorded_at?: string | null;
    timestamp?: string | null;
    last_seen_seconds?: number | null;
    location_status?: string | null;
  },
  nowMs: number = Date.now()
): string | null | undefined {
  const recordedAt = lastSeenAtOf(driver);
  const ageFromTs = localAgeSecondsFromRecordedAt(recordedAt, nowMs);
  const age =
    ageFromTs != null
      ? ageFromTs
      : typeof driver.last_seen_seconds === "number" &&
          Number.isFinite(driver.last_seen_seconds) &&
          driver.last_seen_seconds >= 0
        ? Math.floor(driver.last_seen_seconds)
        : null;

  const current = normalizeStatus(driver.location_status);

  if (current === "stale" || current === "last_known") {
    return current;
  }
  if (current === "offline") {
    return "last_known";
  }
  if (current === "live" || current === "recent") {
    if (age == null) return driver.location_status;
    const fromAge = ageToFreshStatus(age);
    if (current === "recent" && fromAge === "live") return "recent";
    return fromAge;
  }
  if (!current && age != null) return ageToFreshStatus(age);
  return driver.location_status;
}

/**
 * Dégrade location_status au franchissement d’état seulement.
 * last_seen_at (recorded_at) et last_seen_seconds restent ceux de la source.
 */
export function applyLocalLocationFreshness<
  T extends {
    recorded_at?: string | null;
    timestamp?: string | null;
    last_seen_seconds?: number | null;
    location_status?: string | null;
    tracking_display_status?: string | null;
    position_source?: string | null;
  },
>(driver: T, nowMs: number = Date.now()): T {
  const nextLocationStatus = resolveFunctionalLocationStatus(driver, nowMs);
  if (nextLocationStatus === driver.location_status) return driver;
  return {
    ...driver,
    location_status: nextLocationStatus as T["location_status"],
  };
}
