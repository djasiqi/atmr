/**
 * Vieillissement local des positions flotte — écrase un statut serveur `live` figé.
 * Barème : live 0–30s, recent 30–120s, stale/offline >120s, timestamp invalide → offline_unknown.
 */

export const LOCAL_LIVE_MAX_SECONDS = 30;
export const LOCAL_RECENT_MAX_SECONDS = 120;

export type LocalLocationFreshnessStatus =
  | "live"
  | "recent"
  | "stale"
  | "offline"
  | "offline_unknown";

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

export function applyLocalLocationFreshness<
  T extends {
    recorded_at?: string | null;
    timestamp?: string | null;
    last_seen_seconds?: number | null;
    location_status?: string | null;
    tracking_display_status?: string | null;
  },
>(driver: T, nowMs: number = Date.now()): T {
  const recordedAt = driver.recorded_at ?? driver.timestamp ?? null;
  const age = localAgeSecondsFromRecordedAt(recordedAt, nowMs);
  const status = resolveLocalLocationFreshnessStatus(recordedAt, nowMs);
  return {
    ...driver,
    last_seen_seconds: age,
    location_status: status,
    tracking_display_status: status,
  };
}
