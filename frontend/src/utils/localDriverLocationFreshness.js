/**
 * Vieillissement local des positions flotte (parité mobile).
 * live 0–30s, recent 30–120s, stale >120s, timestamp invalide → offline_unknown.
 */

export const LOCAL_LIVE_MAX_SECONDS = 30;
export const LOCAL_RECENT_MAX_SECONDS = 120;

export function recordedAtEpochMs(recordedAt) {
  if (!recordedAt) return null;
  const parsed = Date.parse(recordedAt);
  return Number.isFinite(parsed) ? parsed : null;
}

export function localAgeSecondsFromRecordedAt(recordedAt, nowMs = Date.now()) {
  const epoch = recordedAtEpochMs(recordedAt);
  if (epoch == null) return null;
  return Math.max(0, Math.floor((nowMs - epoch) / 1000));
}

export function resolveLocalLocationFreshnessStatus(recordedAt, nowMs = Date.now()) {
  const age = localAgeSecondsFromRecordedAt(recordedAt, nowMs);
  if (age == null) return 'offline_unknown';
  if (age <= LOCAL_LIVE_MAX_SECONDS) return 'live';
  if (age <= LOCAL_RECENT_MAX_SECONDS) return 'recent';
  return 'stale';
}

export function applyLocalLocationFreshness(driver, nowMs = Date.now()) {
  if (!driver || typeof driver !== 'object') return driver;
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
