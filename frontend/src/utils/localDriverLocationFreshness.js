/**
 * Vieillissement local des positions flotte.
 * Seuils mode-aware (PRESENCE vs mission_live) — voir gpsFreshnessContract.js.
 */

import {
  ageToGpsFreshness,
  getGpsFreshnessThresholds,
  normalizeGpsFreshnessMode,
  resolveDriverLocationMode,
} from './gpsFreshnessContract';

/** @deprecated Préférer getGpsFreshnessThresholds('presence').live */
export const LOCAL_LIVE_MAX_SECONDS = getGpsFreshnessThresholds('presence').live;

/** @deprecated Préférer getGpsFreshnessThresholds('presence').recent */
export const LOCAL_RECENT_MAX_SECONDS = getGpsFreshnessThresholds('presence').recent;

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

/**
 * @param {string|null|undefined} recordedAt
 * @param {number} [nowMs]
 * @param {string|null|undefined} [locationMode]
 */
export function resolveLocalLocationFreshnessStatus(
  recordedAt,
  nowMs = Date.now(),
  locationMode = null
) {
  const age = localAgeSecondsFromRecordedAt(recordedAt, nowMs);
  if (age == null) return 'offline_unknown';
  return ageToGpsFreshness(age, locationMode);
}

function normalizeStatus(value) {
  return String(value ?? '')
    .trim()
    .toLowerCase();
}

function ageToFreshStatus(age, locationMode) {
  return ageToGpsFreshness(age, locationMode);
}

const FRESHNESS_RANK = { live: 0, recent: 1, stale: 2, verify: 3 };

/**
 * Recalcule last_seen_seconds ; peut dégrader location_status live/recent.
 * Ne promeut jamais stale/last_known/offline.
 * Ne doit plus écraser tracking_display_status ni position_source.
 */
export function applyLocalLocationFreshness(driver, nowMs = Date.now()) {
  if (!driver || typeof driver !== 'object') return driver;
  const recordedAt = driver.recorded_at ?? driver.timestamp ?? null;
  const locationMode = resolveDriverLocationMode(driver);
  const ageFromTs = localAgeSecondsFromRecordedAt(recordedAt, nowMs);
  const age =
    ageFromTs != null
      ? ageFromTs
      : typeof driver.last_seen_seconds === 'number' &&
          Number.isFinite(driver.last_seen_seconds) &&
          driver.last_seen_seconds >= 0
        ? Math.floor(driver.last_seen_seconds)
        : null;

  const current = normalizeStatus(driver.location_status);
  let nextLocationStatus = driver.location_status;

  if (current === 'stale' || current === 'last_known') {
    nextLocationStatus = current;
  } else if (current === 'offline') {
    nextLocationStatus = 'last_known';
  } else if (current === 'live' || current === 'recent') {
    if (age != null) {
      const fromAge = ageToFreshStatus(age, locationMode);
      const currentKey = current === 'live' ? 'live' : 'recent';
      if (
        FRESHNESS_RANK[fromAge] > FRESHNESS_RANK[currentKey]
        || (current === 'recent' && fromAge === 'live')
      ) {
        nextLocationStatus = fromAge === 'live' && current === 'recent' ? 'recent' : fromAge;
      }
    }
  } else if (!current && age != null) {
    nextLocationStatus = ageToFreshStatus(age, locationMode);
  }

  return {
    ...driver,
    last_seen_seconds: age,
    location_status: nextLocationStatus,
  };
}

export { normalizeGpsFreshnessMode, getGpsFreshnessThresholds, ageToGpsFreshness };
