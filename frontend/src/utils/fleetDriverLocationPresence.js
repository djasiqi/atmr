/**
 * Présence GPS flotte — seuils mode-aware (PRESENCE vs mission_live).
 */

import {
  ageToGpsFreshness,
  resolveGpsDisplayStatus,
  formatGpsFreshnessLabel as formatGpsLabelFromContract,
  formatRelativeAgeSeconds,
  resolveDriverLocationMode,
} from './gpsFreshnessContract';
import { localAgeSecondsFromRecordedAt } from './localDriverLocationFreshness';

// Re-export for tests
export { formatRelativeAgeSeconds };

const FALLBACK_SOURCES = new Set(['db_fallback', 'company_fallback']);

function hasFiniteCoords(driver) {
  if (driver?.latitude == null && driver?.lat == null) return false;
  if (driver?.longitude == null && driver?.lon == null && driver?.lng == null) return false;
  const lat = Number(driver?.latitude ?? driver?.lat);
  const lon = Number(driver?.longitude ?? driver?.lon ?? driver?.lng);
  return Number.isFinite(lat) && Number.isFinite(lon);
}

function normalizeLabel(value) {
  return String(value ?? '')
    .trim()
    .toLowerCase();
}

/**
 * Priorité âge : recorded_at → timestamp → last_seen_seconds → inconnu.
 */
export function resolvePresenceAgeSeconds(driver, nowMs = Date.now()) {
  const fromRecorded = localAgeSecondsFromRecordedAt(driver?.recorded_at, nowMs);
  if (fromRecorded != null) return fromRecorded;
  const fromTimestamp = localAgeSecondsFromRecordedAt(driver?.timestamp, nowMs);
  if (fromTimestamp != null) return fromTimestamp;
  const lastSeen = driver?.last_seen_seconds;
  if (typeof lastSeen === 'number' && Number.isFinite(lastSeen) && lastSeen >= 0) {
    return Math.floor(lastSeen);
  }
  return null;
}

function ageToPresence(ageSeconds, locationMode) {
  return ageToGpsFreshness(ageSeconds, locationMode);
}

function degradeByAge(serverFresh, ageSeconds, locationMode) {
  if (ageSeconds == null) return serverFresh;
  const fromAge = ageToPresence(ageSeconds, locationMode);
  const rank = { live: 0, recent: 1, stale: 2, verify: 3 };
  const serverKey = serverFresh === 'live' || serverFresh === 'recent' ? serverFresh : 'stale';
  return rank[fromAge] > rank[serverKey] ? fromAge : serverFresh;
}

function viewFor(presence, ageSeconds) {
  const countedAsLocated = presence === 'live' || presence === 'recent';
  const isVisuallyStale =
    presence === 'stale' || presence === 'verify' || presence === 'last_known';
  const showMarker = presence !== 'offline_unknown';
  return { presence, countedAsLocated, isVisuallyStale, showMarker, ageSeconds };
}

function fromTrackingDisplayFallback(tracking, ageSeconds, hasCoords, locationMode) {
  if (tracking === 'stale') return 'stale';
  if (tracking === 'offline_unknown') {
    return hasCoords ? 'last_known' : 'offline_unknown';
  }
  if (tracking === 'degraded_constrained') {
    if (ageSeconds != null) return ageToPresence(ageSeconds, locationMode);
    return hasCoords ? 'last_known' : 'offline_unknown';
  }
  if (tracking === 'live' || tracking === 'recent') {
    if (ageSeconds != null) return ageToPresence(ageSeconds, locationMode);
    return tracking === 'recent' ? 'recent' : 'live';
  }
  if (ageSeconds != null) return ageToPresence(ageSeconds, locationMode);
  return hasCoords ? 'last_known' : 'offline_unknown';
}

export function resolveDriverLocationPresence(driver, nowMs = Date.now()) {
  if (!driver) return viewFor('offline_unknown', null);

  const hasCoords = hasFiniteCoords(driver);
  const ageSeconds = resolvePresenceAgeSeconds(driver, nowMs);
  const locationMode = resolveDriverLocationMode(driver);
  const source = normalizeLabel(driver.position_source);

  if (!hasCoords) return viewFor('offline_unknown', ageSeconds);
  if (FALLBACK_SOURCES.has(source)) return viewFor('last_known', ageSeconds);

  const displayStatus = resolveGpsDisplayStatus(driver, ageSeconds, nowMs);
  if (displayStatus === 'offline_unknown') {
    return viewFor('offline_unknown', ageSeconds);
  }
  if (displayStatus === 'last_known') {
    return viewFor('last_known', ageSeconds);
  }

  const locationStatus = normalizeLabel(driver.location_status);
  const tracking = normalizeLabel(driver.tracking_display_status);

  if (locationStatus === 'offline' || locationStatus === 'last_known') {
    return viewFor(displayStatus === 'verify' ? 'verify' : 'last_known', ageSeconds);
  }
  if (locationStatus === 'stale') {
    return viewFor(degradeByAge('stale', ageSeconds, locationMode), ageSeconds);
  }
  if (locationStatus === 'live' || locationStatus === 'recent') {
    return viewFor(degradeByAge(locationStatus, ageSeconds, locationMode), ageSeconds);
  }
  if (!locationStatus) {
    return viewFor(
      fromTrackingDisplayFallback(tracking, ageSeconds, hasCoords, locationMode),
      ageSeconds
    );
  }
  if (ageSeconds != null) {
    return viewFor(ageToPresence(ageSeconds, locationMode), ageSeconds);
  }
  return viewFor('last_known', ageSeconds);
}

export function formatDriverLocationPresenceLabel(view) {
  return formatGpsLabelFromContract(view?.presence, view?.ageSeconds);
}
