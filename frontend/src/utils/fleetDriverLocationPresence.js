/**
 * Présence GPS flotte (parité mobile driverLocationPresence.ts).
 * Seuils via LOCAL_* — ne pas appeler getFreshnessStatus (20/90/300).
 */

import {
  LOCAL_LIVE_MAX_SECONDS,
  LOCAL_RECENT_MAX_SECONDS,
  localAgeSecondsFromRecordedAt,
} from './localDriverLocationFreshness';

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

function ageToPresence(ageSeconds) {
  if (ageSeconds <= LOCAL_LIVE_MAX_SECONDS) return 'live';
  if (ageSeconds <= LOCAL_RECENT_MAX_SECONDS) return 'recent';
  return 'stale';
}

function degradeByAge(serverFresh, ageSeconds) {
  if (ageSeconds == null) return serverFresh;
  const fromAge = ageToPresence(ageSeconds);
  const rank = (p) => {
    if (p === 'live') return 0;
    if (p === 'recent') return 1;
    return 2;
  };
  return rank(fromAge) > rank(serverFresh) ? fromAge : serverFresh;
}

function viewFor(presence, ageSeconds) {
  const countedAsLocated = presence === 'live' || presence === 'recent';
  const isVisuallyStale = presence === 'stale' || presence === 'last_known';
  const showMarker = presence !== 'offline_unknown';
  return { presence, countedAsLocated, isVisuallyStale, showMarker, ageSeconds };
}

function fromTrackingDisplayFallback(tracking, ageSeconds, hasCoords) {
  if (tracking === 'stale') return 'stale';
  if (tracking === 'offline_unknown') {
    return hasCoords ? 'last_known' : 'offline_unknown';
  }
  if (tracking === 'degraded_constrained') {
    if (ageSeconds != null) return ageToPresence(ageSeconds);
    return hasCoords ? 'last_known' : 'offline_unknown';
  }
  if (tracking === 'live' || tracking === 'recent') {
    if (ageSeconds != null) return ageToPresence(ageSeconds);
    return tracking === 'recent' ? 'recent' : 'live';
  }
  if (ageSeconds != null) return ageToPresence(ageSeconds);
  return hasCoords ? 'last_known' : 'offline_unknown';
}

export function resolveDriverLocationPresence(driver, nowMs = Date.now()) {
  if (!driver) return viewFor('offline_unknown', null);
  const hasCoords = hasFiniteCoords(driver);
  const ageSeconds = resolvePresenceAgeSeconds(driver, nowMs);
  const source = normalizeLabel(driver.position_source);
  const locationStatus = normalizeLabel(driver.location_status);
  const tracking = normalizeLabel(driver.tracking_display_status);

  if (!hasCoords) return viewFor('offline_unknown', ageSeconds);
  if (FALLBACK_SOURCES.has(source)) return viewFor('last_known', ageSeconds);
  if (locationStatus === 'offline' || locationStatus === 'last_known') {
    return viewFor('last_known', ageSeconds);
  }
  if (locationStatus === 'stale') return viewFor('stale', ageSeconds);
  if (locationStatus === 'live' || locationStatus === 'recent') {
    return viewFor(degradeByAge(locationStatus, ageSeconds), ageSeconds);
  }
  if (!locationStatus) {
    return viewFor(fromTrackingDisplayFallback(tracking, ageSeconds, hasCoords), ageSeconds);
  }
  if (ageSeconds != null) return viewFor(ageToPresence(ageSeconds), ageSeconds);
  return viewFor('last_known', ageSeconds);
}

function formatRelativeAge(ageSeconds) {
  if (ageSeconds == null || !Number.isFinite(ageSeconds)) return null;
  if (ageSeconds < 60) return `il y a ${Math.max(0, Math.floor(ageSeconds))} s`;
  const minutes = Math.max(1, Math.round(ageSeconds / 60));
  return `il y a ${minutes} min`;
}

export function formatDriverLocationPresenceLabel(view) {
  const relative = formatRelativeAge(view?.ageSeconds);
  switch (view?.presence) {
    case 'live':
      return relative ? `En direct · ${relative}` : 'En direct';
    case 'recent':
      return relative ? `Position récente · ${relative}` : 'Position récente';
    case 'stale':
      return relative ? `Position périmée · ${relative}` : 'Position périmée';
    case 'last_known':
      return 'Dernière position connue';
    case 'offline_unknown':
    default:
      return 'Aucune position disponible';
  }
}
