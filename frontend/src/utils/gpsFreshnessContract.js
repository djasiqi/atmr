/**
 * Contrat fraîcheur GPS entreprise — seuils mode-aware (PRESENCE vs mission_live).
 * Ne pas calibrer l'UI sur des seuils mission quand le mobile est en availability_presence.
 */

import { isDevicePipelineAlive } from './devicePipelineUtils';

/** @typedef {'presence' | 'mission_live'} GpsFreshnessMode */

/** Seuils en secondes (bornes supérieures incluses pour live/recent/stale). */
export const GPS_FRESHNESS_THRESHOLDS = {
  presence: {
    live: 60,
    recent: 240,
    stale: 600,
    verify: 600,
  },
  mission_live: {
    live: 45,
    recent: 120,
    stale: 180,
    verify: 300,
  },
};

/** Heartbeat absent > 3 min avant de considérer une vraie perte (hystérésis anti-clignotement). */
export const GPS_OFFLINE_PIPELINE_DEAD_SECONDS = 180;

/**
 * Résout location_mode depuis la SoT serveur (fanout / projection), pas depuis l'âge du fix.
 * Fallback métier uniquement si le backend n'a pas encore propagé le mode.
 * @param {object|null|undefined} driver
 * @returns {string}
 */
export function resolveDriverLocationMode(driver) {
  if (!driver) return 'availability_presence';

  const explicit = driver.location_mode;
  if (explicit != null && String(explicit).trim() !== '') {
    return String(explicit).trim();
  }

  const fromHealth = driver.device_health?.location_mode;
  if (fromHealth != null && String(fromHealth).trim() !== '') {
    return String(fromHealth).trim();
  }

  const status = String(driver.status ?? '').toLowerCase();
  if (
    driver.current_booking_id
    || status === 'busy'
    || status === 'assigned'
    || status === 'assigned_constrained'
  ) {
    return 'mission_live';
  }

  return 'availability_presence';
}

/**
 * @param {string|null|undefined} locationMode
 * @returns {GpsFreshnessMode}
 */
export function normalizeGpsFreshnessMode(locationMode) {
  const m = String(locationMode ?? '').trim().toLowerCase();
  if (m === 'availability_presence' || m === 'passive_last_known') {
    return 'presence';
  }
  return 'mission_live';
}

/**
 * @param {GpsFreshnessMode} mode
 */
export function getGpsFreshnessThresholds(mode) {
  return GPS_FRESHNESS_THRESHOLDS[mode] ?? GPS_FRESHNESS_THRESHOLDS.mission_live;
}

/**
 * Fraîcheur GPS purement par âge — jamais offline (offline = signal session/heartbeat).
 * @returns {'live' | 'recent' | 'stale' | 'verify'}
 */
export function ageToGpsFreshness(ageSeconds, locationMode) {
  const mode = normalizeGpsFreshnessMode(locationMode);
  const t = getGpsFreshnessThresholds(mode);
  const age = Math.max(0, Math.floor(Number(ageSeconds) || 0));
  if (age <= t.live) return 'live';
  if (age <= t.recent) return 'recent';
  if (age <= t.stale) return 'stale';
  return 'verify';
}

/**
 * Vraie perte GPS : pas de pipeline vivant (heartbeat) durable — pas seulement age(position).
 * @param {object|null|undefined} driver
 * @param {number} [nowMs]
 */
export function isGpsPipelineOffline(driver, nowMs = Date.now()) {
  if (!driver) return true;
  return !isDevicePipelineAlive(driver, nowMs);
}

/**
 * Offline affichage : perte pipeline durable ET position trop vieille pour le mode.
 * @param {object|null|undefined} driver
 * @param {number|null} ageSeconds
 * @param {number} [nowMs]
 */
export function shouldDisplayGpsOffline(driver, ageSeconds, nowMs = Date.now()) {
  if (!isGpsPipelineOffline(driver, nowMs)) return false;
  if (ageSeconds == null || !Number.isFinite(ageSeconds)) return true;
  const mode = normalizeGpsFreshnessMode(resolveDriverLocationMode(driver));
  const t = getGpsFreshnessThresholds(mode);
  return ageSeconds > t.verify;
}

/**
 * Résout le statut GPS affiché (séparé de l'état chauffeur En service / Hors service).
 * @returns {'live' | 'recent' | 'stale' | 'verify' | 'offline_unknown' | 'last_known'}
 */
export function resolveGpsDisplayStatus(driver, ageSeconds, nowMs = Date.now()) {
  if (!driver) return 'offline_unknown';
  const source = String(driver.position_source ?? '').trim().toLowerCase();
  if (source === 'db_fallback' || source === 'company_fallback') return 'last_known';

  const backend = String(driver.location_status ?? '').trim().toLowerCase();
  if (backend === 'last_known') return 'last_known';

  if (ageSeconds == null) {
    if (isGpsPipelineOffline(driver, nowMs)) return 'offline_unknown';
    return 'verify';
  }

  const locationMode = resolveDriverLocationMode(driver);
  const fromAge = ageToGpsFreshness(ageSeconds, locationMode);

  if (shouldDisplayGpsOffline(driver, ageSeconds, nowMs)) {
    return 'offline_unknown';
  }

  // Backend offline/stale ne doit pas forcer « hors ligne » si le pipeline est vivant (PRESENCE).
  if (
    (backend === 'offline' || backend === 'stale')
    && !isGpsPipelineOffline(driver, nowMs)
    && fromAge !== 'verify'
  ) {
    return fromAge;
  }

  if (backend === 'live' || backend === 'recent') {
    const rank = { live: 0, recent: 1, stale: 2, verify: 3 };
    const fromBackend = backend === 'live' ? 'live' : 'recent';
    return rank[fromAge] > rank[fromBackend] ? fromAge : fromBackend;
  }

  return fromAge;
}

/**
 * Libellé GPS (couche 2) — indépendant de En service / Hors service.
 */
export function formatGpsFreshnessLabel(status, ageSeconds) {
  const relative =
    ageSeconds != null && Number.isFinite(ageSeconds)
      ? formatRelativeAgeSeconds(ageSeconds)
      : null;

  switch (status) {
    case 'live':
      return relative ? `En direct · ${relative}` : 'En direct';
    case 'recent':
      return relative
        ? `Position mise à jour · ${relative}`
        : 'Position mise à jour';
    case 'stale':
      return relative
        ? `Position ancienne · ${relative}`
        : 'Position ancienne';
    case 'verify':
      return relative ? `GPS à vérifier · ${relative}` : 'GPS à vérifier';
    case 'last_known':
      return 'Dernière position connue';
    case 'offline_unknown':
    default:
      return relative ? `GPS indisponible · ${relative}` : 'GPS indisponible';
  }
}

export function formatRelativeAgeSeconds(ageSeconds) {
  if (ageSeconds == null || !Number.isFinite(ageSeconds)) return null;
  const s = Math.max(0, Math.floor(ageSeconds));
  if (s < 60) return `il y a ${s} s`;
  const minutes = Math.max(1, Math.round(s / 60));
  return `il y a ${minutes} min`;
}
