/**
 * Projections chauffeurs par surface consommatrice — évite de propager
 * l'objet driver complet à la carte quand seuls lat/lng changent côté métier.
 */

import { getDriverStatus, isDevicePipelineAlive, isDriverOffDuty } from './mapUtils';
import { resolveDriverLocationPresence } from './fleetDriverLocationPresence';

/** Couleur marqueur carte pour chauffeur en mode batterie restreinte (Tailwind orange-500). */
export const CONSTRAINED_MARKER_COLOR = '#f97316';

/** Couleur marqueur GPS hors ligne / last_known (gris slate). */
export const GPS_OFFLINE_MARKER_COLOR = '#94A3B8';

function numOrNull(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

/**
 * Statuts driver/presence signalant un problème d'optimisation batterie côté app chauffeur.
 * Le backend peut indiquer la contrainte via `presence_status === 'degraded_constrained'`
 * et/ou `status in {'assigned_constrained', 'available_constrained'}`. On accepte les deux
 * formes pour rester compatible avec un déploiement progressif backend.
 */
const CONSTRAINED_DRIVER_STATUSES = new Set([
  'assigned_constrained',
  'available_constrained',
]);

const FRESH_GPS = new Set(['live', 'recent']);
const STALE_GPS = new Set(['stale', 'verify']);

/**
 * Détecte si un chauffeur est en mode "contraint" (app figée / position figée à cause
 * d'une optimisation batterie de l'OS). Signal métier réutilisé par la carte, la liste
 * chauffeur et la bannière dashboard. Renvoie `false` quand les nouveaux champs ne sont
 * pas encore propagés par le backend (compat).
 */
export function isDriverConstrained(driver) {
  if (!driver) return false;
  const presence = String(driver.presence_status || '').toLowerCase();
  if (presence === 'degraded_constrained') return true;
  const tracking = String(driver.tracking_display_status || '').toLowerCase();
  if (tracking === 'degraded_constrained') return true;
  const status = String(driver.status || '').toLowerCase();
  return CONSTRAINED_DRIVER_STATUSES.has(status);
}

/** Raison "machine" remontée par device_health, ou null si non fournie. */
export function getDriverConstraintReason(driver) {
  if (!driver) return null;
  const reason = driver?.device_health?.constraint_reason;
  if (reason == null || reason === '') return null;
  return String(reason);
}

/**
 * Résout la fraîcheur GPS flotte (seuils 30/120, dégrade sans promotion).
 * Ne passe pas par getFreshnessStatus (20/90/300).
 * @param {object} driver
 */
export function resolveGpsFreshness(driver) {
  if (!driver) return 'offline_unknown';
  return resolveDriverLocationPresence(driver).presence;
}

/**
 * Projection duale métier / GPS pour la carte.
 * @param {object} driver
 * @param {{ isFallback?: boolean }} [opts]
 * @returns {{
 *   businessStatus: string,
 *   gpsFreshness: string,
 *   positionSource: string|null,
 *   visualTreatment: string,
 *   lastPositionAt: string|null,
 *   visualStatus: string,
 * }}
 */
export function resolveDriverMapProjection(driver, { isFallback = false } = {}) {
  const businessStatus = isFallback ? 'offline' : getDriverStatus(driver);
  const positionSource = isFallback
    ? 'company_fallback'
    : (driver?.position_source != null ? String(driver.position_source) : null);
  const gpsFreshness = isFallback ? 'company_fallback' : resolveGpsFreshness(driver);
  const lastPositionAt =
    driver?.recorded_at
    ?? driver?.timestamp
    ?? driver?.received_at
    ?? null;
  const constrained = !isFallback && isDriverConstrained(driver);

  let visualTreatment = 'business';
  let visualStatus = businessStatus;

  if (isDriverOffDuty(driver)) {
    visualTreatment = 'business';
    visualStatus = 'off_duty';
  } else if (isFallback || positionSource === 'company_fallback' || gpsFreshness === 'company_fallback') {
    visualTreatment = 'company_fallback';
    visualStatus = 'offline';
  } else if (
    gpsFreshness === 'offline'
    || gpsFreshness === 'offline_unknown'
  ) {
    const pipelineAlive = isDevicePipelineAlive(driver);

    if (pipelineAlive) {
      visualTreatment = 'gps_stale';
      visualStatus = businessStatus === 'offline' ? 'available' : businessStatus;
    } else {
      visualTreatment = 'gps_offline';
      visualStatus = 'offline';
    }
  } else if (
    gpsFreshness === 'stale'
    || gpsFreshness === 'verify'
    || gpsFreshness === 'last_known'
    || positionSource === 'db_fallback'
  ) {
    // C2 situation 4 : stale/last_known ≠ offline
    visualTreatment = constrained ? 'gps_stale_constrained' : 'gps_stale';
    visualStatus = businessStatus === 'offline' ? 'available' : businessStatus;
  } else if (constrained && FRESH_GPS.has(gpsFreshness)) {
    visualTreatment = 'constrained';
    visualStatus = 'constrained';
  } else if (FRESH_GPS.has(gpsFreshness)) {
    visualTreatment = 'business';
    visualStatus = businessStatus;
  } else {
    visualTreatment = 'gps_offline';
    visualStatus = 'offline';
  }

  return {
    businessStatus,
    gpsFreshness,
    positionSource,
    visualTreatment,
    lastPositionAt: lastPositionAt != null ? String(lastPositionAt) : null,
    visualStatus,
  };
}

/**
 * Statut visuel carte (clé couleur marqueur).
 * GPS non-live / db_fallback / offline dominent le statut métier.
 * @param {object} driver
 * @param {{ isFallback?: boolean }} [opts]
 */
export function resolveDriverMapVisualStatus(driver, { isFallback = false } = {}) {
  return resolveDriverMapProjection(driver, { isFallback }).visualStatus;
}

/**
 * Couleur hex du marqueur carte pour un statut visuel (sans blend stale).
 * @param {string} visualStatus
 * @param {Record<string, string>} [statusColors]
 */
export function resolveDriverMapMarkerColor(visualStatus, statusColors = {}) {
  if (visualStatus === 'constrained') return CONSTRAINED_MARKER_COLOR;
  if (visualStatus === 'off_duty' && statusColors.off_duty) return statusColors.off_duty;
  if (visualStatus === 'offline' && statusColors.offline) return statusColors.offline;
  if (visualStatus === 'available' && statusColors.available) return statusColors.available;
  return statusColors[visualStatus] ?? statusColors.available ?? CONSTRAINED_MARKER_COLOR;
}

/** True si la position ne doit pas compter comme « en direct » (live|recent). */
export function isNonLiveGpsPosition(driver, { isFallback = false } = {}) {
  if (isFallback) return true;
  return !resolveDriverLocationPresence(driver).countedAsLocated;
}

/** Champs minimaux pour DriverLiveMap (position + statut visuel + fraîcheur GPS). */
export function projectDriverForMap(driver) {
  if (!driver || driver.id == null) return null;
  const projection = resolveDriverMapProjection(driver);
  return {
    id: driver.id,
    latitude: numOrNull(driver.latitude ?? driver.lat),
    longitude: numOrNull(driver.longitude ?? driver.lng ?? driver.lon),
    accuracy: numOrNull(driver.accuracy),
    speed: numOrNull(driver.speed),
    heading: numOrNull(driver.heading),
    location_mode: driver.location_mode ?? null,
    status: driver.status ?? null,
    location_status: driver.location_status ?? null,
    presence_status: driver.presence_status ?? null,
    tracking_display_status: driver.tracking_display_status ?? null,
    position_source: driver.position_source ?? null,
    offline_reason: driver.offline_reason ?? null,
    recorded_at: driver.recorded_at ?? null,
    received_at: driver.received_at ?? null,
    timestamp: driver.timestamp ?? null,
    device_health: driver.device_health ?? null,
    last_seen_seconds: driver.last_seen_seconds ?? null,
    is_active: driver.is_active ?? true,
    full_name: driver.full_name ?? null,
    first_name: driver.first_name ?? null,
    last_name: driver.last_name ?? null,
    username: driver.username ?? null,
    email: driver.email ?? null,
    phone: driver.phone ?? null,
    client_short: driver.client_short ?? null,
    current_booking_id: driver.current_booking_id ?? null,
    vehicle_name: driver.vehicle_name ?? null,
    vehicle_model: driver.vehicle_model ?? null,
    businessStatus: projection.businessStatus,
    gpsFreshness: projection.gpsFreshness,
    positionSource: projection.positionSource,
    visualTreatment: projection.visualTreatment,
    lastPositionAt: projection.lastPositionAt,
  };
}

/** Champs utiles pour table / modals administratives. */
export function projectDriverForTable(driver) {
  if (!driver || driver.id == null) return null;
  return {
    id: driver.id,
    full_name: driver.full_name ?? null,
    first_name: driver.first_name ?? null,
    last_name: driver.last_name ?? null,
    username: driver.username ?? null,
    email: driver.email ?? null,
    phone: driver.phone ?? null,
    is_active: driver.is_active ?? true,
    status: driver.status ?? null,
    driver_type: driver.driver_type ?? null,
    vehicle_name: driver.vehicle_name ?? null,
    vehicle_model: driver.vehicle_model ?? null,
    latitude: numOrNull(driver.latitude ?? driver.lat),
    longitude: numOrNull(driver.longitude ?? driver.lng ?? driver.lon),
    location_status: driver.location_status ?? null,
    presence_status: driver.presence_status ?? null,
    device_health: driver.device_health ?? null,
    last_seen_seconds: driver.last_seen_seconds ?? null,
  };
}

export function projectDriversForMap(drivers) {
  if (!Array.isArray(drivers)) return [];
  return drivers.map(projectDriverForMap).filter(Boolean);
}

export function projectDriversForTable(drivers) {
  if (!Array.isArray(drivers)) return [];
  return drivers.map(projectDriverForTable).filter(Boolean);
}

/**
 * Empreinte structurelle du set visible (ids ordonnés) — fitBounds uniquement si elle change.
 * @param {Array<{ id: number|string }>} drivers
 * @param {string} [searchQuery]
 */
export function buildDriverStructuralSetKey(drivers, searchQuery = '') {
  if (!Array.isArray(drivers) || drivers.length === 0) {
    return `empty:${String(searchQuery).trim().toLowerCase()}`;
  }
  const ids = drivers
    .map((d) => d?.id)
    .filter((id) => id != null)
    .sort((a, b) => Number(a) - Number(b))
    .join(',');
  return `${ids}|q:${String(searchQuery).trim().toLowerCase()}`;
}

/** Compare deux positions marker (tolérance ~1 m). */
export function isSameMarkerPosition(a, b) {
  if (!a || !b) return false;
  const latA = Number(a.lat ?? a.latitude);
  const lngA = Number(a.lng ?? a.longitude);
  const latB = Number(b.lat ?? b.latitude);
  const lngB = Number(b.lng ?? b.longitude);
  if (!Number.isFinite(latA) || !Number.isFinite(lngA) || !Number.isFinite(latB) || !Number.isFinite(lngB)) {
    return false;
  }
  return Math.abs(latA - latB) < 0.00001 && Math.abs(lngA - lngB) < 0.00001;
}
