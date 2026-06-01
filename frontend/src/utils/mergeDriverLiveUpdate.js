/**
 * P5 — Audit fusion socket vs GET canonique (avant staleTime Infinity sur companyDrivers).
 *
 * Couvert par `mergeDriverLiveUpdate` / `mergeOrUpdateDriverInList` :
 * coords (lat/lon), `location_mode`, `last_seen_seconds`, `location_status`, `presence_status`,
 * `status`, `mission_status`, `recorded_at` / `received_at`, `mission_id`, champs upsert minimal
 * (`first_name`, `last_name`, `company_id`, `is_active`) si ligne créée depuis le socket seul.
 *
 * Non couverts (restent sur snapshot HTTP ou invalidations métier ailleurs) : objet `user` imbriqué,
 * véhicule, préférences, disponibilité métier hors événements live, etc.
 */

/** @param {unknown} s */
function parseIsoMs(s) {
  if (s == null || s === '') return null;
  const t = Date.parse(String(s));
  return Number.isFinite(t) ? t : null;
}

/**
 * Horodatage unique pour comparaisons (spec interne) — aligné sur realtimeEventGuard :
 * temps événementiel d’abord (`recorded_at`, `timestamp`), `received_at` en dernier recours.
 * @param {Record<string, unknown> | null | undefined} o
 * @returns {number | null} ms depuis epoch, ou null si non parsable
 */
export function canonicalTimeMs(o) {
  if (!o || typeof o !== 'object') return null;
  return parseIsoMs(o.recorded_at ?? o.timestamp ?? o.ts ?? o.received_at);
}

function pickLatitude(update, driver) {
  return update.lat ?? update.latitude ?? update.current_lat ?? driver.latitude;
}

function pickLongitude(update, driver) {
  // Canonique backend: `lon`; compat transitoire: `lng` / `longitude` / `current_lon`.
  return update.lon ?? update.lng ?? update.longitude ?? update.current_lon ?? driver.longitude;
}

/**
 * Fusionne un événement temps réel (driver_location_update / driver_live_state_update)
 * dans un objet chauffeur déjà chargé depuis l’API canonique.
 */
export function mergeDriverLiveUpdate(driver, update, fromLiveState) {
  const acceptStatus = update?.accept_status;
  const isObservabilityOnly = acceptStatus === 'accepted_observability_only';

  // Observabilité : ne pas régresser la position si canonical_time du point < celui déjà fusionné.
  // Si le chauffeur n’a aucun canonical_time (premier rendu), on accepte le point.
  if (isObservabilityOnly && !fromLiveState) {
    const inc = canonicalTimeMs(update);
    const cur = canonicalTimeMs(driver);
    if (inc != null && cur != null && inc < cur) {
      return { ...driver };
    }
  }

  const latitude = pickLatitude(update, driver);
  const longitude = pickLongitude(update, driver);
  // `driver_location_update` peut porter un statut de fraîcheur plus récent que le snapshot
  // HTTP initial (souvent "offline"). Prioriser systématiquement la valeur socket si fournie.
  const locationStatus = update.location_status ?? driver.location_status ?? null;
  const presenceStatus = update.presence_status ?? driver.presence_status ?? null;
  const status = fromLiveState ? update.status ?? driver.status : driver.status ?? update.status;
  const deviceHealth = update.device_health ?? driver.device_health ?? null;

  return {
    ...driver,
    latitude,
    longitude,
    location_mode: update.location_mode ?? driver.location_mode ?? null,
    last_seen_seconds: update.last_seen_seconds ?? driver.last_seen_seconds ?? null,
    location_status: locationStatus,
    presence_status: presenceStatus,
    device_health: deviceHealth,
    status,
    mission_status: fromLiveState
      ? update.mission_status ?? driver.mission_status ?? null
      : driver.mission_status ?? update.mission_status ?? null,
    recorded_at: update.recorded_at ?? update.timestamp ?? driver.recorded_at ?? null,
    received_at: update.received_at ?? driver.received_at ?? null,
    mission_id: update.mission_id ?? driver.mission_id ?? null,
  };
}

/**
 * Vérifie si le payload socket contient au moins une paire lat/lon exploitable.
 */
export function hasExploitableCoords(payload) {
  if (!payload || typeof payload !== 'object') return false;
  const lat = payload.lat ?? payload.latitude ?? payload.current_lat;
  const lon = payload.lon ?? payload.lng ?? payload.longitude ?? payload.current_lon;
  if (lat == null || lon == null) return false;
  const la = typeof lat === 'number' ? lat : parseFloat(String(lat));
  const lo = typeof lon === 'number' ? lon : parseFloat(String(lon));
  return Number.isFinite(la) && Number.isFinite(lo);
}

/**
 * Met à jour un chauffeur dans la liste, ou l’ajoute s’il n’existe pas encore (upsert socket).
 * Évite que la carte reste figée quand le fanout arrive avant / sans ligne API pour ce chauffeur.
 */
export function mergeOrUpdateDriverInList(prev, payload, fromLiveState, companyId) {
  const driverId = Number(payload?.driver_id ?? payload?.id);
  if (!Number.isFinite(driverId)) return prev;
  if (!Array.isArray(prev)) return prev;

  const index = prev.findIndex((d) => Number(d.id) === driverId);
  if (index >= 0) {
    const next = [...prev];
    next[index] = mergeDriverLiveUpdate(next[index], payload, fromLiveState);
    return next;
  }

  if (!hasExploitableCoords(payload)) return prev;

  const base = {
    id: driverId,
    company_id: companyId ?? payload.company_id ?? null,
    first_name: payload.first_name ?? '',
    last_name: payload.last_name ?? '',
    is_active: true,
  };
  return [...prev, mergeDriverLiveUpdate(base, payload, fromLiveState)];
}
