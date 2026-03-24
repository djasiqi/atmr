/** @param {unknown} s */
function parseIsoMs(s) {
  if (s == null || s === '') return null;
  const t = Date.parse(String(s));
  return Number.isFinite(t) ? t : null;
}

/**
 * Horodatage unique pour comparaisons (spec interne) — toujours la même chaîne :
 * `received_at` > `recorded_at` > `timestamp` (évite de mélanger des champs non comparables).
 * @param {Record<string, unknown> | null | undefined} o
 * @returns {number | null} ms depuis epoch, ou null si non parsable
 */
export function canonicalTimeMs(o) {
  if (!o || typeof o !== 'object') return null;
  return parseIsoMs(o.received_at ?? o.recorded_at ?? o.timestamp);
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

  const latitude = update.lat ?? update.latitude ?? update.current_lat ?? driver.latitude;
  const longitude =
    update.lon ?? update.lng ?? update.longitude ?? update.current_lon ?? driver.longitude;
  const locationStatus = fromLiveState
    ? update.location_status ?? driver.location_status ?? null
    : driver.location_status ?? update.location_status ?? null;
  const presenceStatus = fromLiveState
    ? update.presence_status ?? driver.presence_status ?? null
    : driver.presence_status ?? update.presence_status ?? null;
  const status = fromLiveState ? update.status ?? driver.status : driver.status ?? update.status;

  return {
    ...driver,
    latitude,
    longitude,
    location_mode: update.location_mode ?? driver.location_mode ?? null,
    last_seen_seconds: update.last_seen_seconds ?? driver.last_seen_seconds ?? null,
    location_status: locationStatus,
    presence_status: presenceStatus,
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
