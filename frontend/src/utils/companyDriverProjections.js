/**
 * Projections chauffeurs par surface consommatrice — évite de propager
 * l'objet driver complet à la carte quand seuls lat/lng changent côté métier.
 */

function numOrNull(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

/** Champs minimaux pour DriverLiveMap (position + statut visuel). */
export function projectDriverForMap(driver) {
  if (!driver || driver.id == null) return null;
  return {
    id: driver.id,
    latitude: numOrNull(driver.latitude ?? driver.lat),
    longitude: numOrNull(driver.longitude ?? driver.lng ?? driver.lon),
    status: driver.status ?? null,
    location_status: driver.location_status ?? null,
    last_seen_seconds: driver.last_seen_seconds ?? null,
    is_active: driver.is_active ?? true,
    full_name: driver.full_name ?? null,
    first_name: driver.first_name ?? null,
    last_name: driver.last_name ?? null,
    username: driver.username ?? null,
    email: driver.email ?? null,
    client_short: driver.client_short ?? null,
    current_booking_id: driver.current_booking_id ?? null,
    vehicle_name: driver.vehicle_name ?? null,
    vehicle_model: driver.vehicle_model ?? null,
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
