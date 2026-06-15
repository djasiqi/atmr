import apiClient from './apiClient';

const ROAD_FACTOR = 1.4;
const FALLBACK_AVG_SPEED_KMH = 30;
const EARTH_RADIUS_KM = 6371;
const GEOCODE_TIMEOUT_MS = 8000;
const DIRECTIONS_TIMEOUT_MS = 12000;

const toRad = (deg) => (deg * Math.PI) / 180;

/** Coordonnée valide (rejette 0 — valeur fréquente pour « absent »). */
export const toCoord = (value) => {
  const n = Number(value);
  if (!Number.isFinite(n) || n === 0) return null;
  return n;
};

const hasValidEndpoints = (endpoints) => {
  const { pickup_lat, pickup_lng, dropoff_lat, dropoff_lng } = endpoints || {};
  return (
    pickup_lat != null && pickup_lng != null
    && dropoff_lat != null && dropoff_lng != null
  );
};

const sortedLegs = (req) => (
  Array.isArray(req?.legs)
    ? [...req.legs].sort((a, b) => (a.sequence_index ?? 0) - (b.sequence_index ?? 0))
    : []
);

/**
 * Trajet aller (1re étape) : coordonnées connues + adresses pour géocodage.
 */
export function resolveOutboundRoute(req) {
  const legs = sortedLegs(req);
  if (legs.length > 0) {
    const first = legs[0];
    return {
      pickup_lat: toCoord(first.pickup_lat ?? req?.pickup_lat),
      pickup_lng: toCoord(first.pickup_lng ?? req?.pickup_lng),
      dropoff_lat: toCoord(first.dropoff_lat ?? req?.dropoff_lat),
      dropoff_lng: toCoord(first.dropoff_lng ?? req?.dropoff_lng),
      pickup_address: String(first.pickup_location || req?.pickup_location || '').trim(),
      dropoff_address: String(
        first.dropoff_location || req?.dropoff_location || '',
      ).trim(),
    };
  }
  return {
    pickup_lat: toCoord(req?.pickup_lat),
    pickup_lng: toCoord(req?.pickup_lng),
    dropoff_lat: toCoord(req?.dropoff_lat),
    dropoff_lng: toCoord(req?.dropoff_lng),
    pickup_address: String(req?.pickup_location || '').trim(),
    dropoff_address: String(req?.dropoff_location || '').trim(),
  };
}

/** @deprecated Préférer resolveOutboundRoute */
export function resolveOutboundRouteEndpoints(req) {
  const route = resolveOutboundRoute(req);
  return {
    pickup_lat: route.pickup_lat,
    pickup_lng: route.pickup_lng,
    dropoff_lat: route.dropoff_lat,
    dropoff_lng: route.dropoff_lng,
  };
}

/** Libellé court du trajet aller (1re étape). */
export function formatOutboundRouteLabel(req) {
  const route = resolveOutboundRoute(req);
  const pickup = route.pickup_address || '—';
  const dropoff = route.dropoff_address || '—';
  return `${pickup} → ${dropoff}`;
}

export function haversineKm(lat1, lon1, lat2, lon2) {
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2;
  return EARTH_RADIUS_KM * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

/** Repli si Google Directions indisponible. */
export function estimateTravelMinutesHaversine(endpoints) {
  if (!hasValidEndpoints(endpoints)) return null;
  const { pickup_lat, pickup_lng, dropoff_lat, dropoff_lng } = endpoints;
  const straightKm = haversineKm(pickup_lat, pickup_lng, dropoff_lat, dropoff_lng);
  if (straightKm <= 0) return null;
  const roadKm = straightKm * ROAD_FACTOR;
  const minutes = Math.round((roadKm / FALLBACK_AVG_SPEED_KMH) * 60);
  return Math.max(5, minutes);
}

/** Instant de départ pour Google (heure mission confirmée si disponible). */
export function resolveDirectionsDepartureUnix(req) {
  const iso = req?.next_confirmed_time || req?.scheduled_time || req?.mission_date;
  if (!iso) return null;
  const ms = Date.parse(iso);
  if (!Number.isFinite(ms)) return null;
  return Math.floor(ms / 1000);
}

async function geocodeAddress(address) {
  const query = String(address || '').trim();
  if (!query) return null;

  const tryGeocode = async (addr) => {
    const response = await apiClient.get('/geocode/geocode', {
      params: { address: addr, country: 'CH' },
      timeout: GEOCODE_TIMEOUT_MS,
    });
    const lat = toCoord(response.data?.lat);
    const lng = toCoord(response.data?.lon ?? response.data?.lng);
    if (lat == null || lng == null) return null;
    return { lat, lng };
  };

  try {
    const aliases = await apiClient.get('/geocode/aliases', {
      params: { q: query },
      timeout: GEOCODE_TIMEOUT_MS,
    });
    const hit = Array.isArray(aliases.data) ? aliases.data[0] : null;
    if (hit?.lat != null && hit?.lon != null) {
      const lat = toCoord(hit.lat);
      const lng = toCoord(hit.lon);
      if (lat != null && lng != null) return { lat, lng };
    }
  } catch {
    // Alias optionnel — on tente le géocodage complet.
  }

  try {
    return await tryGeocode(query);
  } catch {
    // Adresse longue (nom + rue) : retenter sans le préfixe avant la 1re virgule.
    const simplified = query.includes(',') ? query.split(',').slice(1).join(',').trim() : '';
    if (!simplified || simplified === query) return null;
    try {
      return await tryGeocode(simplified);
    } catch {
      return null;
    }
  }
}

/** Complète les coordonnées manquantes via géocodage backend (Google). */
export async function ensureOutboundRouteCoords(route) {
  let {
    pickup_lat: pickupLat,
    pickup_lng: pickupLng,
    dropoff_lat: dropoffLat,
    dropoff_lng: dropoffLng,
    pickup_address: pickupAddress,
    dropoff_address: dropoffAddress,
  } = route;

  const tasks = [];

  if (pickupLat == null || pickupLng == null) {
    tasks.push(
      geocodeAddress(pickupAddress).then((coords) => {
        if (coords) {
          pickupLat = coords.lat;
          pickupLng = coords.lng;
        }
      }),
    );
  }

  if (dropoffLat == null || dropoffLng == null) {
    tasks.push(
      geocodeAddress(dropoffAddress).then((coords) => {
        if (coords) {
          dropoffLat = coords.lat;
          dropoffLng = coords.lng;
        }
      }),
    );
  }

  if (tasks.length > 0) {
    await Promise.all(tasks);
  }

  return {
    pickup_lat: pickupLat,
    pickup_lng: pickupLng,
    dropoff_lat: dropoffLat,
    dropoff_lng: dropoffLng,
  };
}

const minutesFromDirectionsPayload = (data) => {
  const trafficSec = Number(data?.duration_in_traffic_seconds);
  const durationSec = Number(data?.duration_seconds);
  const chosen = Number.isFinite(trafficSec) && trafficSec > 0 ? trafficSec : durationSec;
  if (Number.isFinite(chosen) && chosen > 0) {
    return Math.max(1, Math.round(chosen / 60));
  }
  return null;
};

/**
 * Durée via Google Directions (proxy backend, clé serveur).
 * @returns {Promise<number|null>}
 */
export async function fetchGoogleMapsTravelMinutes(endpoints, departureUnix = null) {
  if (!hasValidEndpoints(endpoints)) return null;

  const attempts = [
    departureUnix != null && departureUnix > 0 ? departureUnix : null,
    null,
  ];

  for (const dep of attempts) {
    try {
      const body = {
        origin: { lat: endpoints.pickup_lat, lng: endpoints.pickup_lng },
        destination: { lat: endpoints.dropoff_lat, lng: endpoints.dropoff_lng },
        mode: 'driving',
        region: 'ch',
      };
      if (dep != null) body.departure_time = dep;

      const response = await apiClient.post('/directions', body, {
        timeout: DIRECTIONS_TIMEOUT_MS,
      });
      const data = response.data || {};
      if (data.status === 'OK') {
        const minutes = minutesFromDirectionsPayload(data);
        if (minutes != null) return minutes;
      }
    } catch {
      // Essai suivant ou repli haversine.
    }
  }

  return estimateTravelMinutesHaversine(endpoints);
}

/**
 * Estimation serveur (GET, JWT) — évite géocodage + CSRF côté client.
 * @returns {Promise<number|null>}
 */
export async function fetchOfferTravelMinutes(offerId) {
  if (offerId == null) return null;
  try {
    const response = await apiClient.get(
      `/company/request-offers/${offerId}/travel-estimate`,
      { timeout: DIRECTIONS_TIMEOUT_MS },
    );
    const minutes = Number(response.data?.travel_minutes);
    if (Number.isFinite(minutes) && minutes > 0) return minutes;
  } catch {
    // Repli client ci-dessous.
  }
  return null;
}

/**
 * Durée de trajet : endpoint serveur → géocodage → Google Directions → haversine.
 * @returns {Promise<number|null>}
 */
export async function fetchRouteTravelMinutes(req, offerId = null) {
  if (offerId != null) {
    const serverMinutes = await fetchOfferTravelMinutes(offerId);
    if (serverMinutes != null) return serverMinutes;
  }

  const route = resolveOutboundRoute(req);
  const endpoints = await ensureOutboundRouteCoords(route);
  const departureUnix = resolveDirectionsDepartureUnix(req);
  return fetchGoogleMapsTravelMinutes(endpoints, departureUnix);
}
