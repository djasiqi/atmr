/**
 * Interpolation fluide des marqueurs chauffeur (visibilité GPS continue).
 */

export const MARKER_MOTION_MIN_MS = 2400;
export const MARKER_MOTION_MAX_MS = 20000;
export const MARKER_MOTION_DEFAULT_MS = 10000;
/** Étire légèrement l'animation sur l'intervalle entre deux fixes GPS. */
export const MARKER_MOTION_DURATION_STRETCH = 1.42;
/** Fraction de l'intervalle GPS utilisée pour la phase de projection (dead reckoning). */
export const MARKER_MOTION_PROJECT_FRACTION = 0.55;
export const MARKER_MOTION_PROJECT_VELOCITY_DECAY = 0.88;

/**
 * Courbe smoothstep — accélération / décélération douce (plus naturelle que linéaire).
 */
export function easeSmoothStep(t) {
  const x = Math.min(1, Math.max(0, t));
  return x * x * (3 - 2 * x);
}

/**
 * Durée d'animation = intervalle entre mises à jour (étiré + borné).
 */
export function resolveMarkerMotionDurationMs(lastTargetAtMs, nowMs = Date.now()) {
  if (!lastTargetAtMs || !Number.isFinite(lastTargetAtMs)) {
    return MARKER_MOTION_DEFAULT_MS;
  }
  const elapsed = nowMs - lastTargetAtMs;
  if (elapsed <= 0) return MARKER_MOTION_MIN_MS;
  const stretched = elapsed * MARKER_MOTION_DURATION_STRETCH;
  return Math.min(
    MARKER_MOTION_MAX_MS,
    Math.max(MARKER_MOTION_MIN_MS, stretched)
  );
}

export function haversineDistanceMeters(from, to) {
  const lat1 = (from.lat * Math.PI) / 180;
  const lat2 = (to.lat * Math.PI) / 180;
  const dLat = lat2 - lat1;
  const dLng = ((to.lng - from.lng) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLng / 2) ** 2;
  return 2 * 6371000 * Math.asin(Math.min(1, Math.sqrt(a)));
}

/** Ajuste la durée selon la distance (courts trajets = glide plus long, perçu plus fluide). */
export function resolveMotionDurationFromDistance(durationMs, distanceM) {
  if (!Number.isFinite(distanceM) || distanceM <= 0) return durationMs;
  if (distanceM < 12) return Math.max(durationMs, 3200);
  if (distanceM > 180) {
    return Math.min(MARKER_MOTION_MAX_MS, durationMs * 1.12);
  }
  return durationMs;
}

export function interpolateMarkerPosition(from, to, progress, easing = easeSmoothStep) {
  const t = easing(progress);
  return {
    lat: from.lat + (to.lat - from.lat) * t,
    lng: from.lng + (to.lng - from.lng) * t,
  };
}

/** Projection dead-reckoning après arrivée sur le fix GPS. */
export function projectPositionAlongVelocity(lat, lng, velLatPerMs, velLngPerMs, deltaMs) {
  return {
    lat: lat + velLatPerMs * deltaMs,
    lng: lng + velLngPerMs * deltaMs,
  };
}

export function normalizeMarkerLatLng(pos) {
  if (!pos) return null;
  const lat = Number(pos.lat ?? pos.latitude);
  const lng = Number(pos.lng ?? pos.longitude);
  if (!Number.isFinite(lat) || !Number.isFinite(lng)) return null;
  return { lat, lng };
}
