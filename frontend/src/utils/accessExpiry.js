/**
 * Métadonnée non sensible d'expiration de l'access token (cookie HttpOnly).
 * Permet au scheduler de programmer T−5 min sans décoder le JWT.
 */

export const ACCESS_EXPIRES_AT_STORAGE_KEY = 'app_access_expires_at';

const FUTURE_SKEW_MS = 5 * 1000;

/**
 * Parse une valeur ISO / epoch / access_expires_in en timestamp ms.
 * @param {{ access_expires_at?: string, access_expires_in?: number, expires_in?: number }|string|number|null|undefined} payload
 * @param {number} [nowMs]
 * @returns {number|null}
 */
export function resolveAccessExpiresAtMs(payload, nowMs = Date.now()) {
  if (payload == null) {
    return null;
  }
  if (typeof payload === 'number' && Number.isFinite(payload)) {
    // Epoch secondes vs ms
    return payload < 1e12 ? payload * 1000 : payload;
  }
  if (typeof payload === 'string') {
    const asNumber = Number(payload);
    if (Number.isFinite(asNumber) && asNumber > 0) {
      return asNumber < 1e12 ? asNumber * 1000 : asNumber;
    }
    const parsed = Date.parse(payload);
    return Number.isFinite(parsed) ? parsed : null;
  }
  if (typeof payload === 'object') {
    if (payload.access_expires_at) {
      return resolveAccessExpiresAtMs(payload.access_expires_at, nowMs);
    }
    const expiresIn = payload.access_expires_in ?? payload.expires_in;
    if (typeof expiresIn === 'number' && Number.isFinite(expiresIn) && expiresIn > 0) {
      return nowMs + expiresIn * 1000;
    }
  }
  return null;
}

export function getStoredAccessExpiresAtMs() {
  if (typeof localStorage === 'undefined') {
    return null;
  }
  try {
    const raw = localStorage.getItem(ACCESS_EXPIRES_AT_STORAGE_KEY);
    if (!raw) {
      return null;
    }
    const ms = resolveAccessExpiresAtMs(raw);
    if (ms == null) {
      return null;
    }
    // Rejet des timestamps aberrants (trop dans le passé lointain ou futur absurde)
    const now = Date.now();
    if (ms > now + 90 * 24 * 3600 * 1000 + FUTURE_SKEW_MS) {
      return null;
    }
    return ms;
  } catch (_) {
    return null;
  }
}

/**
 * Persiste l'instant d'expiration (ms) dérivé de la réponse login/refresh.
 * @returns {number|null} timestamp ms stocké
 */
export function noteAccessExpiryFromResponse(payload, nowMs = Date.now()) {
  const ms = resolveAccessExpiresAtMs(payload, nowMs);
  if (ms == null || typeof localStorage === 'undefined') {
    return null;
  }
  try {
    localStorage.setItem(ACCESS_EXPIRES_AT_STORAGE_KEY, new Date(ms).toISOString());
  } catch (_) {
    // mode privé / quota
  }
  return ms;
}

export function clearStoredAccessExpiry() {
  if (typeof localStorage === 'undefined') {
    return;
  }
  try {
    localStorage.removeItem(ACCESS_EXPIRES_AT_STORAGE_KEY);
  } catch (_) {
    // no-op
  }
}

/**
 * @param {number} aheadMs — marge avant expiration (ex. 5 min)
 * @param {number} [nowMs]
 */
export function isAccessNearExpiry(aheadMs, nowMs = Date.now()) {
  const expiresAt = getStoredAccessExpiresAtMs();
  if (expiresAt == null) {
    return false;
  }
  return expiresAt - nowMs <= aheadMs;
}

export function isAccessExpired(nowMs = Date.now()) {
  const expiresAt = getStoredAccessExpiresAtMs();
  if (expiresAt == null) {
    return false;
  }
  return expiresAt <= nowMs;
}
