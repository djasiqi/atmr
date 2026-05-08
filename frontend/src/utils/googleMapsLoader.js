/**
 * Chargement unique et partagé du SDK Google Maps (évite double <script> et courses concurrentes).
 * Cœur : `shared/google-maps/bootstrap.js` (alias `@atmr/google-maps-bootstrap`).
 * Préchargeable depuis le layout entreprise ; consommé par {@link GoogleMapsProvider}.
 */

import {
  GOOGLE_MAPS_SCRIPT_ID,
  isGoogleMapsSdkReady,
  loadGoogleMapsScriptWithKey,
  parseGoogleMapsLibraryList,
} from '@atmr/google-maps-bootstrap';

/**
 * Librairies chargées en plus du cœur `maps` (paramètre `libraries=`).
 * Seul `marker` est requis pour `google.maps.marker.AdvancedMarkerElement`.
 * Surcharge : REACT_APP_GOOGLE_MAPS_LIBRARIES=marker,places
 * @returns {string[]}
 */
export function getGoogleMapsLibraryList() {
  return parseGoogleMapsLibraryList(process.env.REACT_APP_GOOGLE_MAPS_LIBRARIES);
}

export { isGoogleMapsSdkReady, GOOGLE_MAPS_SCRIPT_ID };

/**
 * Garantit que le SDK est chargé. Même promesse partagée entre appelants concurrents.
 * @returns {Promise<void>}
 */
export function loadGoogleMapsScript() {
  const apiKey = process.env.REACT_APP_GOOGLE_MAPS_API_KEY;
  if (!apiKey) {
    return Promise.reject(new Error('API key manquante'));
  }
  return loadGoogleMapsScriptWithKey(apiKey, {
    libraryList: getGoogleMapsLibraryList(),
  });
}

/**
 * Précharge le SDK tôt (dashboard entreprise) : requêtes réseau différées (idle + repli 1,5 s).
 * Les échecs sont silencieux ici ; l’utilisateur verra l’état d’erreur via le provider si besoin.
 * @returns {() => void} annuler l’ordonnancement (unmount)
 */
export function schedulePrefetchGoogleMaps() {
  if (typeof window === 'undefined' || isGoogleMapsSdkReady()) {
    return () => {};
  }
  if (!process.env.REACT_APP_GOOGLE_MAPS_API_KEY) {
    return () => {};
  }

  const run = () => {
    loadGoogleMapsScript().catch(() => {});
  };

  let idleId = null;
  const timeoutId = window.setTimeout(run, 1500);
  if (typeof window.requestIdleCallback === 'function') {
    idleId = window.requestIdleCallback(run, { timeout: 900 });
  }

  return () => {
    if (timeoutId != null) {
      window.clearTimeout(timeoutId);
    }
    if (idleId != null && typeof window.cancelIdleCallback === 'function') {
      window.cancelIdleCallback(idleId);
    }
  };
}
