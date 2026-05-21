/**
 * Chargement unique et partagé du SDK Google Maps (évite double <script> et courses concurrentes).
 * Cœur : `shared/google-maps/bootstrap.js` (alias `@atmr/google-maps-bootstrap`).
 * Préchargeable depuis le layout entreprise ; consommé par {@link GoogleMapsProvider}.
 *
 * Clé API : uniquement `REACT_APP_GOOGLE_MAPS_API_KEY` (build CRA).
 * Ne pas utiliser `GOOGLE_MAPS_API_KEY` (serveur backend/.env) côté navigateur.
 * Readiness : voir `isGoogleMapsSdkReady` / `isGoogleMapsImportLibraryAvailable` dans bootstrap (importLibrary).
 */

import {
  GOOGLE_MAPS_SCRIPT_ID,
  isGoogleMapsImportLibraryAvailable,
  isGoogleMapsSdkReady,
  loadGoogleMapsScriptWithKey,
  parseGoogleMapsLibraryList,
} from '../shared/google-maps/bootstrap';

/**
 * @returns {string | undefined}
 */
export function resolveWebMapsApiKey() {
  const react =
    typeof process.env.REACT_APP_GOOGLE_MAPS_API_KEY === 'string'
      ? process.env.REACT_APP_GOOGLE_MAPS_API_KEY.trim()
      : '';
  return react.length > 0 ? react : undefined;
}

/**
 * Librairies chargées en plus du cœur `maps` (paramètre `libraries=` + importLibrary).
 * Surcharge : REACT_APP_GOOGLE_MAPS_LIBRARIES=marker,places
 * @returns {string[]}
 */
export function getGoogleMapsLibraryList() {
  const raw = process.env.REACT_APP_GOOGLE_MAPS_LIBRARIES || '';
  return parseGoogleMapsLibraryList(raw);
}

export {
  GOOGLE_MAPS_SCRIPT_ID,
  isGoogleMapsImportLibraryAvailable,
  isGoogleMapsSdkReady,
};

/**
 * Garantit que le SDK est chargé. Même promesse partagée entre appelants concurrents.
 * @returns {Promise<void>}
 */
export function loadGoogleMapsScript() {
  const apiKey = resolveWebMapsApiKey();
  if (!apiKey) {
    return Promise.reject(new Error('API key manquante (REACT_APP_GOOGLE_MAPS_API_KEY)'));
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
  if (!resolveWebMapsApiKey()) {
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
