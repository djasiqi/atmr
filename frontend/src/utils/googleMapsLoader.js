/**
 * Chargement unique et partagé du SDK Google Maps (évite double <script> et courses concurrentes).
 * Préchargeable depuis le layout entreprise ; consommé par {@link GoogleMapsProvider}.
 */

const GOOGLE_MAPS_SCRIPT_ID = 'google-maps-script';

/**
 * Librairies chargées en plus du cœur `maps` (paramètre `libraries=`).
 * Audit code (2026) : `places` et `geometry` ne sont pas utilisés côté front
 * (autocomplétion adresses = API backend + Photon). Seul `marker` est requis
 * pour `google.maps.marker.AdvancedMarkerElement`.
 * Surcharge (urgence / nouvelle feature) : REACT_APP_GOOGLE_MAPS_LIBRARIES=marker,places
 * @returns {string[]}
 */
export function getGoogleMapsLibraryList() {
  const raw = (process.env.REACT_APP_GOOGLE_MAPS_LIBRARIES || 'marker').trim();
  if (!raw) return ['marker'];
  return raw.split(/[\s,]+/).filter(Boolean);
}

/** @returns {boolean} */
export function isGoogleMapsSdkReady() {
  return (
    typeof window !== 'undefined' && typeof window.google?.maps?.Map === 'function'
  );
}

const NAMESPACE_WAIT_MS = 15000;
const NAMESPACE_POLL_MS = 50;

/** Attend que `google.maps` existe (bootstrap chargé), avant `importLibrary`. */
function waitForGoogleMapsNamespace() {
  return new Promise((resolve, reject) => {
    const startedAt = Date.now();
    const tick = () => {
      if (typeof window !== 'undefined' && window.google?.maps) {
        resolve();
        return;
      }
      if (Date.now() - startedAt > NAMESPACE_WAIT_MS) {
        reject(new Error('SDK Google Maps indisponible (timeout)'));
        return;
      }
      window.setTimeout(tick, NAMESPACE_POLL_MS);
    };
    tick();
  });
}

/**
 * Avec `loading=async` (recommandé par Google), `google.maps.Map` n'existe pas tant que
 * la librairie « maps » n'est pas importée via `importLibrary()`.
 * Sans cet appel, le chargement du script réussit mais `Map` reste indéfini → timeout.
 *
 * @returns {Promise<void>}
 */
async function bootstrapGoogleMapsLibraries() {
  if (isGoogleMapsSdkReady()) {
    return;
  }
  const gm = window.google?.maps;
  if (!gm) {
    throw new Error('google.maps absent après chargement du script');
  }
  // Ancien chargement synchrone : rien à faire
  if (typeof gm.Map === 'function') {
    return;
  }
  if (typeof gm.importLibrary !== 'function') {
    throw new Error(
      'Google Maps : importLibrary indisponible (script incomplet ou API obsolète)'
    );
  }
  await gm.importLibrary('maps');
  const extra = getGoogleMapsLibraryList().filter((name) => name !== 'maps');
  await Promise.all(extra.map((lib) => gm.importLibrary(lib)));
}

let inFlight = null;

/**
 * Garantit que le SDK est chargé. Même promesse partagée entre appelants concurrents.
 * @returns {Promise<void>}
 */
export function loadGoogleMapsScript() {
  if (isGoogleMapsSdkReady()) {
    return Promise.resolve();
  }
  if (inFlight) {
    return inFlight;
  }

  const apiKey = process.env.REACT_APP_GOOGLE_MAPS_API_KEY;
  if (!apiKey) {
    return Promise.reject(new Error('API key manquante'));
  }

  inFlight = new Promise((resolve, reject) => {
    const previousGmf = window.gm_authFailure;

    const cleanupGmf = (replacement) => {
      if (window.gm_authFailure === replacement) {
        window.gm_authFailure = previousGmf;
      }
    };

    const fail = (err) => {
      inFlight = null;
      cleanupGmf(gmf);
      reject(err);
    };

    const ok = () => {
      cleanupGmf(gmf);
      resolve();
    };

    const gmf = () => {
      if (typeof previousGmf === 'function') {
        previousGmf();
      }
      fail(
        new Error('Authentification Google Maps refusée (clé API, domaine ou billing)')
      );
    };
    window.gm_authFailure = gmf;

    const existing = document.getElementById(GOOGLE_MAPS_SCRIPT_ID);
    if (existing) {
      waitForGoogleMapsNamespace()
        .then(() => bootstrapGoogleMapsLibraries())
        .then(ok)
        .catch(fail);
      return;
    }

    const libs = getGoogleMapsLibraryList().join(',');
    const script = document.createElement('script');
    script.id = GOOGLE_MAPS_SCRIPT_ID;
    script.src = `https://maps.googleapis.com/maps/api/js?key=${apiKey}&libraries=${libs}&loading=async`;
    script.async = true;
    script.defer = true;

    script.onload = () => {
      waitForGoogleMapsNamespace()
        .then(() => bootstrapGoogleMapsLibraries())
        .then(ok)
        .catch(fail);
    };
    script.onerror = () => {
      fail(new Error('Échec chargement Google Maps SDK'));
    };

    document.head.appendChild(script);
  });

  return inFlight;
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

export { GOOGLE_MAPS_SCRIPT_ID };
