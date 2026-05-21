/**
 * Source unique ATMR/LIRIE pour le chargement du SDK Google Maps JS (web).
 * Utilisé par le frontend entreprise (CRA) et l’app unifiée Expo web (import relatif).
 *
 * Règles : un seul SCRIPT_ID, URL avec v=weekly + libraries + loading=async,
 * pas d’injection de <script> en dehors de ce module.
 *
 * Readiness : le code métier utilise `google.maps.importLibrary()` (Dynamic Library Import).
 * Ne pas se contenter de `google.maps.Map` : un ancien script peut exposer Map sans importLibrary.
 */

export const GOOGLE_MAPS_SCRIPT_ID = 'google-maps-script';

/**
 * @param {string | undefined} raw
 * @returns {string[]}
 */
export function parseGoogleMapsLibraryList(raw) {
  const r = (raw || 'marker').trim();
  if (!r) return ['marker'];
  return r.split(/[\s,]+/).filter(Boolean);
}

/**
 * Présence du chargeur dynamique (recommandation Google Maps JS API moderne).
 * @returns {boolean}
 */
export function isGoogleMapsImportLibraryAvailable() {
  return (
    typeof window !== 'undefined' && typeof window.google?.maps?.importLibrary === 'function'
  );
}

/**
 * SDK utilisable par notre code : `importLibrary` + noyau `maps` chargé (constructeur Map).
 * @returns {boolean}
 */
export function isGoogleMapsSdkReady() {
  if (typeof window === 'undefined') return false;
  const gm = window.google?.maps;
  return (
    typeof gm?.importLibrary === 'function' &&
    typeof gm?.Map === 'function'
  );
}

const NAMESPACE_WAIT_MS = 15000;
const NAMESPACE_POLL_MS = 50;

/**
 * Attend `google.maps` avec `importLibrary` (pas seulement l’objet namespace vide).
 */
function waitForGoogleMapsNamespace() {
  return new Promise((resolve, reject) => {
    const startedAt = Date.now();
    const tick = () => {
      if (
        typeof window !== 'undefined' &&
        window.google?.maps &&
        typeof window.google.maps.importLibrary === 'function'
      ) {
        resolve();
        return;
      }
      if (Date.now() - startedAt > NAMESPACE_WAIT_MS) {
        reject(
          new Error(
            'SDK Google Maps : importLibrary indisponible après timeout — vérifier l’URL du script (v=weekly, loading=async) et l’absence d’un autre chargeur Maps obsolète.'
          )
        );
        return;
      }
      window.setTimeout(tick, NAMESPACE_POLL_MS);
    };
    tick();
  });
}

/** Inclut toujours `maps` dans la query `libraries=` pour le bootstrap réseau. */
function buildLibrariesQuery(libraryList) {
  const ordered = ['maps'];
  for (const name of libraryList) {
    if (name && name !== 'maps' && !ordered.includes(name)) {
      ordered.push(name);
    }
  }
  return ordered.join(',');
}

/**
 * @param {string[]} libraryList
 * @returns {Promise<void>}
 */
async function bootstrapGoogleMapsLibraries(libraryList) {
  const gm = window.google?.maps;
  if (!gm) {
    throw new Error('google.maps absent après chargement du script');
  }
  if (typeof gm.importLibrary !== 'function') {
    throw new Error(
      'Google Maps : importLibrary indisponible — exiger le script officiel v=weekly avec loading=async (voir bootstrap.js).'
    );
  }
  await gm.importLibrary('maps');
  const extra = libraryList.filter((name) => name !== 'maps');
  await Promise.all(extra.map((lib) => gm.importLibrary(lib)));
}

let inFlight = null;

/**
 * Si le namespace existe déjà (autre intégration), réutiliser sans second <script>.
 * @param {string} apiKey
 * @param {{ libraryList?: string[] }} [opts]
 * @returns {Promise<void>}
 */
export function loadGoogleMapsScriptWithKey(apiKey, opts = {}) {
  if (typeof window === 'undefined' || typeof document === 'undefined') {
    return Promise.resolve();
  }

  const libraryList =
    opts.libraryList && opts.libraryList.length > 0
      ? opts.libraryList
      : parseGoogleMapsLibraryList('marker');

  if (!apiKey) {
    return Promise.reject(new Error('API key manquante'));
  }

  if (isGoogleMapsSdkReady()) {
    return Promise.resolve();
  }

  if (inFlight) {
    return inFlight;
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

    const finishBootstrap = () =>
      waitForGoogleMapsNamespace()
        .then(() => bootstrapGoogleMapsLibraries(libraryList))
        .then(ok)
        .catch(fail);

    if (window.google?.maps) {
      finishBootstrap();
      return;
    }

    const existing = document.getElementById(GOOGLE_MAPS_SCRIPT_ID);
    if (existing) {
      finishBootstrap();
      return;
    }

    const libs = buildLibrariesQuery(libraryList);
    const script = document.createElement('script');
    script.id = GOOGLE_MAPS_SCRIPT_ID;
    script.setAttribute('data-atmr-maps-loader', '1');
    script.async = true;
    script.defer = true;
    script.src = `https://maps.googleapis.com/maps/api/js?key=${encodeURIComponent(
      apiKey
    )}&v=weekly&libraries=${libs}&loading=async`;

    script.onload = () => finishBootstrap();
    script.onerror = () => {
      fail(new Error('Échec chargement Google Maps SDK'));
    };

    document.head.appendChild(script);
  });

  return inFlight;
}
