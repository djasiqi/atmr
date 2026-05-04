/**
 * Normalise les URLs PDF pour l’affichage (iframe, impression, nouvel onglet).
 *
 * En développement (CRA) :
 * - Les URLs absolues vers le backend local (`http://127.0.0.1:5000/uploads/...`) sont
 *   réécrites en **chemins relatifs** `/uploads/...` pour passer par `setupProxy.js`.
 *   Sinon le navigateur tente une connexion directe au port 5000 → souvent
 *   « 127.0.0.1 n'autorise pas la connexion » / ERR_CONNECTION_REFUSED si seul le
 *   proxy webpack est utilisé ou si Docker n’expose pas ce port.
 *
 * Fallback historique : sur la page en `localhost`, remplacer `localhost` par `127.0.0.1`
 * dans l’URL (certains cas IPv6 / Docker Windows).
 *
 * @param {string} url - URL du PDF (absolue ou relative)
 * @returns {string} URL utilisable dans le navigateur
 */
export function ensurePdfUrlWorksInDev(url) {
  if (!url || typeof url !== 'string') return url;
  const trimmed = url.trim();
  if (typeof window === 'undefined') return trimmed;

  /** Déjà relatif : laisser tel quel (déjà servi par le même origine / proxy). */
  if (trimmed.startsWith('/')) {
    return trimmed;
  }

  if (process.env.NODE_ENV === 'development') {
    try {
      const u = new URL(trimmed);
      const loopback = u.hostname === '127.0.0.1' || u.hostname === 'localhost';
      const port = u.port;
      /** Ports backend habituels du repo (setupProxy, docker-compose). */
      const looksLikeLocalApi =
        loopback && (port === '5000' || port === '5100');
      if (looksLikeLocalApi && u.pathname.startsWith('/uploads')) {
        return `${u.pathname}${u.search}`;
      }
    } catch {
      /* URL invalide : retomber sur les règles ci-dessous */
    }
  }

  if (
    window.location.hostname === 'localhost' &&
    trimmed.includes('localhost')
  ) {
    return trimmed.replace(/localhost/g, '127.0.0.1');
  }
  return trimmed;
}

/**
 * Ajoute un fragment « PDF Open » (#toolbar=0&navpanes=0) pour masquer la barre d’outils
 * du lecteur PDF **intégré à Chromium** (Chrome, Edge, etc.) dans un `<iframe>`.
 *
 * Ne supprime pas le contenu côté extension Adobe Acrobat (overlay sur le canvas) :
 * seul le chrome du viewer navigateur est concerné.
 *
 * @param {string} url - URL complète ou chemin (ex. `/uploads/…pdf?x=1`)
 * @returns {string}
 */
export function appendPdfEmbedChromiumViewerFragment(url) {
  if (!url || typeof url !== 'string') return url;
  const trimmed = url.trim();
  if (!trimmed) return trimmed;

  const fragmentParams = 'toolbar=0&navpanes=0';

  try {
    const base =
      typeof window !== 'undefined' && window.location?.origin
        ? window.location.origin
        : 'http://localhost';
    const u = new URL(trimmed, base);
    const existing = u.hash ? u.hash.replace(/^#/, '') : '';
    u.hash = existing ? `${existing}&${fragmentParams}` : fragmentParams;

    if (trimmed.startsWith('/')) {
      return `${u.pathname}${u.search}${u.hash}`;
    }
    return u.toString();
  } catch {
    if (trimmed.includes('#')) {
      const sep = trimmed.endsWith('#') || trimmed.endsWith('&') ? '' : '&';
      return `${trimmed}${sep}${fragmentParams}`;
    }
    return `${trimmed}#${fragmentParams}`;
  }
}
