/**
 * Fallback pour les URLs PDF en développement.
 * Remplace localhost par 127.0.0.1 pour éviter ERR_CONNECTION_RESET
 * (IPv6 localhost + Docker Windows provoque un reset sur les réponses volumineuses).
 *
 * @param {string} url - URL du PDF (ex: http://localhost:5000/uploads/invoices/xxx.pdf)
 * @returns {string} URL corrigée si on est en dev sur localhost
 */
export function ensurePdfUrlWorksInDev(url) {
  if (!url || typeof url !== 'string') return url;
  if (typeof window === 'undefined') return url;
  // En dev uniquement : si on est sur localhost et que l'URL contient localhost, remplacer par 127.0.0.1
  if (
    window.location.hostname === 'localhost' &&
    url.includes('localhost')
  ) {
    return url.replace(/localhost/g, '127.0.0.1');
  }
  return url;
}
