/**
 * Chemin interne sûr pour une redirection post-login (évite les open redirects).
 * @param {string} pathname
 * @param {string} [search] — ex. "?bookingId=1" (tel que useLocation().search)
 * @returns {string|null}
 */
export function buildSafeAppPath(pathname, search = '') {
  if (typeof pathname !== 'string' || !pathname.startsWith('/') || pathname.startsWith('//')) {
    return null;
  }
  if (pathname.includes('..')) return null;
  if (pathname === '/login') return null;
  const s = typeof search === 'string' ? search : '';
  return `${pathname}${s}`;
}

/**
 * Interprète ?next= (path relatif ou absolu même origine).
 * @param {string} nextParam — valeur brute du query param
 * @returns {string|null}
 */
export function pathFromNextQueryParam(nextParam) {
  if (!nextParam || typeof nextParam !== 'string') return null;
  try {
    const u = new URL(decodeURIComponent(nextParam), window.location.origin);
    if (u.origin !== window.location.origin) return null;
    return buildSafeAppPath(u.pathname, u.search);
  } catch {
    return null;
  }
}
