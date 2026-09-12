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

const COMPANY_RETURN_PATH_PREFIXES = Object.freeze([
  '/dashboard/company/',
  '/company/',
]);

/**
 * Interprète ?returnTo= pour les écrans entreprise (chemin interne, préfixes autorisés).
 * @param {string|null} returnToParam — valeur brute du query param
 * @returns {string|null}
 */
export function pathFromCompanyReturnTo(returnToParam) {
  const path = pathFromNextQueryParam(returnToParam);
  if (!path) return null;
  const pathname = path.split('?')[0];
  const allowed = COMPANY_RETURN_PATH_PREFIXES.some((prefix) => pathname.startsWith(prefix));
  return allowed ? path : null;
}
