import { PUBLIC_SEO_PATHS } from '../config/publicSeo';

/** Navigation SPA déclenchée hors composants React (logout, session expirée). */
export const AUTH_NAVIGATE_EVENT = 'lirie:navigate';

export const GUEST_SAFERPAY_RETURN_PATH = '/guest/payment/saferpay/return';

/** Parcours AUTH-PRO consultables sans session (formulaire / invitation / orientation). */
const AUTH_PRO_ANONYMOUS_PATH_PREFIXES = Object.freeze([
  '/inscription/organisation',
  '/invitation/chauffeur',
]);

function normalizePathname(pathname) {
  const raw = typeof pathname === 'string' ? pathname.split('?')[0] : '';
  if (!raw) return '';
  if (raw.length > 1 && raw.endsWith('/')) {
    return raw.slice(0, -1);
  }
  return raw;
}

/** Retour bancaire invité : page publique sans compte. */
export function isGuestSaferpayReturnPath(pathname) {
  const path = normalizePathname(pathname);
  return path === GUEST_SAFERPAY_RETURN_PATH
    || path.startsWith(`${GUEST_SAFERPAY_RETURN_PATH}/`);
}

/**
 * Pages anonymes : une session morte ne doit pas forcer /login.
 * Les portails métier (dashboard, driver, admin) restent hors de cette liste.
 */
export function isAnonymousPublicPath(pathname) {
  const path = normalizePathname(pathname);
  if (!path) return false;
  if (isGuestSaferpayReturnPath(path)) return true;
  if (PUBLIC_SEO_PATHS.includes(path)) return true;
  return AUTH_PRO_ANONYMOUS_PATH_PREFIXES.some(
    (prefix) => path === prefix || path.startsWith(`${prefix}/`)
  );
}

export const requestAuthNavigate = (to, { replace = true } = {}) => {
  if (typeof window === 'undefined' || !to) return;
  const destination = String(to);
  if (
    isAnonymousPublicPath(window.location.pathname)
    && (destination === '/login' || destination.startsWith('/login?'))
  ) {
    return;
  }
  const event = new CustomEvent(AUTH_NAVIGATE_EVENT, {
    detail: { to, replace },
    cancelable: true,
  });
  window.dispatchEvent(event);
  if (!event.defaultPrevented) {
    try {
      if (replace && typeof window.location.replace === 'function') {
        window.location.replace(to);
      } else if (typeof window.location.assign === 'function') {
        window.location.assign(to);
      } else {
        window.location.href = to;
      }
    } catch (_) {
      // jsdom / tests sans Location API complète
    }
  }
};
