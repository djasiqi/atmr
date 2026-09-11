/** Navigation SPA déclenchée hors composants React (logout, session expirée). */
export const AUTH_NAVIGATE_EVENT = 'lirie:navigate';

export const requestAuthNavigate = (to, { replace = true } = {}) => {
  if (typeof window === 'undefined' || !to) return;
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
