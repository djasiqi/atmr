/** Navigation SPA déclenchée hors composants React (logout, session expirée). */
export const AUTH_NAVIGATE_EVENT = 'lirie:navigate';

export const requestAuthNavigate = (to, { replace = true } = {}) => {
  if (typeof window === 'undefined' || !to) return;
  window.dispatchEvent(
    new CustomEvent(AUTH_NAVIGATE_EVENT, {
      detail: { to, replace },
    })
  );
};
