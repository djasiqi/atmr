/**
 * Enregistre le service worker PWA (production uniquement).
 */
export function registerPwaServiceWorker() {
  if (typeof window === 'undefined') return;
  if (process.env.NODE_ENV !== 'production') return;
  if (!('serviceWorker' in navigator)) return;

  window.addEventListener('load', () => {
    const base = (process.env.PUBLIC_URL || '').replace(/\/$/, '');
    const swPath = `${base}/service-worker.js`;
    navigator.serviceWorker.register(swPath).catch(() => {
      /* silencieux : hébergeur sans SW, HTTP non sécurisé hors localhost, etc. */
    });
  });
}
