/**
 * Polyfill pour APIs web utilisées par Expo HMR sur React Native.
 * Évite "ReferenceError: Property 'document' doesn't exist" (Hermes/Android).
 */
if (typeof global.document === 'undefined') {
  global.document = {
    querySelectorAll: () => [],
    addEventListener: () => {},
    removeEventListener: () => {},
  };
}

/**
 * Axios (certaines versions récentes) lit window.location.href au chargement.
 * En runtime RN bridgeless, window peut exister sans location -> crash startup.
 * On normalise un objet location minimal pour stabiliser le boot.
 */
const g = globalThis;
if (typeof g.window === 'undefined') {
  g.window = g;
}
if (!g.window.location) {
  g.window.location = { href: 'http://localhost/' };
} else if (typeof g.window.location.href !== 'string') {
  g.window.location.href = 'http://localhost/';
}
if (typeof g.location === 'undefined') {
  g.location = g.window.location;
}
