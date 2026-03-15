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
