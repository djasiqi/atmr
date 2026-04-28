/* Jest exécute ce fichier en premier (setupFiles[0]) : retirer fetch *avant* axios,
 * sinon l’import d’axios 1.x charge fetch.js + polyfill streams Expo et crashe. */
if (typeof globalThis.fetch === "function") {
  // eslint-disable-next-line no-undef
  globalThis.__JEST_SAVED_FETCH__ = globalThis.fetch;
  // eslint-disable-next-line no-undef
  delete globalThis.fetch;
}
