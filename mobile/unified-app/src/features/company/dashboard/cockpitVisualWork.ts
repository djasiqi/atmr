/**
 * Politique MOUNTED ≠ ACTIVE pour le cockpit (onglets `lazy: false`).
 * L’écran reste monté pour une nav instantanée ; le travail visuel coûteux
 * s’arrête hors focus. Aucun changement de sémantique GPS / métier.
 */

export function resolveCockpitVisualWork(isScreenFocused: boolean): {
  visualWorkEnabled: boolean;
  shouldRecordScreenRender: boolean;
} {
  return {
    visualWorkEnabled: isScreenFocused,
    shouldRecordScreenRender: isScreenFocused,
  };
}

/** Hors focus : ignorer drivers / missions pour ne pas recalculer la carte invisible. */
export function shouldFreezeCockpitMapData(
  prevVisualWorkEnabled: boolean,
  nextVisualWorkEnabled: boolean
): boolean {
  return !prevVisualWorkEnabled && !nextVisualWorkEnabled;
}
