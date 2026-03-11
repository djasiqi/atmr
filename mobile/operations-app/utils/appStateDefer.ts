/**
 * Utilitaires pour différer le travail lors des transitions AppState.
 * Évite les Background ANR en ne bloquant jamais le main thread pendant
 * les transitions foreground ↔ background.
 *
 * Règle : tout travail déclenché par AppState doit être différé.
 */

import { InteractionManager } from "react-native";

/** Délai minimum (ms) avant de lancer le travail au retour au premier plan. */
const FOREGROUND_DELAY_MS = 150;

/** Diffère l'exécution pour une transition vers arrière-plan (background/inactive). */
export function deferForBackground(callback: () => void): void {
  InteractionManager.runAfterInteractions(() => {
    setTimeout(callback, 0);
  });
}

/** Diffère l'exécution pour une transition vers premier plan (active). */
export function deferForForeground(callback: () => void): void {
  InteractionManager.runAfterInteractions(() => {
    setTimeout(callback, FOREGROUND_DELAY_MS);
  });
}

/** Diffère l'exécution (foreground ou background). */
export function deferAppStateWork(
  callback: () => void,
  options: { isForeground: boolean }
): void {
  if (options.isForeground) {
    deferForForeground(callback);
  } else {
    deferForBackground(callback);
  }
}
