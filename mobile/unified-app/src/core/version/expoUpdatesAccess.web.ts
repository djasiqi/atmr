/**
 * Variante web / SSR : n'importe pas `expo-updates`.
 * Chaque rendu serveur réexécutait ce module, empilait un listener et faisait grossir le heap jusqu'à l'OOM.
 */
import type { ExpoUpdateSnapshot } from "./expoUpdateSnapshot";

const EMPTY_UPDATE: ExpoUpdateSnapshot = {
  updateId: null,
  channel: null,
  runtimeVersion: null,
  isEnabled: false,
  isEmbeddedLaunch: true,
};

export function getUpdateInfo(): ExpoUpdateSnapshot {
  return EMPTY_UPDATE;
}

export async function checkForOtaUpdate(): Promise<{ isAvailable: boolean }> {
  return { isAvailable: false };
}

export async function fetchOtaUpdate(): Promise<{ isNew: boolean }> {
  return { isNew: false };
}

export async function reloadOtaUpdate(): Promise<void> {
  return;
}

export function useExpoUpdatesState(): {
  isUpdatePending: boolean;
  isUpdateAvailable: boolean;
} {
  return { isUpdatePending: false, isUpdateAvailable: false };
}
