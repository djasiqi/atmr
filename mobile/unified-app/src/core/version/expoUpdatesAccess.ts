/**
 * Repli résolu par tsc : le tsconfig n'a pas de moduleSuffixes .native/.web.
 * Metro préfère expoUpdatesAccess.native.ts (iOS/Android) et
 * expoUpdatesAccess.web.ts (web). Ce module ne charge pas expo-updates.
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
