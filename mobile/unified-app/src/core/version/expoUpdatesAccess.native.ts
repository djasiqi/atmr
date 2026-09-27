import { useUpdates } from "expo-updates";
import * as Updates from "expo-updates";
import type { ExpoUpdateSnapshot } from "./expoUpdateSnapshot";

export type { ExpoUpdateSnapshot };

export function getUpdateInfo(): ExpoUpdateSnapshot {
  return {
    updateId: Updates.updateId ?? null,
    channel: Updates.channel ?? null,
    runtimeVersion: Updates.runtimeVersion ?? null,
    isEnabled: Updates.isEnabled,
    isEmbeddedLaunch: Updates.isEmbeddedLaunch ?? true,
  };
}

export async function checkForOtaUpdate(): Promise<{ isAvailable: boolean }> {
  return Updates.checkForUpdateAsync();
}

export async function fetchOtaUpdate(): Promise<{ isNew: boolean }> {
  return Updates.fetchUpdateAsync();
}

export async function reloadOtaUpdate(): Promise<void> {
  await Updates.reloadAsync();
}

export function useExpoUpdatesState(): {
  isUpdatePending: boolean;
  isUpdateAvailable: boolean;
} {
  return useUpdates();
}
