import Constants from "expo-constants";

import type * as ExpoNotifications from "expo-notifications";

function isExpoGoRuntime(): boolean {
  return Constants.appOwnership === "expo";
}

export function canUseExpoNotifications(): boolean {
  return !isExpoGoRuntime();
}

export function getExpoNotificationsModule(): typeof ExpoNotifications | null {
  if (!canUseExpoNotifications()) {
    return null;
  }
  // Chargement lazy pour éviter l'exception Expo Go (SDK 53+ Android).
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  return require("expo-notifications") as typeof ExpoNotifications;
}
