import Constants from "expo-constants";
import { Platform } from "react-native";

import type NotifeeType from "@notifee/react-native";

/**
 * Détecte si on tourne dans Expo Go. Notifee est un module natif tiers
 * (Java/Obj-C) absent de l'application Expo Go : tenter de l'importer
 * lance "Notifee native module not found" même si l'`await import(...)`
 * est wrap dans un try/catch.
 */
function isExpoGoRuntime(): boolean {
  return Constants.appOwnership === "expo";
}

/**
 * `true` si Notifee peut être chargé sans erreur (dev client / standalone
 * iOS/Android). `false` sur web et Expo Go.
 */
export function canUseNotifee(): boolean {
  if (Platform.OS === "web") return false;
  return !isExpoGoRuntime();
}

/**
 * Charge Notifee de façon dynamique uniquement quand l'environnement le
 * permet. Renvoie `null` sinon — l'appelant doit gérer ce cas
 * silencieusement (cohérent avec les try/catch déjà en place).
 */
export async function loadNotifee(): Promise<typeof NotifeeType | null> {
  if (!canUseNotifee()) return null;
  try {
    return await import("@notifee/react-native");
  } catch {
    return null;
  }
}
