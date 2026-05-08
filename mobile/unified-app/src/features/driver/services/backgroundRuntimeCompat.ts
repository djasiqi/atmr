import Constants from "expo-constants";
import { Platform } from "react-native";

/**
 * Détecte si on tourne dans Expo Go. Le tracking GPS background nécessite :
 *  - `expo-task-manager` natif (TaskManager.defineTask + Location.startLocationUpdatesAsync)
 *  - le foreground service Android (notif persistante "Suivi en cours…")
 *  - les modes UIBackgroundModes iOS
 *
 * Aucune de ces capacités n'existe dans Expo Go : il faut un dev client custom
 * (`npx expo run:android` / `eas build --profile development`).
 */
export function isExpoGoRuntime(): boolean {
  return Constants.appOwnership === "expo";
}

/**
 * Détermine si l'environnement courant peut faire tourner le tracking GPS
 * en background. Retourne `false` sur web et Expo Go.
 *
 * À appeler avant tout `Location.startLocationUpdatesAsync` ou
 * `TaskManager.defineTask` pour court-circuiter proprement et émettre une
 * télémétrie compréhensible plutôt qu'un crash silencieux.
 */
export function canUseBackgroundLocation(): boolean {
  if (Platform.OS === "web") return false;
  return !isExpoGoRuntime();
}

/**
 * Libellé court pour la télémétrie/diagnostic — évite de répéter la condition.
 */
export function describeBackgroundRuntime():
  | "web_unsupported"
  | "expo_go_unsupported"
  | "dev_client_or_standalone" {
  if (Platform.OS === "web") return "web_unsupported";
  if (isExpoGoRuntime()) return "expo_go_unsupported";
  return "dev_client_or_standalone";
}
