import { Platform } from "react-native";

export type ExpoLocationPermissionResult = {
  granted?: boolean;
  status?: string;
  android?: { accuracy?: "fine" | "coarse" | "none" | string };
  /** Certaines versions Expo futures peuvent exposer ce champ ; absent dans la version actuelle. */
  ios?: { accuracy?: "full" | "reduced" | string; scope?: string };
};

export type LocationAccuracyStatus = "precise" | "approximate" | "unknown";

/** Aligné sur expo-notifications : `granted` ou `status === "granted"`. */
export function isExpoLocationPermissionGranted(
  perm: ExpoLocationPermissionResult | null | undefined
): boolean {
  return Boolean(perm?.granted || perm?.status === "granted");
}

/**
 * Précision de permission (pas la précision GPS d'un fix).
 * Aligné sur les types expo-location du dépôt :
 * - Android : `android.accuracy` = fine | coarse | none
 * - iOS : pas de champ accuracy dans la version installée → si FG accordée, `precise`
 *   (fail-open iOS pour ne pas bloquer tous les appareils) ; si le champ apparaît, il prime.
 */
export function resolveLocationAccuracy(
  permission: ExpoLocationPermissionResult | null | undefined
): LocationAccuracyStatus {
  if (!isExpoLocationPermissionGranted(permission)) {
    return "unknown";
  }

  if (Platform.OS === "android") {
    const acc = permission?.android?.accuracy;
    if (acc === "fine") return "precise";
    if (acc === "coarse") return "approximate";
    return "unknown";
  }

  if (Platform.OS === "ios") {
    const acc = permission?.ios?.accuracy;
    if (acc === "full") return "precise";
    if (acc === "reduced") return "approximate";
    // Expo actuel : PermissionDetailsLocationIOS n'expose que `scope`.
    return "precise";
  }

  return "unknown";
}
