export type ExpoLocationPermissionResult = {
  granted?: boolean;
  status?: string;
};

/** Aligné sur expo-notifications : `granted` ou `status === "granted"`. */
export function isExpoLocationPermissionGranted(
  perm: ExpoLocationPermissionResult | null | undefined
): boolean {
  return Boolean(perm?.granted || perm?.status === "granted");
}
