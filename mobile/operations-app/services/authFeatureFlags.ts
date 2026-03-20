/**
 * Feature flags auth / perf (Sprint 3 — STR-02).
 * Défaut : tout désactivé. Activer via app.config extra ou EXPO_PUBLIC_*.
 */
import Constants from "expo-constants";

const getExtra = (key: string): string | undefined => {
  const extra = Constants.expoConfig?.extra as Record<string, unknown> | undefined;
  const v = extra?.[key];
  return v != null && v !== "" ? String(v) : undefined;
};

const envTrue = (name: string): boolean => {
  const v =
    (typeof process !== "undefined" &&
      process.env?.[name as keyof typeof process.env]) ||
    getExtra(name);
  return String(v || "").toLowerCase() === "true" || v === "1";
};

/**
 * Quand true : `setDriverLoading(false)` juste après `notifyAuthReady()` sur le chemin
 * token driver valide, avant la fin de `fetchDriverProfile` (profil continue en « arrière-plan »
 * du point de vue de `driverLoading`). Combiné à STR-01 (`splashBlocking`), réduit la sérialisation perçue.
 * Risque : écrans qui supposent `!driverLoading` ⇒ profil déjà là ; garder les gardes sur `driver`.
 */
export const FEATURE_RELEASE_DRIVER_LOADING_BEFORE_PROFILE =
  envTrue("EXPO_PUBLIC_FEATURE_RELEASE_DRIVER_LOADING_BEFORE_PROFILE");
