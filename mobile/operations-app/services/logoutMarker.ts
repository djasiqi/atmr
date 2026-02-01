/**
 * P1.C — Marqueur de logout pour bannière UX "Session expirée".
 * TTL court (5 min) pour éviter un message obsolète si l'utilisateur rouvre l'app plus tard.
 */

import AsyncStorage from "@react-native-async-storage/async-storage";
import { isSessionExpiredReason as isSessionExpiredFromReasons } from "./authLogoutReasons";

const LOGOUT_MARKER_KEY = "auth.logout_marker";
const TTL_MS = 5 * 60 * 1000; // 5 minutes

export type LogoutMarker = {
  route: "driver" | "enterprise";
  reason: string;
  ts: number;
};

/** Re-export depuis authLogoutReasons (source unique). */
export const isSessionExpiredReason = isSessionExpiredFromReasons;

/**
 * Enregistre un marqueur de logout (appelé dans forceLogout*Internal).
 */
export async function setLogoutMarker(params: LogoutMarker): Promise<void> {
  try {
    await AsyncStorage.setItem(
      LOGOUT_MARKER_KEY,
      JSON.stringify({ ...params, ts: Date.now() })
    );
  } catch (e) {
    if (__DEV__) {
      console.warn("[logoutMarker] setLogoutMarker failed:", e);
    }
  }
}

/**
 * Lit et supprime le marqueur pour la route donnée.
 * Retourne null si absent, expiré (TTL > 5 min), ou route non correspondante.
 */
export async function consumeLogoutMarker(
  route: "driver" | "enterprise"
): Promise<LogoutMarker | null> {
  try {
    const raw = await AsyncStorage.getItem(LOGOUT_MARKER_KEY);
    if (!raw) return null;

    const marker = JSON.parse(raw) as LogoutMarker;
    if (marker.route !== route) return null;

    const age = Date.now() - (marker.ts ?? 0);
    if (age > TTL_MS) {
      await AsyncStorage.removeItem(LOGOUT_MARKER_KEY);
      return null;
    }

    await AsyncStorage.removeItem(LOGOUT_MARKER_KEY);
    return marker;
  } catch {
    await AsyncStorage.removeItem(LOGOUT_MARKER_KEY).catch(() => {});
    return null;
  }
}
