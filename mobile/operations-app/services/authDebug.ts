/**
 * Logs forensic enterprise auth (refresh_token timeline) — actifs uniquement si DEBUG_AUTH=1.
 * N'expose jamais la valeur des tokens (seulement présence, longueur, clé).
 * Usage : EXPO_PUBLIC_DEBUG_AUTH=1 dans .env ou variables d'environnement.
 */
const DEBUG_AUTH_ENABLED =
  typeof process !== "undefined" &&
  process.env?.EXPO_PUBLIC_DEBUG_AUTH === "1";

export function isDebugAuthEnabled(): boolean {
  return DEBUG_AUTH_ENABLED;
}

type DebugAuthPhase =
  | "ent_refresh_write"
  | "ent_refresh_read"
  | "boot_storage"
  | "notify_auth_ready"
  | "interceptor_before_refresh";

/** payload ne doit contenir aucune valeur de token, seulement des métadonnées (présence, longueur, clé, flags). */
export function debugAuthLog(
  phase: DebugAuthPhase,
  payload: Record<string, string | number | boolean | undefined>
): void {
  if (!DEBUG_AUTH_ENABLED) return;
  try {
    const msg = `[DEBUG_AUTH] ${phase} ${JSON.stringify(payload)}`;
    if (typeof console !== "undefined" && console.debug) {
      console.debug(msg);
    }
  } catch {
    // Ne jamais casser le flux
  }
}
