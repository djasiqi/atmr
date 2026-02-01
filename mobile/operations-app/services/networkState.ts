/**
 * P0.2.C — Snapshot réseau pour logs (corrélation logout ↔ offline).
 * P1.C — Support subscription pour OfflineBanner (event-driven, pas de poll).
 */

let cachedState: Record<string, unknown> | null = null;
let unsubscribe: (() => void) | null = null;
type Listener = () => void;
const listeners = new Set<Listener>();

function notifyListeners(): void {
  listeners.forEach((fn) => {
    try {
      fn();
    } catch (e) {
      if (__DEV__) console.warn("[networkState] listener error:", e);
    }
  });
}

/**
 * S'abonner aux changements d'état réseau (event-driven).
 * Retourne une fonction de désabonnement.
 */
export function subscribeToNetworkState(onChange: () => void): () => void {
  listeners.add(onChange);
  return () => {
    listeners.delete(onChange);
  };
}

/**
 * Initialise le cache réseau (appelé au démarrage de l'app).
 */
export function initNetworkStateCache(): void {
  if (unsubscribe) return;
  try {
    const NetInfo = require("@react-native-community/netinfo").default;
    const updateCache = (state: unknown) => {
      const s = state as { isConnected?: boolean | null; isInternetReachable?: boolean | null; type?: string } | null;
      cachedState = s
        ? {
            isConnected: s.isConnected ?? null,
            isInternetReachable: s.isInternetReachable ?? null,
            type: s.type ?? "unknown",
          }
        : null;
      notifyListeners();
    };
    unsubscribe = NetInfo.addEventListener(updateCache);
    NetInfo.fetch().then(updateCache).catch(() => {});
  } catch {
    // NetInfo non disponible (web, tests)
  }
}

/**
 * Retourne le dernier état réseau connu (sync).
 */
export function getNetworkStateSnapshot(): Record<string, unknown> | null {
  if (!cachedState) return null;
  return {
    isConnected: cachedState.isConnected,
    isInternetReachable: cachedState.isInternetReachable,
    type: cachedState.type,
  };
}
