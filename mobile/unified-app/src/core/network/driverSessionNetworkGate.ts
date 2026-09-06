/**
 * DRIVER-RUNTIME-01B — barrière réseau chauffeur unique.
 * Le shell peut se peindre depuis le snapshot local ; le réseau authentifié
 * n’ouvre qu’après SESSION_READY (bootstrap), pas sur le ready UI anticipé.
 */

type ReadyListener = () => void;

let driverSessionNetworkReady = false;
const listeners = new Set<ReadyListener>();

export function isDriverSessionNetworkReady(): boolean {
  return driverSessionNetworkReady;
}

export function setDriverSessionNetworkReady(next: boolean): void {
  if (driverSessionNetworkReady === next) return;
  driverSessionNetworkReady = next;
  for (const listener of listeners) {
    listener();
  }
}

export function subscribeDriverSessionNetworkReady(listener: ReadyListener): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function isAuthOnlyRequestUrl(url: string): boolean {
  const normalized = url.startsWith("/") ? url : `/${url}`;
  return normalized.includes("/auth/");
}

export function isDriverProtectedRequestUrl(url: string): boolean {
  const normalized = url.startsWith("/") ? url : `/${url}`;
  return (
    normalized.startsWith("/driver/") ||
    normalized.includes("/messages/") ||
    normalized.includes("/conversations/") ||
    normalized.includes("/telemetry/")
  );
}

/**
 * true = ne pas envoyer la requête (pas de 401).
 * /auth/* toujours autorisé. Contexte chauffeur : tout le reste attend SESSION_READY.
 */
export function shouldBlockDriverRequestUntilSessionReady(
  url: string,
  contextId?: string | null
): boolean {
  if (isAuthOnlyRequestUrl(url)) return false;
  if (driverSessionNetworkReady) return false;
  if (isDriverProtectedRequestUrl(url)) return true;
  return typeof contextId === "string" && contextId.startsWith("driver:");
}

export function resetDriverSessionNetworkGateForTests(): void {
  driverSessionNetworkReady = false;
  listeners.clear();
}
