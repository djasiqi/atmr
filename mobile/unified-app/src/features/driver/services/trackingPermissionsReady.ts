/**
 * Capacité de garantir le contrat GPS (FG + BG).
 * Distinct de driverAvailable (intention métier en service).
 */

type Listener = () => void;

let permissionsReady = false;
const listeners = new Set<Listener>();

export function getTrackingPermissionsReady(): boolean {
  return permissionsReady;
}

export function setTrackingPermissionsReady(ready: boolean): void {
  if (permissionsReady === ready) return;
  permissionsReady = ready;
  listeners.forEach((listener) => listener());
}

export function subscribeTrackingPermissionsReady(listener: Listener): () => void {
  listeners.add(listener);
  listener();
  return () => {
    listeners.delete(listener);
  };
}

export function resetTrackingPermissionsReadyForTests(): void {
  permissionsReady = false;
  listeners.clear();
}
