type Listener = () => void;

/** null = UNKNOWN (pas encore hydraté depuis DB/cache). */
let availabilityActive: boolean | null = null;
const listeners = new Set<Listener>();

export function getDriverAvailabilityActive(): boolean | null {
  return availabilityActive;
}

export function setDriverAvailabilityActive(active: boolean | null): void {
  if (availabilityActive === active) return;
  availabilityActive = active;
  listeners.forEach((listener) => listener());
}

export function subscribeDriverAvailability(listener: Listener): () => void {
  listeners.add(listener);
  listener();
  return () => {
    listeners.delete(listener);
  };
}

export function resetDriverAvailabilityBridgeForTests(): void {
  availabilityActive = null;
  listeners.clear();
}
