type Listener = () => void;

let availabilityActive = true;
const listeners = new Set<Listener>();

export function getDriverAvailabilityActive(): boolean {
  return availabilityActive;
}

export function setDriverAvailabilityActive(active: boolean): void {
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
  availabilityActive = true;
  listeners.clear();
}
