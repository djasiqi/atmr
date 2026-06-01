type Listener = () => void;

let pushPermissionDenied = false;
const listeners = new Set<Listener>();

export function getPushPermissionDenied(): boolean {
  return pushPermissionDenied;
}

export function setPushPermissionDenied(value: boolean): void {
  if (pushPermissionDenied === value) return;
  pushPermissionDenied = value;
  listeners.forEach((listener) => listener());
}

export function subscribePushPermissionDenied(listener: Listener): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function clearPushPermissionDenied(): void {
  setPushPermissionDenied(false);
}
