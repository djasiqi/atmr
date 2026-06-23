/** Écouteurs appelés après un refresh JWT réussi (levée suspension queue tracking). */

const listeners = new Set<() => void>();

export function onAuthRefreshSuccess(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function notifyAuthRefreshSuccess(): void {
  for (const listener of listeners) {
    try {
      listener();
    } catch {
      /* noop */
    }
  }
}
