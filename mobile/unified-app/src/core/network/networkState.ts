export type NetworkSnapshot = {
  connected: boolean;
  internetReachable: boolean;
  type: string;
  cellularGeneration: string | null;
  updatedAt: string;
};

const listeners = new Set<(snapshot: NetworkSnapshot) => void>();
let unsubscribeNative: (() => void) | null = null;
let currentSnapshot: NetworkSnapshot = {
  connected: true,
  internetReachable: true,
  type: "unknown",
  cellularGeneration: null,
  updatedAt: new Date().toISOString(),
};

function publish(snapshot: NetworkSnapshot) {
  currentSnapshot = snapshot;
  listeners.forEach((listener) => listener(snapshot));
}

function toSnapshot(input: Record<string, unknown>): NetworkSnapshot {
  return {
    connected: Boolean(input.isConnected ?? true),
    internetReachable: Boolean(input.isInternetReachable ?? input.isConnected ?? true),
    type: typeof input.type === "string" ? input.type : "unknown",
    cellularGeneration:
      typeof input.cellularGeneration === "string" ? input.cellularGeneration : null,
    updatedAt: new Date().toISOString(),
  };
}

async function startNativeSubscription() {
  if (unsubscribeNative) return;
  try {
    const netInfo = await import("@react-native-community/netinfo");
    const initial = await netInfo.default.fetch();
    publish(toSnapshot(initial as unknown as Record<string, unknown>));
    const subscription = netInfo.default.addEventListener((state) => {
      publish(toSnapshot(state as unknown as Record<string, unknown>));
    });
    unsubscribeNative = () => {
      subscription();
      unsubscribeNative = null;
    };
  } catch {
    const fallback = () => {
      const online = typeof navigator !== "undefined" ? Boolean(navigator.onLine) : true;
      publish({
        connected: online,
        internetReachable: online,
        type: "fallback",
        cellularGeneration: null,
        updatedAt: new Date().toISOString(),
      });
    };
    fallback();
    const interval = setInterval(fallback, 15_000);
    unsubscribeNative = () => {
      clearInterval(interval);
      unsubscribeNative = null;
    };
  }
}

export function subscribeNetworkState(listener: (snapshot: NetworkSnapshot) => void): () => void {
  listeners.add(listener);
  listener(currentSnapshot);
  void startNativeSubscription();
  return () => {
    listeners.delete(listener);
    if (listeners.size === 0 && unsubscribeNative) {
      unsubscribeNative();
    }
  };
}

export function getNetworkSnapshot(): NetworkSnapshot {
  return currentSnapshot;
}
