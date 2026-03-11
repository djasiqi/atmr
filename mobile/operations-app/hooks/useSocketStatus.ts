// mobile/operations-app/hooks/useSocketStatus.ts
// Plan 2G/3G Phase 7 : subscribeSocketStatus uniquement (sans polling 2s)

import { useEffect, useState } from "react";
import { subscribeSocketStatus, type ConnectionState } from "@/services/socket";

export interface SocketStatus {
  connected: boolean;
  reconnecting: boolean;
  latency: number | null;
  lastConnected: Date | null;
  /** P0.3: ONLINE | RECONNECTING | OFFLINE */
  connectionState?: ConnectionState;
}

/**
 * Hook pour exposer l'état de connexion Socket.IO (event-driven, pas de polling).
 */
export function useSocketStatus(): SocketStatus {
  const [connectionState, setConnectionState] = useState<ConnectionState>("OFFLINE");
  const [lastConnected, setLastConnected] = useState<Date | null>(null);

  useEffect(() => {
    const unsubscribe = subscribeSocketStatus((payload) => {
      setConnectionState(payload.connectionState);
      if (payload.connectionState === "ONLINE") {
        setLastConnected(new Date());
      }
    });
    return unsubscribe;
  }, []);

  const connected = connectionState === "ONLINE";
  const reconnecting = connectionState === "RECONNECTING";

  return {
    connected,
    reconnecting,
    latency: null,
    lastConnected,
    connectionState,
  };
}

