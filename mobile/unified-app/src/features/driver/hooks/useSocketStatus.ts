import { useEffect, useState } from "react";
import { realtimeManager } from "../../../core/realtime/realtimeManager";

export type SocketStatus = {
  connected: boolean;
  reconnecting: boolean;
  degraded: boolean;
  authExhausted: boolean;
  authAttempts: number;
  transportMode: "idle" | "polling" | "socket";
  transportAuthority: "socket" | "polling" | "reconcile" | "degraded";
  lastError: string | null;
  lastConnectedAt?: number;
};

export function useSocketStatus(): SocketStatus {
  const initialSnapshot = realtimeManager.getSnapshot();
  const [connected, setConnected] = useState(initialSnapshot.connected);
  const [reconnecting, setReconnecting] = useState(
    initialSnapshot.mode === "polling" && initialSnapshot.desiredTransport === "socket"
  );
  const [degraded, setDegraded] = useState(initialSnapshot.degradedMode);
  const [authExhausted, setAuthExhausted] = useState(initialSnapshot.authExhausted);
  const [authAttempts, setAuthAttempts] = useState(initialSnapshot.authAttempts);
  const [transportMode, setTransportMode] = useState(initialSnapshot.actualTransport);
  const [transportAuthority, setTransportAuthority] = useState(initialSnapshot.transportAuthority);
  const [lastError, setLastError] = useState(initialSnapshot.lastError);
  const [lastConnectedAt, setLastConnectedAt] = useState<number | undefined>(
    initialSnapshot.connected ? Date.now() : undefined
  );

  useEffect(() => {
    return realtimeManager.subscribe((snapshot) => {
      setConnected(snapshot.connected);
      setReconnecting(
        snapshot.mode === "polling" && snapshot.desiredTransport === "socket"
      );
      setDegraded(snapshot.degradedMode);
      setAuthExhausted(snapshot.authExhausted);
      setAuthAttempts(snapshot.authAttempts);
      setTransportMode(snapshot.actualTransport);
      setTransportAuthority(snapshot.transportAuthority);
      setLastError(snapshot.lastError);
      if (snapshot.connected) {
        setLastConnectedAt(Date.now());
      }
    });
  }, []);

  return {
    connected,
    reconnecting,
    degraded,
    authExhausted,
    authAttempts,
    transportMode,
    transportAuthority,
    lastError,
    lastConnectedAt,
  };
}

