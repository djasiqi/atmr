// mobile/operations-app/hooks/useSocketStatus.ts

import { useEffect, useState } from "react";
import { getSocket } from "@/services/socket";

export interface SocketStatus {
  connected: boolean;
  reconnecting: boolean;
  latency: number | null;
  lastConnected: Date | null;
}

/**
 * Hook pour exposer l'état de connexion Socket.IO
 * @returns {SocketStatus} { connected, reconnecting, latency, lastConnected }
 */
export function useSocketStatus(): SocketStatus {
  const [connected, setConnected] = useState(false);
  const [reconnecting, setReconnecting] = useState(false);
  const [latency, setLatency] = useState<number | null>(null);
  const [lastConnected, setLastConnected] = useState<Date | null>(null);

  useEffect(() => {
    const socket = getSocket();
    if (!socket) {
      setConnected(false);
      return;
    }

    // État initial
    setConnected(socket.connected);
    if (socket.connected) {
      setLastConnected(new Date());
    }

    // Écouter les événements de connexion
    const handleConnect = () => {
      setConnected(true);
      setReconnecting(false);
      setLastConnected(new Date());
    };

    const handleDisconnect = () => {
      setConnected(false);
      setReconnecting(false);
    };

    const handleReconnect = (attemptNumber: number) => {
      setReconnecting(true);
      // Après reconnexion réussie, handleConnect sera appelé
    };

    // Attacher les listeners
    socket.on("connect", handleConnect);
    socket.on("disconnect", handleDisconnect);
    socket.on("reconnect", handleReconnect);

    // Cleanup
    return () => {
      socket.off("connect", handleConnect);
      socket.off("disconnect", handleDisconnect);
      socket.off("reconnect", handleReconnect);
    };
  }, []);

  return {
    connected,
    reconnecting,
    latency, // null pour l'instant, peut être enrichi plus tard
    lastConnected,
  };
}

