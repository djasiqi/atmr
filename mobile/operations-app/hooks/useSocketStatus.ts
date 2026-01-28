// mobile/operations-app/hooks/useSocketStatus.ts

import { useEffect, useState, useRef } from "react";
import { getSocket, getConnectionState, type ConnectionState } from "@/services/socket";

export interface SocketStatus {
  connected: boolean;
  reconnecting: boolean;
  latency: number | null;
  lastConnected: Date | null;
  /** P0.3: ONLINE | RECONNECTING | OFFLINE (OFFLINE = logout explicite ou jamais connecté) */
  connectionState?: ConnectionState;
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
  const [connectionState, setConnectionState] = useState<ConnectionState>("OFFLINE");
  const socketRef = useRef<any>(null);
  const reconnectAttemptRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const connectedRef = useRef(false);

  useEffect(() => {
    // Fonction pour mettre à jour l'état depuis le socket actuel + P0.3 getConnectionState()
    const updateStatus = () => {
      const state = getConnectionState();
      setConnectionState(state);
      const socket = getSocket();
      if (!socket) {
        setConnected(false);
        setReconnecting(state === "RECONNECTING");
        connectedRef.current = false;
        return;
      }

      const wasConnected = connectedRef.current;
      const isNowConnected = socket.connected;
      setConnected(isNowConnected);
      connectedRef.current = isNowConnected;
      setReconnecting(state === "RECONNECTING");
      if (isNowConnected && !wasConnected) {
        setLastConnected(new Date());
      }
    };

    // Vérifier périodiquement l'état (fallback si les événements ne se déclenchent pas)
    const statusCheckInterval = setInterval(() => {
      updateStatus();
    }, 2000); // Vérifier toutes les 2 secondes

    // Obtenir le socket initial et état P0.3
    const socket = getSocket();
    socketRef.current = socket;
    updateStatus(); // mise à jour immédiate (connected, reconnecting, connectionState)

    if (!socket) {
      setConnectionState(getConnectionState());
      return () => {
        clearInterval(statusCheckInterval);
      };
    }

    // État initial
    const initialConnected = socket.connected;
    setConnected(initialConnected);
    connectedRef.current = initialConnected;
    if (initialConnected) {
      setLastConnected(new Date());
    }

    // Écouter les événements de connexion
    const handleConnect = () => {
      setConnectionState(getConnectionState());
      setConnected(true);
      connectedRef.current = true;
      setReconnecting(false);
      setLastConnected(new Date());
      // Nettoyer le timer de reconnexion si présent
      if (reconnectAttemptRef.current) {
        clearTimeout(reconnectAttemptRef.current);
        reconnectAttemptRef.current = null;
      }
    };

    const handleDisconnect = () => {
      setConnectionState(getConnectionState());
      setConnected(false);
      connectedRef.current = false;
      setReconnecting(getConnectionState() === "RECONNECTING");
    };

    const handleReconnect = () => {
      setConnectionState(getConnectionState());
      setConnected(true);
      connectedRef.current = true;
      setReconnecting(false);
      setLastConnected(new Date());
      // Nettoyer le timer de reconnexion si présent
      if (reconnectAttemptRef.current) {
        clearTimeout(reconnectAttemptRef.current);
        reconnectAttemptRef.current = null;
      }
    };

    const handleReconnectAttempt = () => {
      setConnectionState(getConnectionState());
      setReconnecting(true);
      setConnected(false);
      connectedRef.current = false;
    };

    const handleConnectError = () => {
      setConnectionState(getConnectionState());
      setConnected(false);
      connectedRef.current = false;
      setReconnecting(true);
    };

    // Attacher les listeners
    socket.on("connect", handleConnect);
    socket.on("disconnect", handleDisconnect);
    socket.on("reconnect", handleReconnect);
    socket.on("reconnect_attempt", handleReconnectAttempt);
    socket.on("connect_error", handleConnectError);

    // Cleanup
    return () => {
      clearInterval(statusCheckInterval);
      if (reconnectAttemptRef.current) {
        clearTimeout(reconnectAttemptRef.current);
        reconnectAttemptRef.current = null;
      }
      if (socket) {
        socket.off("connect", handleConnect);
        socket.off("disconnect", handleDisconnect);
        socket.off("reconnect", handleReconnect);
        socket.off("reconnect_attempt", handleReconnectAttempt);
        socket.off("connect_error", handleConnectError);
      }
    };
  }, []); // Dépendances vides - on s'abonne une seule fois aux événements

  return {
    connected,
    reconnecting,
    latency,
    lastConnected,
    connectionState,
  };
}

