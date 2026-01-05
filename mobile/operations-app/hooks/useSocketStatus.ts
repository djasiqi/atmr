// mobile/operations-app/hooks/useSocketStatus.ts

import { useEffect, useState, useRef } from "react";
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
  const socketRef = useRef<any>(null);
  const reconnectAttemptRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const connectedRef = useRef(false); // Pour suivre l'état précédent dans les closures

  useEffect(() => {
    // Fonction pour mettre à jour l'état depuis le socket actuel
    const updateStatus = () => {
      const socket = getSocket();
      if (!socket) {
        setConnected(false);
        setReconnecting(false);
        connectedRef.current = false;
        return;
      }

      // Mettre à jour l'état initial
      const wasConnected = connectedRef.current;
      const isNowConnected = socket.connected;
      
      setConnected(isNowConnected);
      connectedRef.current = isNowConnected;
      
      // Si le socket vient de se connecter, mettre à jour lastConnected
      if (isNowConnected && !wasConnected) {
        setLastConnected(new Date());
      }
      
      // Si le socket est déconnecté mais qu'on était connecté avant, ne pas marquer comme reconnexion immédiatement
      if (!isNowConnected && wasConnected) {
        // Laisser les événements gérer la reconnexion
      }
    };

    // Vérifier périodiquement l'état (fallback si les événements ne se déclenchent pas)
    const statusCheckInterval = setInterval(() => {
      updateStatus();
    }, 2000); // Vérifier toutes les 2 secondes

    // Obtenir le socket initial
    const socket = getSocket();
    socketRef.current = socket;

    if (!socket) {
      setConnected(false);
      setReconnecting(false);
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
      console.log("[useSocketStatus] Socket connecté");
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

    const handleDisconnect = (reason?: string) => {
      console.log("[useSocketStatus] Socket déconnecté:", reason);
      setConnected(false);
      connectedRef.current = false;
      setReconnecting(false);
    };

    const handleReconnect = (attemptNumber: number) => {
      console.log("[useSocketStatus] Socket reconnecté, tentative:", attemptNumber);
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

    const handleReconnectAttempt = (attemptNumber: number) => {
      console.log("[useSocketStatus] Tentative de reconnexion:", attemptNumber);
      setReconnecting(true);
      setConnected(false);
      connectedRef.current = false;
    };

    const handleConnectError = (error: any) => {
      console.warn("[useSocketStatus] Erreur de connexion:", error);
      setConnected(false);
      connectedRef.current = false;
      // Marquer comme en reconnexion après un court délai
      if (reconnectAttemptRef.current) {
        clearTimeout(reconnectAttemptRef.current);
      }
      reconnectAttemptRef.current = setTimeout(() => {
        setReconnecting(true);
      }, 1000);
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
    latency, // null pour l'instant, peut être enrichi plus tard
    lastConnected,
  };
}

