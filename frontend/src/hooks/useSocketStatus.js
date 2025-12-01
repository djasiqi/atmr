// frontend/src/hooks/useSocketStatus.js

import { useEffect, useState } from 'react';
import { getCompanySocket } from '../services/companySocket';

/**
 * Hook pour exposer l'état de connexion Socket.IO
 * @returns {Object} { connected, reconnecting, latency, lastConnected }
 */
export function useSocketStatus() {
  const [connected, setConnected] = useState(false);
  const [reconnecting, setReconnecting] = useState(false);
  const [latency] = useState(null); // Réservé pour usage futur
  const [lastConnected, setLastConnected] = useState(null);

  useEffect(() => {
    const socket = getCompanySocket();
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

    const handleReconnect = (_attemptNumber) => {
      setReconnecting(true);
      // Après reconnexion réussie, handleConnect sera appelé
    };

    // Écouter les pong pour calculer la latence
    const handlePong = (_data) => {
      // La latence est calculée dans companySocket.js
      // On peut l'exposer via un événement personnalisé ou via un getter
      // Pour l'instant, on ne track pas la latence ici car elle est déjà loggée
      // On pourrait ajouter un système d'événements personnalisés si nécessaire
    };

    // Attacher les listeners
    socket.on('connect', handleConnect);
    socket.on('disconnect', handleDisconnect);
    socket.on('reconnect', handleReconnect);
    socket.on('pong', handlePong);

    // Cleanup
    return () => {
      socket.off('connect', handleConnect);
      socket.off('disconnect', handleDisconnect);
      socket.off('reconnect', handleReconnect);
      socket.off('pong', handlePong);
    };
  }, []);

  return {
    connected,
    reconnecting,
    latency, // null pour l'instant, peut être enrichi plus tard
    lastConnected,
  };
}

