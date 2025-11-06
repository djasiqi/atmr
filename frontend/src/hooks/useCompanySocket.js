// frontend/src/hooks/useCompanySocket.js
import { useEffect, useState } from 'react';
import { getCompanySocket } from '../services/companySocket';

/**
 * Hook React qui utilise le service singleton companySocket
 * Au lieu de créer sa propre instance
 */
export default function useCompanySocket() {
  const [socket, setSocket] = useState(null);

  useEffect(() => {
    // Récupère ou crée le socket singleton
    const socketInstance = getCompanySocket();

    if (!socketInstance) {
      // eslint-disable-next-line no-console
      console.warn('[useCompanySocket] Impossible de créer le socket');
      return;
    }

    setSocket(socketInstance);

    // ✅ Auto-reconnect avec exponential backoff
    let reconnectAttempts = 0;
    let reconnectDelay = 1000; // Start at 1s
    const maxReconnectDelay = 30000; // Max 30s
    let reconnectTimeoutId = null;

    const handleConnect = () => {
      // eslint-disable-next-line no-console
      console.log('[useCompanySocket] Socket connecté');
      // Réinitialiser les compteurs de reconnexion après connexion réussie
      reconnectAttempts = 0;
      reconnectDelay = 1000;
    };

    const handleDisconnect = (reason) => {
      // eslint-disable-next-line no-console
      console.log('[useCompanySocket] Socket déconnecté:', reason);

      // ✅ Auto-reconnect si déconnecté (sauf si disconnect volontaire)
      if (reason === 'io server disconnect') {
        // Le serveur a déconnecté, on ne reconnecte pas automatiquement
        console.log('[useCompanySocket] Déconnexion serveur, reconnexion automatique désactivée');
        return;
      }

      // ✅ Tentative de reconnexion avec exponential backoff
      const attemptReconnect = () => {
        if (!socketInstance.disconnected) {
          return; // Déjà reconnecté
        }

        reconnectAttempts += 1;
        console.log(`[useCompanySocket] Tentative de reconnexion ${reconnectAttempts}...`);

        // Socket.IO gère déjà la reconnexion automatique, mais on peut forcer
        if (socketInstance && socketInstance.disconnected) {
          socketInstance.connect();
        }

        // Augmenter le délai pour la prochaine tentative (exponential backoff)
        reconnectDelay = Math.min(reconnectDelay * 1.5, maxReconnectDelay);

        // Si toujours déconnecté après un délai, réessayer
        if (socketInstance.disconnected && reconnectAttempts < 10) {
          reconnectTimeoutId = setTimeout(attemptReconnect, reconnectDelay);
        } else if (reconnectAttempts >= 10) {
          console.warn('[useCompanySocket] Maximum reconnexion attempts atteint');
        }
      };

      // Démarrer la reconnexion après un court délai
      reconnectTimeoutId = setTimeout(attemptReconnect, reconnectDelay);
    };

    const handleReconnect = (attemptNumber) => {
      console.log(`[useCompanySocket] Reconnexion réussie après ${attemptNumber} tentatives`);
      reconnectAttempts = 0;
      reconnectDelay = 1000;
    };

    socketInstance.on('connect', handleConnect);
    socketInstance.on('disconnect', handleDisconnect);
    socketInstance.on('reconnect', handleReconnect);

    // Cleanup
    return () => {
      socketInstance.off('connect', handleConnect);
      socketInstance.off('disconnect', handleDisconnect);
      socketInstance.off('reconnect', handleReconnect);
      if (reconnectTimeoutId) {
        clearTimeout(reconnectTimeoutId);
      }
      // Ne pas disconnect le socket singleton ici
    };
  }, []);

  return socket;
}

// Export aussi l'état de connexion si besoin
export function useSocketConnected() {
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    const socketInstance = getCompanySocket();
    if (!socketInstance) return;

    const handleConnect = () => setIsConnected(true);
    const handleDisconnect = () => setIsConnected(false);

    socketInstance.on('connect', handleConnect);
    socketInstance.on('disconnect', handleDisconnect);

    setIsConnected(socketInstance.connected);

    return () => {
      socketInstance.off('connect', handleConnect);
      socketInstance.off('disconnect', handleDisconnect);
    };
  }, []);

  return isConnected;
}
