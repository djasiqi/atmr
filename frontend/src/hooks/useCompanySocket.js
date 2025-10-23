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

    // Écoute les événements de connexion
    const handleConnect = () => {
      // eslint-disable-next-line no-console
      console.log('[useCompanySocket] Socket connecté');
    };

    const handleDisconnect = () => {
      // eslint-disable-next-line no-console
      console.log('[useCompanySocket] Socket déconnecté');
    };

    socketInstance.on('connect', handleConnect);
    socketInstance.on('disconnect', handleDisconnect);

    // Cleanup
    return () => {
      socketInstance.off('connect', handleConnect);
      socketInstance.off('disconnect', handleDisconnect);
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
