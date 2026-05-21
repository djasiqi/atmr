// frontend/src/hooks/useCompanySocket.js
import { useEffect, useState } from 'react';
import { COMPANY_SOCKET_STATE_EVENT, getCompanySocket } from '../services/companySocket';

/**
 * Hook React qui utilise le service singleton companySocket.
 * La reconnexion est déléguée à Socket.IO (pas de boucle parallèle dans le hook).
 */
export default function useCompanySocket() {
  const [socket, setSocket] = useState(null);

  useEffect(() => {
    const socketInstance = getCompanySocket();

    if (!socketInstance) {
      setSocket(null);
      return undefined;
    }

    setSocket(socketInstance);

    return undefined;
  }, []);

  return socket;
}

export function useSocketConnected() {
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    const socketInstance = getCompanySocket();
    if (!socketInstance) {
      setIsConnected(false);
      return undefined;
    }

    const handleConnect = () => setIsConnected(true);
    const handleDisconnect = () => setIsConnected(false);

    socketInstance.on('connect', handleConnect);
    socketInstance.on('disconnect', handleDisconnect);
    setIsConnected(socketInstance.connected);

    const handleStateEvent = (event) => {
      const detail = event?.detail;
      if (detail && typeof detail.connected === 'boolean') {
        setIsConnected(detail.connected);
      }
    };

    if (typeof window !== 'undefined') {
      window.addEventListener(COMPANY_SOCKET_STATE_EVENT, handleStateEvent);
    }

    return () => {
      socketInstance.off('connect', handleConnect);
      socketInstance.off('disconnect', handleDisconnect);
      if (typeof window !== 'undefined') {
        window.removeEventListener(COMPANY_SOCKET_STATE_EVENT, handleStateEvent);
      }
    };
  }, []);

  return isConnected;
}
