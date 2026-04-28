import { useEffect, useState } from 'react';
import { getCompanySocket } from '../../services/companySocket';

/**
 * État de connexion du client Socket.IO entreprise (dashboard LIRIE).
 * @returns {boolean}
 */
export function useCompanySocketConnected() {
  const [connected, setConnected] = useState(() => {
    const s = getCompanySocket();
    return Boolean(s?.connected);
  });

  useEffect(() => {
    const socket = getCompanySocket();
    if (!socket) {
      setConnected(false);
      return undefined;
    }
    const onConnect = () => setConnected(true);
    const onDisconnect = () => setConnected(false);
    socket.on('connect', onConnect);
    socket.on('disconnect', onDisconnect);
    setConnected(Boolean(socket.connected));
    return () => {
      socket.off('connect', onConnect);
      socket.off('disconnect', onDisconnect);
    };
  }, []);

  return connected;
}
