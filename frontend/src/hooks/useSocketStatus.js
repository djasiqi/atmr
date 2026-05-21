// frontend/src/hooks/useSocketStatus.js

import { useEffect, useState } from 'react';
import { COMPANY_SOCKET_STATE_EVENT, getCompanySocket } from '../services/companySocket';

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
    // État initial (instance courante peut être remplacée plus tard sans remonter le hook)
    if (!socket) {
      setConnected(false);
      setReconnecting(false);
    } else {
      setConnected(Boolean(socket.connected));
      setReconnecting(false);
      if (socket.connected) {
        setLastConnected(new Date());
      }
    }

    const handleDocumentState = (e) => {
      const d = e.detail || {};
      if (typeof d.connected === 'boolean') {
        setConnected(d.connected);
        if (d.connected) {
          setLastConnected(new Date());
        }
      }
      if (typeof d.reconnecting === 'boolean') {
        setReconnecting(d.reconnecting);
      }
    };

    window.addEventListener(COMPANY_SOCKET_STATE_EVENT, handleDocumentState);

    // Les handlers socket ci-dessous ne survivent pas à disposeCompanySocketInstance() ;
    // l’événement document assure la synchro après recréation de l’instance.
    let s = socket;
    const handleConnect = () => {
      setConnected(true);
      setReconnecting(false);
      setLastConnected(new Date());
    };
    const handleDisconnect = () => {
      setConnected(false);
      setReconnecting(false);
    };
    const handleReconnect = () => setReconnecting(true);
    const handlePong = () => {};

    if (s) {
      s.on('connect', handleConnect);
      s.on('disconnect', handleDisconnect);
      s.on('reconnect', handleReconnect);
      s.on('pong', handlePong);
    }

    return () => {
      window.removeEventListener(COMPANY_SOCKET_STATE_EVENT, handleDocumentState);
      if (s) {
        s.off('connect', handleConnect);
        s.off('disconnect', handleDisconnect);
        s.off('reconnect', handleReconnect);
        s.off('pong', handlePong);
      }
    };
  }, []);

  return {
    connected,
    reconnecting,
    latency, // null pour l'instant, peut être enrichi plus tard
    lastConnected,
  };
}

