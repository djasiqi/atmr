// frontend/src/hooks/useSocketStatus.js

import { useCallback, useEffect, useState } from 'react';
import {
  COMPANY_SOCKET_STATE_EVENT,
  getCompanySocket,
  getCompanySocketStatusSnapshot,
  retryCompanySocket,
} from '../services/companySocket';

/**
 * Hook pour exposer l'état de connexion Socket.IO
 * @returns {{ connected: boolean, reconnecting: boolean, latency: null, lastConnected: Date|null, reasonCode: string|null, reasonLabel: string|null, retry: Function }}
 */
export function useSocketStatus() {
  const snapshot = getCompanySocketStatusSnapshot();
  const [connected, setConnected] = useState(() => Boolean(snapshot.connected));
  const [reconnecting, setReconnecting] = useState(() => Boolean(snapshot.reconnecting));
  const [latency] = useState(null);
  const [lastConnected, setLastConnected] = useState(null);
  const [reasonCode, setReasonCode] = useState(() => snapshot.reasonCode || null);
  const [reasonLabel, setReasonLabel] = useState(() => snapshot.reasonLabel || null);

  const applyDetail = useCallback((d) => {
    if (!d || typeof d !== 'object') return;
    if (typeof d.connected === 'boolean') {
      setConnected(d.connected);
      if (d.connected) {
        setLastConnected(new Date());
        setReasonCode(null);
        setReasonLabel(null);
      }
    }
    if (typeof d.reconnecting === 'boolean') {
      setReconnecting(d.reconnecting);
    }
    if (!d.connected) {
      if (Object.prototype.hasOwnProperty.call(d, 'reasonCode')) {
        setReasonCode(d.reasonCode || null);
      }
      if (Object.prototype.hasOwnProperty.call(d, 'reasonLabel')) {
        setReasonLabel(d.reasonLabel || null);
      }
    }
  }, []);

  useEffect(() => {
    const socket = getCompanySocket();
    const snap = getCompanySocketStatusSnapshot();
    applyDetail(snap);

    if (socket?.connected) {
      setConnected(true);
      setLastConnected(new Date());
      setReasonCode(null);
      setReasonLabel(null);
    }

    const handleDocumentState = (e) => {
      applyDetail(e.detail || {});
    };

    window.addEventListener(COMPANY_SOCKET_STATE_EVENT, handleDocumentState);

    let s = socket;
    const handleConnect = () => {
      setConnected(true);
      setReconnecting(false);
      setLastConnected(new Date());
      setReasonCode(null);
      setReasonLabel(null);
    };
    const handleDisconnect = () => {
      setConnected(false);
      setReconnecting(false);
    };
    const handleReconnect = () => setReconnecting(true);

    if (s) {
      s.on('connect', handleConnect);
      s.on('disconnect', handleDisconnect);
      s.on('reconnect', handleReconnect);
    }

    return () => {
      window.removeEventListener(COMPANY_SOCKET_STATE_EVENT, handleDocumentState);
      if (s) {
        s.off('connect', handleConnect);
        s.off('disconnect', handleDisconnect);
        s.off('reconnect', handleReconnect);
      }
    };
  }, [applyDetail]);

  const retry = useCallback(() => {
    setReconnecting(true);
    retryCompanySocket();
  }, []);

  return {
    connected,
    reconnecting,
    latency,
    lastConnected,
    reasonCode,
    reasonLabel,
    retry,
  };
}
