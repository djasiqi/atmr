// frontend/src/components/common/SocketStatusBadge.jsx

import React, { useEffect, useId, useRef, useState } from 'react';
import { useSocketStatus } from '../../hooks/useSocketStatus';
import styles from './SocketStatusBadge.module.css';

/**
 * Badge visuel indiquant le statut de connexion Socket.IO
 * - Vert : connecté
 * - Rouge : déconnecté (+ cause au clic / tooltip)
 * - Orange : reconnexion en cours
 */
export default function SocketStatusBadge() {
  const { connected, reconnecting, lastConnected, reasonLabel, reasonCode, retry } =
    useSocketStatus();
  const [open, setOpen] = useState(false);
  const rootRef = useRef(null);
  const panelId = useId();

  let statusClass = styles.disconnected;
  let statusText = 'Déconnecté';
  let statusIcon = '●';

  if (reconnecting) {
    statusClass = styles.reconnecting;
    statusText = 'Reconnexion...';
    statusIcon = '⟳';
  } else if (connected) {
    statusClass = styles.connected;
    statusText = 'Connecté';
    statusIcon = '●';
  }

  const explanation =
    reasonLabel ||
    (connected
      ? null
      : lastConnected
        ? 'Connexion temps réel coupée'
        : 'Jamais connecté au serveur temps réel');

  const tooltipParts = [];
  if (connected && lastConnected) {
    tooltipParts.push(`Dernière connexion : ${lastConnected.toLocaleTimeString('fr-FR')}`);
  } else if (explanation) {
    tooltipParts.push(explanation);
  }
  if (reasonCode && !connected) {
    tooltipParts.push(`Code : ${reasonCode}`);
  }
  const tooltipText = tooltipParts.join(' — ') || statusText;

  useEffect(() => {
    if (!open) return undefined;
    const onDoc = (e) => {
      if (rootRef.current && !rootRef.current.contains(e.target)) {
        setOpen(false);
      }
    };
    const onKey = (e) => {
      if (e.key === 'Escape') setOpen(false);
    };
    document.addEventListener('mousedown', onDoc);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDoc);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  const showDetail = !connected || reconnecting;

  return (
    <div className={styles.root} ref={rootRef}>
      <button
        type="button"
        className={`${styles.badge} ${statusClass}`}
        title={tooltipText}
        aria-label={`Statut connexion : ${statusText}${explanation && !connected ? `. ${explanation}` : ''}`}
        aria-expanded={showDetail ? open : undefined}
        aria-controls={showDetail ? panelId : undefined}
        onClick={() => {
          if (showDetail) setOpen((v) => !v);
        }}
      >
        <span className={styles.icon} aria-hidden="true">
          {statusIcon}
        </span>
        <span className={styles.text}>{statusText}</span>
      </button>

      {open && showDetail && (
        <div
          id={panelId}
          className={styles.detail}
          role="status"
          aria-live="polite"
        >
          <p className={styles.detailTitle}>{statusText}</p>
          <p className={styles.detailReason}>{explanation}</p>
          {reasonCode ? (
            <p className={styles.detailCode}>Réf. : {reasonCode}</p>
          ) : null}
          {lastConnected ? (
            <p className={styles.detailMeta}>
              Dernière connexion : {lastConnected.toLocaleTimeString('fr-FR')}
            </p>
          ) : (
            <p className={styles.detailMeta}>Aucune connexion réussie depuis l’ouverture de la page.</p>
          )}
          <button
            type="button"
            className={styles.retryBtn}
            onClick={() => {
              retry();
              setOpen(false);
            }}
          >
            Réessayer
          </button>
        </div>
      )}
    </div>
  );
}
