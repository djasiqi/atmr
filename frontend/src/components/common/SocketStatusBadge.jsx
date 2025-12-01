// frontend/src/components/common/SocketStatusBadge.jsx

import React from 'react';
import { useSocketStatus } from '../../hooks/useSocketStatus';
import styles from './SocketStatusBadge.module.css';

/**
 * Badge visuel indiquant le statut de connexion Socket.IO
 * - Vert : connecté
 * - Rouge : déconnecté
 * - Orange : reconnexion en cours
 */
export default function SocketStatusBadge() {
  const { connected, reconnecting, lastConnected } = useSocketStatus();

  // Déterminer le statut et le style
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

  // Formater la date de dernière connexion pour le tooltip
  const tooltipText = lastConnected
    ? `Dernière connexion: ${lastConnected.toLocaleTimeString('fr-FR')}`
    : 'Jamais connecté';

  return (
    <div
      className={`${styles.badge} ${statusClass}`}
      title={tooltipText}
      aria-label={`Statut connexion: ${statusText}`}
    >
      <span className={styles.icon} aria-hidden="true">
        {statusIcon}
      </span>
      <span className={styles.text}>{statusText}</span>
    </div>
  );
}

