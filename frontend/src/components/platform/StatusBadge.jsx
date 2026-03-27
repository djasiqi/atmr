import React from 'react';
import styles from './StatusBadge.module.css';

const LABELS = {
  ok: 'OK',
  degraded: 'Dégradé',
  unavailable: 'Indisponible',
  unknown: 'Inconnu',
};

/**
 * Badge normalisé pour statuts plateforme (ok / degraded / unavailable / unknown).
 * @param {string} [labelOverride] — libellé affiché à la place du libellé par défaut (ex. démo non configurée).
 */
export default function StatusBadge({ status, title, labelOverride }) {
  const key = String(status || 'unknown').toLowerCase();
  const label = labelOverride || LABELS[key] || LABELS.unknown;
  const cls = styles[key] || styles.unknown;

  return (
    <span className={`${styles.badge} ${cls}`} title={title || label}>
      {label}
    </span>
  );
}
