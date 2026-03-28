import React from 'react';
import styles from './StatusBadge.module.css';

const LABELS = {
  ok: 'OK',
  degraded: 'Dégradé',
  down: 'Indisponible',
  unavailable: 'Indisponible',
  unknown: 'Inconnu',
  not_configured: 'Non configuré',
  not_implemented: 'Non implémenté',
};

/**
 * Badge normalisé pour statuts plateforme (contrat API v2 : ok / degraded / down / unknown / …).
 * @param {string} [labelOverride] — libellé affiché à la place du libellé par défaut.
 */
export default function StatusBadge({ status, title, labelOverride }) {
  const raw = String(status || 'unknown').toLowerCase();
  const key = raw === 'unavailable' ? 'down' : raw;
  const label = labelOverride || LABELS[key] || LABELS.unknown;
  const cls = styles[key] || styles.unknown;

  return (
    <span className={`${styles.badge} ${cls}`} title={title || label}>
      {label}
    </span>
  );
}
