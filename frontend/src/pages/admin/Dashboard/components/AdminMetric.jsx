import React from 'react';
import styles from './AdminMetric.module.css';

/**
 * Indicateur métier compact.
 * @param {{ label: string, value: string|number, hint?: string }} props
 */
export default function AdminMetric({ label, value, hint }) {
  return (
    <article className={styles.metric}>
      <span className={styles.label}>{label}</span>
      <strong className={styles.value}>{value}</strong>
      {hint ? <span className={styles.hint}>{hint}</span> : null}
    </article>
  );
}
