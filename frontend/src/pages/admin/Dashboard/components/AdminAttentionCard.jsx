import React from 'react';
import { Link } from 'react-router-dom';
import { FiChevronRight } from 'react-icons/fi';
import styles from './AdminAttentionCard.module.css';

/**
 * Carte d’attention dashboard admin (À traiter).
 * @param {{ title: string, value: string|number, explanation: string, to: string, variant?: 'attention'|'ok'|'danger', linkLabel?: string }} props
 */
export default function AdminAttentionCard({
  title,
  value,
  explanation,
  to,
  variant = 'attention',
  linkLabel = 'Voir',
}) {
  const variantClass =
    variant === 'ok'
      ? styles.ok
      : variant === 'danger'
        ? styles.danger
        : styles.attention;

  return (
    <Link to={to} className={`${styles.card} ${variantClass}`}>
      <div className={styles.top}>
        <span className={styles.title}>{title}</span>
        <FiChevronRight className={styles.chevron} size={16} aria-hidden />
      </div>
      <p className={styles.value}>{value}</p>
      <p className={styles.explanation}>{explanation}</p>
      <span className={styles.linkHint}>{linkLabel}</span>
    </Link>
  );
}
