import React from 'react';
import { Link } from 'react-router-dom';
import { FiAlertTriangle, FiCheck, FiHelpCircle, FiLoader } from 'react-icons/fi';
import styles from './AdminHealthStatus.module.css';

/**
 * Santé plateforme — loading | ok | degraded | unknown.
 * @param {{ status: 'loading'|'ok'|'degraded'|'unknown', detail?: string, detailsTo?: string }} props
 */
export default function AdminHealthStatus({ status, detail, detailsTo }) {
  if (status === 'loading') {
    return (
      <div className={`${styles.row} ${styles.loading}`} role="status">
        <FiLoader size={14} aria-hidden />
        <span>Vérification de la plateforme…</span>
      </div>
    );
  }

  if (status === 'ok') {
    return (
      <div className={`${styles.row} ${styles.ok}`} role="status">
        <FiCheck size={14} aria-hidden />
        <span>Plateforme opérationnelle</span>
      </div>
    );
  }

  if (status === 'degraded') {
    return (
      <div className={`${styles.banner} ${styles.degraded}`} role="status">
        <div className={styles.bannerMain}>
          <FiAlertTriangle size={16} aria-hidden />
          <div>
            <strong>Attention requise</strong>
            {detail ? <p className={styles.bannerDetail}>{detail}</p> : null}
          </div>
        </div>
        {detailsTo ? (
          <Link to={detailsTo} className={styles.bannerLink}>
            Voir les détails
          </Link>
        ) : null}
      </div>
    );
  }

  return (
    <div className={`${styles.row} ${styles.unknown}`} role="status">
      <FiHelpCircle size={14} aria-hidden />
      <span>État indisponible</span>
    </div>
  );
}
