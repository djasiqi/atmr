import React from 'react';

/**
 * Composant d'affichage du résumé et des statistiques
 */
const DispatchSummary = ({ summary, delays = [], styles = {} }) => {
  if (!summary) return null;

  return (
    <div className={styles.summaryCards}>
      <div className={styles.summaryCard}>
        <div className={styles.cardIcon}>📊</div>
        <div className={styles.cardContent}>
          <p>Courses assignées</p>
          <h3>{summary.assigned || 0}</h3>
        </div>
      </div>
      <div className={styles.summaryCard}>
        <div className={styles.cardIcon}>⏳</div>
        <div className={styles.cardContent}>
          <p>Courses en attente</p>
          <h3>{summary.pending || 0}</h3>
        </div>
      </div>
      <div className={styles.summaryCard}>
        <div className={styles.cardIcon}>⚠️</div>
        <div className={styles.cardContent}>
          <p>Retards détectés</p>
          <h3>{Array.isArray(delays) ? delays.length : 0}</h3>
        </div>
      </div>
      <div className={styles.summaryCard}>
        <div className={styles.cardIcon}>✅</div>
        <div className={styles.cardContent}>
          <p>Taux d'assignation</p>
          <h3>{summary.total > 0 ? Math.round((summary.assigned / summary.total) * 100) : 0}%</h3>
        </div>
      </div>
    </div>
  );
};

export default DispatchSummary;
