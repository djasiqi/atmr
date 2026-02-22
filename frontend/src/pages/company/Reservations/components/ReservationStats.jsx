import React from 'react';
import { FiHash, FiActivity, FiCheckCircle, FiDollarSign } from 'react-icons/fi';
import styles from './ReservationStats.module.css';

function formatCompactRevenue(val) {
  const num = Number(val) || 0;
  if (num >= 1000) {
    return `${(num / 1000).toFixed(1)}k CHF`;
  }
  return `${num.toFixed(2)} CHF`;
}

const ReservationStats = ({ stats }) => {
  const items = [
    {
      label: 'Total',
      value: stats.total,
      icon: FiHash,
      accent: 'default',
    },
    {
      label: 'En cours',
      value: stats.inProgress,
      icon: FiActivity,
      accent: 'warning',
    },
    {
      label: 'Terminees',
      value: stats.completed,
      icon: FiCheckCircle,
      accent: 'success',
    },
    {
      label: 'Revenus',
      value: formatCompactRevenue(stats.revenue),
      icon: FiDollarSign,
      accent: 'info',
    },
  ];

  return (
    <div className={styles.statsGrid}>
      {items.map((item) => {
        const Icon = item.icon;
        return (
          <div
            key={item.label}
            className={`${styles.kpiCard} ${styles[`accent_${item.accent}`] || ''}`}
          >
            <div className={styles.kpiIconWrap}>
              <Icon size={18} className={styles.kpiIcon} />
            </div>
            <div className={styles.kpiContent}>
              <span className={styles.kpiLabel}>{item.label}</span>
              <span className={styles.kpiValue}>{item.value}</span>
            </div>
          </div>
        );
      })}
    </div>
  );
};

export default ReservationStats;
