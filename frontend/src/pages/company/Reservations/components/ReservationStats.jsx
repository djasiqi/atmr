import React from 'react';
import styles from './ReservationStats.module.css';

const ReservationStats = ({ stats }) => {
  const statItems = [
    {
      label: 'Total',
      value: stats.total,
      icon: '📅',
      color: '#00796b',
    },
    {
      label: 'En cours',
      value: stats.inProgress,
      icon: '🚗',
      color: '#ff9800',
    },
    {
      label: 'Terminées',
      value: stats.completed,
      icon: '✅',
      color: '#4caf50',
    },
    {
      label: 'Annulées',
      value: stats.canceled,
      icon: '❌',
      color: '#f44336',
    },
    {
      label: 'Revenus',
      value: `${stats.revenue.toFixed(2)} CHF`,
      icon: '💰',
      color: '#2196f3',
    },
  ];

  return (
    <div className={styles.statsContainer}>
      {statItems.map((item, index) => (
        <div key={index} className={styles.statCard}>
          <div className={styles.statIcon} style={{ color: item.color }}>
            {item.icon}
          </div>
          <div className={styles.statContent}>
            <h3 className={styles.statLabel}>{item.label}</h3>
            <p className={styles.statValue}>{item.value}</p>
          </div>
        </div>
      ))}
    </div>
  );
};

export default ReservationStats;
