// frontend/src/pages/company/Settings/tabs/components/KPICards.jsx
import React from 'react';
import {
  FiUsers,
  FiArrowUpRight,
  FiArrowDownLeft,
  FiDollarSign,
  FiTrendingUp,
  FiBarChart2,
} from 'react-icons/fi';
import styles from './KPICards.module.css';

const KPICards = ({ stats, loading }) => {
  if (loading) {
    return (
      <div className={styles.kpiContainer}>
        {[1, 2, 3, 4, 5, 6].map((i) => (
          <div key={i} className={styles.kpiCard}>
            <div className={styles.skeleton} />
          </div>
        ))}
      </div>
    );
  }

  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('fr-CH', {
      style: 'currency',
      currency: 'CHF',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(amount);
  };

  const kpis = [
    {
      label: 'Partenaires actifs',
      value: stats?.active_partnerships || 0,
      Icon: FiUsers,
      tone: 'success',
    },
    {
      label: 'Courses envoyées',
      value: stats?.sent_transfers_current_month || 0,
      Icon: FiArrowUpRight,
      tone: 'neutral',
      subtitle: 'mois en cours',
    },
    {
      label: 'Courses reçues',
      value: stats?.received_transfers_current_month || 0,
      Icon: FiArrowDownLeft,
      tone: 'warning',
      subtitle: 'mois en cours',
    },
    {
      label: 'À payer',
      value: formatCurrency(stats?.amount_to_pay || 0),
      Icon: FiDollarSign,
      tone: 'danger',
    },
    {
      label: 'À recevoir',
      value: formatCurrency(stats?.amount_to_receive || 0),
      Icon: FiTrendingUp,
      tone: 'success',
    },
    {
      label: 'Solde net',
      value: formatCurrency(stats?.net_balance || 0),
      Icon: FiBarChart2,
      tone: (stats?.net_balance ?? 0) >= 0 ? 'success' : 'danger',
      highlight: true,
    },
  ];

  return (
    <div className={styles.kpiContainer}>
      {kpis.map((kpi, index) => (
        <div
          key={index}
          className={`${styles.kpiCard} ${kpi.highlight ? styles.highlight : ''}`}
        >
          <div
            className={`${styles.kpiIcon} ${styles[`tone${kpi.tone.charAt(0).toUpperCase() + kpi.tone.slice(1)}`]}`}
          >
            <kpi.Icon size={20} />
          </div>
          <div className={styles.kpiContent}>
            <div className={styles.kpiValue}>{kpi.value}</div>
            <div className={styles.kpiLabel}>{kpi.label}</div>
            {kpi.subtitle && (
              <div className={styles.kpiSubtitle}>{kpi.subtitle}</div>
            )}
          </div>
        </div>
      ))}
    </div>
  );
};

export default KPICards;

