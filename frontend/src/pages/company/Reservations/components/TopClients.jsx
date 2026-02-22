import React from 'react';
import { FiUsers, FiChevronDown, FiChevronUp } from 'react-icons/fi';
import styles from './TopClients.module.css';

const TopClients = ({ reservations, isOpen, onToggle }) => {
  const clientStats = reservations.reduce((acc, reservation) => {
    const clientName =
      reservation.client_name || reservation.client?.full_name || 'Client anonyme';
    if (!acc[clientName]) {
      acc[clientName] = {
        name: clientName,
        count: 0,
        revenue: 0,
      };
    }
    acc[clientName].count++;
    acc[clientName].revenue += Number(reservation.amount || 0);
    return acc;
  }, {});

  const topClients = Object.values(clientStats)
    .sort((a, b) => b.count - a.count)
    .slice(0, 3);

  const ChevronIcon = isOpen ? FiChevronUp : FiChevronDown;

  return (
    <div className={styles.topClientsSection}>
      <button
        type="button"
        className={styles.topClientsHeader}
        onClick={onToggle}
        aria-expanded={isOpen}
      >
        <FiUsers size={16} className={styles.headerIcon} />
        <span className={styles.headerTitle}>Top clients</span>
        <ChevronIcon size={16} className={styles.headerChevron} />
      </button>

      <div className={`${styles.topClientsBody} ${isOpen ? styles.bodyOpen : ''}`}>
        {topClients.length === 0 ? (
          <div className={styles.noData}>Aucune donnee disponible</div>
        ) : (
          <div className={styles.clientsList}>
            {topClients.map((client, index) => (
              <div key={client.name} className={styles.clientItem}>
                <div className={styles.clientRank}>#{index + 1}</div>
                <div className={styles.clientInfo}>
                  <div className={styles.clientName}>{client.name}</div>
                  <div className={styles.clientStats}>
                    {client.count} course{client.count > 1 ? 's' : ''} - {client.revenue.toFixed(2)} CHF
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default TopClients;
