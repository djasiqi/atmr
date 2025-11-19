import React from 'react';
import styles from './TopClients.module.css';

const TopClients = ({ reservations }) => {
  // Calculer le top des clients
  const clientStats = reservations.reduce((acc, reservation) => {
    const clientName =
      reservation.customer_name || reservation.client?.full_name || 'Client anonyme';
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
    .slice(0, 3); // Top 3 uniquement

  return (
    <div className={styles.topClientsContainer}>
      <div className={styles.topClientsHeader}>
        <span className={styles.topClientsIcon}>🏆</span>
        <span className={styles.topClientsTitle}>Top 3 Clients</span>
      </div>
      <div className={styles.topClientsList}>
        {topClients.length === 0 ? (
          <div className={styles.noData}>Aucune donnée disponible</div>
        ) : (
          topClients.map((client, index) => (
            <div key={index} className={styles.clientItem}>
              <div className={styles.clientRank}>#{index + 1}</div>
              <div className={styles.clientInfo}>
                <div className={styles.clientName}>{client.name}</div>
                <div className={styles.clientStats}>
                  {client.count} course{client.count > 1 ? 's' : ''} • {client.revenue.toFixed(2)}{' '}
                  CHF
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default TopClients;
