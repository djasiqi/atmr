import React from 'react';
import styles from './ReservationAlerts.module.css';

const ReservationAlerts = ({ alerts }) => {
  const getAlertIcon = (type) => {
    switch (type) {
      case 'delay':
        return '⏰';
      case 'unassigned':
        return '👤';
      case 'urgent':
        return '🚨';
      default:
        return '⚠️';
    }
  };

  const getSeverityClass = (severity) => {
    switch (severity) {
      case 'high':
        return styles.high;
      case 'medium':
        return styles.medium;
      case 'low':
        return styles.low;
      default:
        return styles.medium;
    }
  };

  if (alerts.length === 0) {
    return null;
  }

  return (
    <div className={styles.alertsContainer}>
      <div className={styles.alertsHeader}>
        <span className={styles.alertsIcon}>🔔</span>
        <span className={styles.alertsTitle}>Alertes ({alerts.length})</span>
      </div>
      <div className={styles.alertsList}>
        {alerts.slice(0, 3).map((alert) => (
          <div key={alert.id} className={`${styles.alert} ${getSeverityClass(alert.severity)}`}>
            <span className={styles.alertIcon}>{getAlertIcon(alert.type)}</span>
            <span className={styles.alertMessage}>{alert.message}</span>
          </div>
        ))}
        {alerts.length > 3 && (
          <div className={styles.moreAlerts}>+{alerts.length - 3} autres alertes</div>
        )}
      </div>
    </div>
  );
};

export default ReservationAlerts;
