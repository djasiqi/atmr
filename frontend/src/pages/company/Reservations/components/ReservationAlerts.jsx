import React from 'react';
import { FiAlertCircle, FiClock, FiUserX } from 'react-icons/fi';
import { formatDelay } from '../../../../utils/formatDelay';
import styles from './ReservationAlerts.module.css';

const ReservationAlerts = ({ alerts, onFilterByAlert }) => {
  if (!alerts || alerts.length === 0) {
    return null;
  }

  const delayAlerts = alerts.filter((a) => a.type === 'delay');
  const unassignedAlert = alerts.find((a) => a.type === 'unassigned');

  const delayCount = delayAlerts.length;
  const unassignedCount = unassignedAlert?.count || 0;

  if (delayCount === 0 && unassignedCount === 0) {
    return null;
  }

  const maxDelayMinutes = delayCount > 0
    ? Math.max(...delayAlerts.map((a) => {
        const scheduled = new Date(a.reservation?.scheduled_time);
        const now = new Date();
        return Math.floor((now - scheduled) / (1000 * 60));
      }).filter((v) => v > 0))
    : 0;

  const maxDelayFormatted = formatDelay(maxDelayMinutes);

  return (
    <div className={styles.alertBar}>
      <FiAlertCircle size={13} className={styles.barIcon} />

      {delayCount > 0 && (
        <button
          type="button"
          className={styles.alertChip}
          onClick={() => onFilterByAlert?.('delays')}
        >
          <FiClock size={12} />
          <span>
            {delayCount} retard{delayCount > 1 ? 's' : ''}
            {maxDelayFormatted && ` (max ${maxDelayFormatted})`}
          </span>
        </button>
      )}

      {delayCount > 0 && unassignedCount > 0 && (
        <span className={styles.barSep} />
      )}

      {unassignedCount > 0 && (
        <button
          type="button"
          className={styles.alertChip}
          onClick={() => onFilterByAlert?.('unassigned')}
        >
          <FiUserX size={12} />
          <span>{unassignedCount} sans chauffeur</span>
        </button>
      )}
    </div>
  );
};

export default ReservationAlerts;
