// src/pages/company/Dashboard/components/QuickAssignPanel.jsx
import React, { useState, useMemo } from 'react';
import {
  FiX,
  FiUser,
  FiMapPin,
  FiClock,
  FiAlertTriangle,
  FiCheck,
} from 'react-icons/fi';
import styles from './QuickAssignPanel.module.css';

const QuickAssignPanel = ({
  isOpen,
  onClose,
  opportunity,
  booking,
  drivers,
  onAssign,
  assigning,
}) => {
  const [selectedDriverId, setSelectedDriverId] = useState(null);

  const resolveDriverAvailability = (driver) => {
    const backendStatus = String(driver?.status || '').toLowerCase();
    if (backendStatus === 'busy') return { label: 'En course', isAvailable: false };
    if (backendStatus === 'assigned') return { label: 'Assigné', isAvailable: false };
    if (backendStatus === 'offline') return { label: 'Hors ligne', isAvailable: false };
    if (backendStatus === 'available') return { label: 'Disponible', isAvailable: true };
    return { label: 'Indisponible', isAvailable: false };
  };

  const sortedDrivers = useMemo(() => {
    if (!drivers) return [];
    return [...drivers]
      .filter((d) => d.is_active)
      .sort((a, b) => {
        const aAvail = resolveDriverAvailability(a).isAvailable;
        const bAvail = resolveDriverAvailability(b).isAvailable;
        if (aAvail && !bAvail) return -1;
        if (!aAvail && bAvail) return 1;
        const nameA = a.full_name || a.username || '';
        const nameB = b.full_name || b.username || '';
        return nameA.localeCompare(nameB);
      })
      .slice(0, 8);
  }, [drivers]);

  if (!isOpen || !opportunity) return null;

  const delayMinutes = opportunity.current_delay_minutes;
  const bookingId = booking?.id || opportunity.booking_id;

  const handleConfirm = () => {
    if (!selectedDriverId || !bookingId) return;
    onAssign(bookingId, selectedDriverId);
  };

  const actionLabel =
    delayMinutes >= 15
      ? 'Réassigner'
      : !booking?.driver_id
        ? 'Assigner'
        : 'Optimiser';

  return (
    <>
      <div className={styles.backdrop} onClick={onClose} />
      <aside
        className={styles.panel}
        role="dialog"
        aria-label="Assignation rapide"
        data-tour-id="quick-assign-panel"
      >
        <header className={styles.panelHeader}>
          <h3 className={styles.panelTitle}>Assignation rapide</h3>
          <button className={styles.closeBtn} onClick={onClose} title="Fermer">
            <FiX size={18} />
          </button>
        </header>

        <div className={styles.panelBody}>
          <section className={styles.bookingSummary}>
            <div className={styles.summaryRow}>
              <FiUser size={14} className={styles.summaryIcon} />
              <div>
                <span className={styles.summaryLabel}>Patient</span>
                <span className={styles.summaryValue}>
                  {booking?.client?.full_name || booking?.client_name || 'N/A'}
                </span>
              </div>
            </div>

            {booking?.client?.institution_name && (
              <div className={styles.summaryRow}>
                <FiMapPin size={14} className={styles.summaryIcon} />
                <div>
                  <span className={styles.summaryLabel}>Institution</span>
                  <span className={styles.summaryValue}>{booking.client.institution_name}</span>
                </div>
              </div>
            )}

            <div className={styles.summaryRow}>
              <FiMapPin size={14} className={styles.summaryIcon} />
              <div>
                <span className={styles.summaryLabel}>Trajet</span>
                <span className={styles.summaryValue}>
                  {booking?.pickup_location || '?'} → {booking?.dropoff_location || '?'}
                </span>
              </div>
            </div>

            <div className={styles.summaryRow}>
              <FiClock size={14} className={styles.summaryIcon} />
              <div>
                <span className={styles.summaryLabel}>Heure prévue</span>
                <span className={styles.summaryValue}>
                  {booking?.scheduled_time
                    ? new Date(booking.scheduled_time).toLocaleTimeString('fr-CH', {
                        hour: '2-digit',
                        minute: '2-digit',
                      })
                    : 'N/A'}
                </span>
              </div>
            </div>

            {delayMinutes > 0 && (
              <div className={`${styles.summaryRow} ${styles.summaryDelay}`}>
                <FiAlertTriangle size={14} className={styles.summaryIcon} />
                <div>
                  <span className={styles.summaryLabel}>Retard</span>
                  <span className={styles.summaryValueDanger}>+{Math.round(delayMinutes)} min</span>
                </div>
              </div>
            )}
          </section>

          <section className={styles.driverSection} data-tour-id="quick-assign-driver-section">
            <h4 className={styles.driverSectionTitle}>
              Chauffeurs disponibles ({sortedDrivers.filter((d) => resolveDriverAvailability(d).isAvailable).length})
            </h4>

            <div className={styles.driverList}>
              {sortedDrivers.length === 0 && (
                <p className={styles.noDrivers}>Aucun chauffeur actif</p>
              )}
              {sortedDrivers.map((d) => {
                const isSelected = selectedDriverId === d.id;
                const driverName = d.full_name || d.username || `Chauffeur #${d.id}`;
                const availability = resolveDriverAvailability(d);
                return (
                  <button
                    key={d.id}
                    data-tour-id="quick-assign-driver-option"
                    className={`${styles.driverCard} ${isSelected ? styles.driverCardSelected : ''} ${!availability.isAvailable ? styles.driverCardBusy : ''}`}
                    onClick={() => setSelectedDriverId(d.id)}
                    type="button"
                  >
                    <div className={styles.driverInfo}>
                      <span className={styles.driverName}>{driverName}</span>
                      <span
                        className={`${styles.driverStatus} ${availability.isAvailable ? styles.statusAvailable : styles.statusBusy}`}
                      >
                        {availability.label}
                      </span>
                    </div>
                    {isSelected && <FiCheck size={16} className={styles.checkIcon} />}
                  </button>
                );
              })}
            </div>
          </section>
        </div>

        <footer className={styles.panelFooter}>
          <button
            data-tour-id="quick-assign-confirm"
            className={styles.confirmBtn}
            disabled={!selectedDriverId || assigning}
            onClick={handleConfirm}
            type="button"
          >
            {assigning ? 'Assignation en cours...' : `${actionLabel} ce chauffeur`}
          </button>
        </footer>
      </aside>
    </>
  );
};

export default QuickAssignPanel;
