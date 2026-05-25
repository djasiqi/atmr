// src/components/reservations/ReservationModals.jsx
import React, { useState, useEffect } from 'react';
import { FiClock, FiZap } from 'react-icons/fi';
import Modal from '../common/Modal';
import Button from '../common/Button';
import CancellationModal from './CancellationModal';
import EditReservationModal from './EditReservationModal';
import InlineDatePicker from '../ui/InlineDatePicker';
import InlineTimePicker from '../ui/InlineTimePicker';
import styles from './ReservationModals.module.css';

const DEMO_FIRST_NAMES = [
  'Lucas',
  'Nicolas',
  'Karim',
  'Yann',
  'Alexandre',
  'Sami',
  'Mehdi',
  'Romain',
];

const DEMO_LAST_NAMES = [
  'Perrin',
  'Morel',
  'Dubois',
  'Favre',
  'Bianchi',
  'Rey',
  'Lombard',
  'Muller',
];

const hashString = (value) => {
  let hash = 0;
  const str = String(value || '');
  for (let i = 0; i < str.length; i += 1) {
    hash = (hash * 31 + str.charCodeAt(i)) >>> 0;
  }
  return hash;
};

const resolveDriverDisplayName = (driver, index) => {
  const fullName = String(driver?.full_name || '').trim();
  if (fullName) return fullName;

  const username = String(driver?.username || '').trim();
  if (/^demo_driver_/i.test(username)) {
    const seed = hashString(`${username}:${driver?.id ?? index}`);
    const first = DEMO_FIRST_NAMES[seed % DEMO_FIRST_NAMES.length];
    const last = DEMO_LAST_NAMES[Math.floor(seed / 7) % DEMO_LAST_NAMES.length];
    return `${first} ${last}`;
  }

  if (username) return username;
  return `Chauffeur ${index + 1}`;
};

const ScheduleReturnTimeModal = ({ isOpen, onClose, reservation, onConfirm }) => {
  const [selectedDate, setSelectedDate] = useState('');
  const [selectedTime, setSelectedTime] = useState('');
  const [loading, setLoading] = useState(false);

  const parseValidDate = (value) => {
    if (!value) return null;
    const date = value instanceof Date ? value : new Date(value);
    if (Number.isNaN(date.getTime())) return null;
    return date;
  };

  useEffect(() => {
    if (isOpen) {
      // Priorité: date retour existante > date aller (original_booking) > now+1h.
      const dateObj =
        parseValidDate(reservation?.scheduled_time)
        || parseValidDate(reservation?.original_booking?.scheduled_time)
        || parseValidDate(reservation?.pickup_time)
        || new Date(Date.now() + 60 * 60 * 1000);

      const year = dateObj.getFullYear();
      const month = String(dateObj.getMonth() + 1).padStart(2, '0');
      const day = String(dateObj.getDate()).padStart(2, '0');
      setSelectedDate(`${year}-${month}-${day}`);

      const hours = dateObj.getHours();
      const minutes = dateObj.getMinutes();
      if (hours === 0 && minutes === 0) {
        setSelectedTime('');
      } else {
        setSelectedTime(`${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}`);
      }
    }
  }, [isOpen, reservation]);

  const handleConfirm = async () => {
    if (!selectedDate || !selectedTime) return;

    setLoading(true);
    try {
      if (typeof onConfirm === 'function') {
        // Envoyer la date/heure locale telle quelle, sans conversion UTC via Date
        await onConfirm({ return_time: `${selectedDate}T${selectedTime}` });
      }
      onClose();
    } catch (error) {
      console.error('Erreur lors de la planification:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleUrgent = () => {
    if (onConfirm) {
      onConfirm({ urgent: true });
      onClose();
    }
  };

  if (!isOpen) return null;

  const clientName = reservation?.client_name || reservation?.client?.full_name || '';
  const formatAllerTime = (isoStr) => {
    if (!isoStr) return null;
    const d = parseValidDate(isoStr);
    if (!d) return null;
    const datePart = d.toLocaleDateString('fr-CH', { day: '2-digit', month: '2-digit', year: 'numeric' });
    if (d.getHours() === 0 && d.getMinutes() === 0) return datePart;
    const timePart = d.toLocaleTimeString('fr-CH', { hour: '2-digit', minute: '2-digit' });
    return `${datePart} ${timePart}`;
  };
  const allerTime = formatAllerTime(reservation?.original_booking?.scheduled_time)
    || formatAllerTime(reservation?.scheduled_time);

  return (
    <div className={styles.schedOverlay} onClick={onClose}>
      <div className={styles.schedModal} onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className={styles.schedHeader}>
          <div className={styles.schedHeaderIcon}><FiClock size={14} /></div>
          <h3 className={styles.schedTitle}>Planifier le retour</h3>
        </div>

        {/* Info card */}
        {reservation && (
          <div className={styles.schedInfo}>
            {clientName && (
              <div className={styles.schedInfoRow}>
                <span className={styles.schedInfoLabel}>Client</span>
                <span className={styles.schedInfoValue}>{clientName}</span>
              </div>
            )}
            {allerTime && (
              <div className={styles.schedInfoRow}>
                <span className={styles.schedInfoLabel}>Aller</span>
                <span className={styles.schedInfoValue}>{allerTime}</span>
              </div>
            )}
            {reservation.pickup_location && (
              <div className={styles.schedInfoRow}>
                <span className={styles.schedInfoLabel}>Trajet</span>
                <span className={styles.schedInfoValue}>
                  {reservation.pickup_location}
                  <span className={styles.schedArrow}> → </span>
                  {reservation.dropoff_location}
                </span>
              </div>
            )}
          </div>
        )}

        {/* Form */}
        <div className={styles.schedForm}>
          <div className={styles.schedFieldRow}>
            <div className={styles.schedField}>
              <label className={styles.schedLabel}>Date <span>*</span></label>
              <InlineDatePicker value={selectedDate} onChange={(iso) => setSelectedDate(iso)} />
            </div>
            <div className={styles.schedField}>
              <label className={styles.schedLabel}>Heure <span>*</span></label>
              <InlineTimePicker value={selectedTime} onChange={(hhmm) => setSelectedTime(hhmm)} />
            </div>
          </div>
        </div>

        {/* Actions */}
        <div className={styles.schedActions}>
          <button type="button" className={styles.schedCancel} onClick={onClose} disabled={loading}>
            Annuler
          </button>
          <button type="button" className={styles.schedUrgent} onClick={handleUrgent} disabled={loading}>
            <FiZap size={12} /> Urgent
          </button>
          <button
            type="button"
            className={styles.schedConfirm}
            onClick={handleConfirm}
            disabled={!selectedDate || !selectedTime || loading}
          >
            {loading ? 'Planification...' : 'Planifier'}
          </button>
        </div>
      </div>
    </div>
  );
};

/**
 * Modal centralisée pour assigner un chauffeur
 * Remplace AssignmentModal
 */
const AssignDriverModal = ({ isOpen, onClose, reservation, drivers = [], onAssign }) => {
  if (!isOpen) return null;

  const handleAssign = (driverId) => {
    if (onAssign && reservation) {
      const reservationId = reservation.id || reservation;
      onAssign(reservationId, driverId);
      onClose();
    }
  };

  return (
    <Modal onClose={onClose} size="compact">
      <div className={styles.modalWrapper} data-tour-id="assign-driver-modal">
        <h3>Assigner un chauffeur</h3>

        {reservation && (
          <div className={styles.reservationInfo}>
            <div className={styles.infoRow}>
              <span className={styles.label}>Réservation :</span>
              <strong>#{reservation.id}</strong>
            </div>
            <div className={styles.infoRow}>
              <span className={styles.label}>Client :</span>
              <span>{reservation.customer_name || reservation.client?.full_name}</span>
            </div>
          </div>
        )}

        <div className={styles.driverList}>
          {drivers.length === 0 ? (
            <p className={styles.noDrivers}>Aucun chauffeur disponible</p>
          ) : (
            drivers.map((driver, index) => (
              <div key={driver.id} className={styles.driverItem}>
                <div className={styles.driverInfo} data-tour-id="assign-driver-option">
                  <span className={styles.driverName}>{resolveDriverDisplayName(driver, index)}</span>
                </div>
                <Button
                  data-tour-id="assign-driver-confirm"
                  variant="primary"
                  size="sm"
                  onClick={() => handleAssign(driver.id)}
                >
                  Assigner
                </Button>
              </div>
            ))
          )}
        </div>

        <div className={styles.buttonGroup}>
          <Button variant="secondary" onClick={onClose} fullWidth>
            Annuler
          </Button>
        </div>
      </div>
    </Modal>
  );
};

/**
 * Composant principal qui gère toutes les modales de réservation
 */
const ReservationModals = ({
  // Schedule Return Time Modal
  scheduleModalOpen,
  scheduleModalReservation,
  onScheduleConfirm,
  onScheduleClose,

  // Assign Driver Modal
  assignModalOpen,
  assignModalReservation,
  assignModalDrivers,
  onAssignConfirm,
  onAssignClose,

  // Edit Reservation Modal
  editModalOpen,
  editModalReservation,
  onEditConfirm,
  onEditClose,

  // Delete Confirmation Modal
  deleteModalOpen,
  deleteModalReservation,
  onDeleteConfirm,
  onDeleteClose,
}) => {
  return (
    <>
      {/* Modal Planifier l'heure */}
      <ScheduleReturnTimeModal
        isOpen={scheduleModalOpen}
        reservation={scheduleModalReservation}
        onConfirm={onScheduleConfirm}
        onClose={onScheduleClose}
      />

      {/* Modal Assigner un chauffeur */}
      <AssignDriverModal
        isOpen={assignModalOpen}
        reservation={assignModalReservation}
        drivers={assignModalDrivers}
        onAssign={onAssignConfirm}
        onClose={onAssignClose}
      />

      {/* Modal Éditer la réservation */}
      <EditReservationModal
        isOpen={editModalOpen}
        reservation={editModalReservation}
        onConfirm={onEditConfirm}
        onClose={onEditClose}
      />

      {/* Modal Annulation / Suppression intelligent */}
      <CancellationModal
        isOpen={deleteModalOpen}
        reservation={deleteModalReservation}
        onConfirm={onDeleteConfirm}
        onClose={onDeleteClose}
      />
    </>
  );
};

export default ReservationModals;
export { ScheduleReturnTimeModal, AssignDriverModal, EditReservationModal };
