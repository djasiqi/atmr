// src/components/reservations/ReservationModals.jsx
import React, { useState, useEffect } from 'react';
import Modal from '../common/Modal';
import Button from '../common/Button';
import ConfirmationModal from '../common/ConfirmationModal';
import EditReservationModal from './EditReservationModal';
import styles from './ReservationModals.module.css';

/**
 * Modal centralisée pour planifier l'heure de retour
 * Remplace ReturnTimeModal et ScheduleReturnModal
 */
const ScheduleReturnTimeModal = ({ isOpen, onClose, reservation, onConfirm }) => {
  const [selectedDate, setSelectedDate] = useState('');
  const [selectedTime, setSelectedTime] = useState('');
  const [loading, setLoading] = useState(false);

  // Initialiser avec la date/heure fournie ou maintenant + 1h par défaut
  useEffect(() => {
    if (isOpen) {
      let dateObj;

      if (reservation?.scheduled_time) {
        dateObj = new Date(reservation.scheduled_time);
      } else {
        // Pas de scheduled_time : utiliser la date du jour et suggérer maintenant + 1h
        dateObj = new Date(Date.now() + 60 * 60 * 1000);
      }

      // Format date YYYY-MM-DD
      const year = dateObj.getFullYear();
      const month = String(dateObj.getMonth() + 1).padStart(2, '0');
      const day = String(dateObj.getDate()).padStart(2, '0');
      setSelectedDate(`${year}-${month}-${day}`);

      // Format heure HH:mm (24h)
      const hours = String(dateObj.getHours()).padStart(2, '0');
      const minutes = String(dateObj.getMinutes()).padStart(2, '0');
      setSelectedTime(`${hours}:${minutes}`);
    }
  }, [isOpen, reservation]);

  // Date minimale = aujourd'hui
  const minDate = new Date().toISOString().split('T')[0];
  // Heure minimale = maintenant si date = aujourd'hui, sinon 00:00
  const today = new Date().toISOString().split('T')[0];
  const minTime =
    selectedDate === today
      ? `${String(new Date().getHours()).padStart(2, '0')}:${String(
          new Date().getMinutes()
        ).padStart(2, '0')}`
      : '00:00';

  const handleConfirm = async () => {
    if (!selectedDate || !selectedTime) {
      return;
    }

    setLoading(true);
    try {
      // Format attendu selon le contexte (string ou objet)
      if (typeof onConfirm === 'function') {
        const [hours, minutes] = selectedTime.split(':');
        const dateTime = new Date(`${selectedDate}T${hours}:${minutes}`);

        // Essayer d'appeler avec l'objet format (comme ReturnTimeModal)
        try {
          await onConfirm({ return_time: dateTime.toISOString().slice(0, 16) });
        } catch (e) {
          // Si ça échoue, essayer avec le format string (comme ScheduleReturnModal)
          const isoDatetime = `${selectedDate} ${selectedTime}`;
          await onConfirm(isoDatetime);
        }
      }
      onClose();
    } catch (error) {
      console.error('Erreur lors de la planification:', error);
      alert(error?.response?.data?.error || 'Erreur lors de la planification');
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

  return (
    <Modal onClose={onClose} size="compact">
      <div className={styles.modalWrapper}>
        <h3>Planifier l'heure de retour</h3>

        {reservation && (
          <div className={styles.reservationInfo}>
            <div className={styles.infoRow}>
              <span className={styles.label}>Client :</span>
              <strong>{reservation.customer_name || reservation.client?.full_name}</strong>
            </div>
            <div className={styles.infoRow}>
              <span className={styles.label}>Aller :</span>
              <span>
                {reservation.original_booking?.scheduled_time
                  ? new Date(reservation.original_booking.scheduled_time).toLocaleString('fr-CH', {
                      day: '2-digit',
                      month: '2-digit',
                      year: 'numeric',
                      hour: '2-digit',
                      minute: '2-digit',
                    })
                  : reservation.scheduled_time
                  ? new Date(reservation.scheduled_time).toLocaleString('fr-CH', {
                      day: '2-digit',
                      month: '2-digit',
                      year: 'numeric',
                      hour: '2-digit',
                      minute: '2-digit',
                    })
                  : 'Non spécifié'}
              </span>
            </div>
            <div className={styles.infoRow}>
              <span className={styles.label}>Trajet retour :</span>
              <span style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                <span>{reservation.pickup_location}</span>
                <span>→ {reservation.dropoff_location}</span>
              </span>
            </div>
          </div>
        )}

        <div className={styles.formGroup}>
          <label htmlFor="return-date" className={styles.label}>
            Date du retour <span>*</span>
          </label>
          <input
            type="date"
            id="return-date"
            className={styles.input}
            value={selectedDate}
            onChange={(e) => setSelectedDate(e.target.value)}
            min={minDate}
            required
            disabled={loading}
          />
        </div>

        <div className={styles.formGroup}>
          <label htmlFor="return-time" className={styles.label}>
            Heure du retour <span>*</span>
          </label>
          <input
            type="time"
            id="return-time"
            className={styles.input}
            value={selectedTime}
            onChange={(e) => setSelectedTime(e.target.value)}
            min={selectedDate === today ? minTime : undefined}
            required
            disabled={loading}
          />
          <small className={styles.hint}>Format 24h (ex: 14:30)</small>
        </div>

        <div className={styles.buttonGroup}>
          <Button variant="secondary" onClick={onClose} disabled={loading}>
            Annuler
          </Button>
          <Button variant="warning" onClick={handleUrgent} disabled={loading}>
            ⚡ Urgent
          </Button>
          <Button
            variant="primary"
            onClick={handleConfirm}
            loading={loading}
            disabled={!selectedDate || !selectedTime}
          >
            Planifier
          </Button>
        </div>
      </div>
    </Modal>
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
      <div className={styles.modalWrapper}>
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
            drivers.map((driver) => (
              <div key={driver.id} className={styles.driverItem}>
                <div className={styles.driverInfo}>
                  <span className={styles.driverName}>{driver.username || driver.full_name}</span>
                </div>
                <Button variant="primary" size="sm" onClick={() => handleAssign(driver.id)}>
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

      {/* Modal Confirmation suppression */}
      <ConfirmationModal
        isOpen={deleteModalOpen}
        onClose={onDeleteClose}
        onConfirm={onDeleteConfirm}
        title={
          deleteModalReservation
            ? `Supprimer la réservation #${deleteModalReservation.id}`
            : 'Supprimer la réservation'
        }
        confirmText="Oui, supprimer"
        confirmButtonVariant="danger"
      >
        {deleteModalReservation && (
          <p>
            Êtes-vous sûr de vouloir supprimer la réservation pour{' '}
            <strong>
              {deleteModalReservation.customer_name || deleteModalReservation.client?.full_name}
            </strong>{' '}
            ?
          </p>
        )}
      </ConfirmationModal>
    </>
  );
};

export default ReservationModals;
export { ScheduleReturnTimeModal, AssignDriverModal, EditReservationModal };
