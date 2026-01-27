// frontend/src/pages/company/BillingReview/components/LockUnlockModal.jsx
import React, { useState } from 'react';
import styles from './LockUnlockModal.module.css';

const LockUnlockModal = ({ booking, mode, onClose, onConfirm }) => {
  const [reason, setReason] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const isLock = mode === 'lock';
  const title = isLock ? 'Verrouiller le booking' : 'Déverrouiller le booking';
  const actionLabel = isLock ? 'Verrouiller' : 'Déverrouiller';

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!reason.trim()) {
      setError('Le motif est obligatoire');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      await onConfirm(booking.booking_id, reason);
    } catch (err) {
      setError(err.response?.data?.error || `Erreur lors du ${isLock ? 'verrouillage' : 'déverrouillage'}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={styles.modalOverlay} onClick={onClose}>
      <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
        <div className={styles.modalHeader}>
          <h2>{title}</h2>
          <button className={styles.closeBtn} onClick={onClose}>
            ×
          </button>
        </div>

        <form onSubmit={handleSubmit} className={styles.form}>
          <div className={styles.info}>
            <p>
              <strong>Booking ID:</strong> {booking.booking_id}
            </p>
            <p>
              <strong>Patient:</strong> {booking.patient_name}
            </p>
            <p>
              <strong>Montant:</strong> {booking.amount.toFixed(2)} CHF
            </p>
          </div>

          <div className={styles.formGroup}>
            <label htmlFor="reason">Motif {isLock ? 'du verrouillage' : 'du déverrouillage'} *</label>
            <textarea
              id="reason"
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              required
              rows={4}
              placeholder={`Expliquez pourquoi vous ${isLock ? 'verrouillez' : 'déverrouillez'} ce booking...`}
            />
          </div>

          {error && <div className={styles.error}>{error}</div>}

          <div className={styles.modalActions}>
            <button type="button" onClick={onClose} className={styles.btnCancel}>
              Annuler
            </button>
            <button type="submit" disabled={loading} className={styles.btnConfirm}>
              {loading ? 'Traitement...' : actionLabel}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default LockUnlockModal;
