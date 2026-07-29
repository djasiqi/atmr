import React, { useState } from 'react';
import { FaBan } from 'react-icons/fa';
import styles from './ConfirmCancelModal.module.css';

const MIN_REASON_LEN = 10;

const ConfirmCancelModal = ({
  onClose,
  onConfirm,
  loading = false,
  requireReason = true,
  minLength = MIN_REASON_LEN,
}) => {
  const [reason, setReason] = useState('');
  const trimmed = reason.trim();
  const tooShort = requireReason && trimmed.length < minLength;

  const handleConfirm = () => {
    if (tooShort || loading) return;
    onConfirm(trimmed);
  };

  return (
    <div className={styles.overlay} onClick={onClose}>
      <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
        <div className={styles.header}>
          <div className={styles.iconWrapper}>
            <FaBan size={16} />
          </div>
          <h3 className={styles.title}>Demander l&apos;annulation</h3>
        </div>

        <div className={styles.body}>
          <p className={styles.message}>
            Une seule demande sera transmise au transporteur. La course reste
            active jusqu&apos;à sa confirmation. Si une demande est déjà en
            cours, elle ne sera pas renvoyée.
          </p>
          <label className={styles.label} htmlFor="cancel-reason">
            Motif d&apos;annulation
            {requireReason && <span className={styles.required}> *</span>}
          </label>
          <textarea
            id="cancel-reason"
            className={styles.textarea}
            value={reason}
            onChange={(e) => setReason(e.target.value)}
            placeholder={`Motif obligatoire (min. ${minLength} caractères si en route)`}
            rows={3}
            autoFocus
            disabled={loading}
          />
          {tooShort && trimmed.length > 0 && (
            <p className={styles.error}>
              Le motif doit contenir au moins {minLength} caractères.
            </p>
          )}
        </div>

        <div className={styles.actions}>
          <button
            type="button"
            className={styles.cancelBtn}
            onClick={onClose}
            disabled={loading}
          >
            Retour
          </button>
          <button
            type="button"
            className={styles.confirmBtn}
            onClick={handleConfirm}
            disabled={loading || tooShort}
          >
            {loading ? 'Envoi…' : (
              <>
                <FaBan size={12} />
                Demander l&apos;annulation
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

export default ConfirmCancelModal;
