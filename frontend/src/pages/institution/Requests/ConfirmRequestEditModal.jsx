import React from 'react';
import { FaEdit } from 'react-icons/fa';
import styles from './ConfirmSendModal.module.css';

const STATUS_COPY = {
  SENT: {
    title: 'Confirmer la modification',
    message: (
      <>
        Cette demande a déjà été <strong>envoyée aux transporteurs</strong>.
        Votre modification (trajet, horaires, besoins…) sera visible par les
        entreprises consultées.
      </>
    ),
    hint: 'Les transporteurs en attente devront tenir compte du parcours mis à jour avant d\u2019accepter la course.',
    confirmLabel: 'Confirmer et enregistrer',
  },
  ACCEPTED: {
    title: 'Confirmer la modification',
    message: (
      <>
        Un transporteur a déjà <strong>accepté cette demande</strong>.
        Votre modification sera transmise pour validation avant application définitive.
      </>
    ),
    hint: 'En cas de changement majeur, le transporteur pourra être sollicité à nouveau selon les règles en vigueur.',
    confirmLabel: 'Confirmer et enregistrer',
  },
};

const ConfirmRequestEditModal = ({
  requestStatus = 'SENT',
  onClose,
  onConfirm,
  loading = false,
}) => {
  const copy = STATUS_COPY[requestStatus] || STATUS_COPY.SENT;

  return (
    <div className={styles.overlay} onClick={onClose}>
      <div
        className={styles.modal}
        data-tour-id="institution-request-edit-confirm-modal"
        onClick={(e) => e.stopPropagation()}
      >
        <div className={styles.header}>
          <div className={styles.iconWrapper}>
            <FaEdit size={16} />
          </div>
          <h3 className={styles.title}>{copy.title}</h3>
        </div>

        <div className={styles.body}>
          <p className={styles.message}>{copy.message}</p>
          <p className={styles.hint}>{copy.hint}</p>
        </div>

        <div className={styles.actions}>
          <button
            type="button"
            className={styles.cancelBtn}
            onClick={onClose}
            disabled={loading}
          >
            Annuler
          </button>
          <button
            type="button"
            className={styles.confirmBtn}
            data-tour-id="institution-request-edit-confirm-btn"
            onClick={onConfirm}
            disabled={loading}
          >
            {loading ? 'Enregistrement…' : copy.confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
};

export default ConfirmRequestEditModal;
