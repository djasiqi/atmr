import React from 'react';
import { FaPaperPlane } from 'react-icons/fa';
import styles from './ConfirmSendModal.module.css';

const ConfirmSendModal = ({ onClose, onConfirm, loading = false }) => (
  <div className={styles.overlay} onClick={onClose}>
    <div className={styles.modal} data-tour-id="institution-request-send-confirm-modal" onClick={(e) => e.stopPropagation()}>
      <div className={styles.header}>
        <div className={styles.iconWrapper}>
          <FaPaperPlane size={16} />
        </div>
        <h3 className={styles.title}>Confirmer l'envoi</h3>
      </div>

      <div className={styles.body}>
        <p className={styles.message}>
          Cette demande sera transmise aux <strong>transporteurs disponibles</strong>.
          Ils pourront alors accepter ou décliner la course.
        </p>
        <p className={styles.hint}>
          Vous serez notifié dès qu'un transporteur accepte la demande.
        </p>
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
          data-tour-id="institution-request-send-confirm-btn"
          onClick={onConfirm}
          disabled={loading}
        >
          {loading ? 'Envoi…' : (
            <>
              <FaPaperPlane size={12} />
              Envoyer
            </>
          )}
        </button>
      </div>
    </div>
  </div>
);

export default ConfirmSendModal;
