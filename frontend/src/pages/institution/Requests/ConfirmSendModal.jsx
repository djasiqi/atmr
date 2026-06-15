import React from 'react';
import { FaPaperPlane, FaRedo } from 'react-icons/fa';
import styles from './ConfirmSendModal.module.css';

const COPY = {
  send: {
    title: 'Confirmer l\'envoi',
    message: (
      <>
        Cette demande sera transmise aux <strong>transporteurs disponibles</strong>.
        Ils pourront alors accepter ou décliner la course.
      </>
    ),
    hint: 'Vous serez notifié dès qu\'un transporteur accepte la demande.',
    confirmLabel: 'Envoyer',
    Icon: FaPaperPlane,
  },
  relaunch: {
    title: 'Relancer la diffusion',
    message: (
      <>
        La demande sera <strong>renvoyée aux transporteurs</strong> car aucune offre
        n&apos;est actuellement en attente.
      </>
    ),
    hint: 'Les transporteurs pourront à nouveau accepter ou proposer un horaire.',
    confirmLabel: 'Relancer',
    Icon: FaRedo,
  },
};

const ConfirmSendModal = ({ onClose, onConfirm, loading = false, mode = 'send' }) => {
  const copy = COPY[mode] || COPY.send;
  const Icon = copy.Icon;

  return (
    <div className={styles.overlay} onClick={onClose}>
      <div
        className={styles.modal}
        data-tour-id={mode === 'relaunch' ? 'institution-request-relaunch-confirm-modal' : 'institution-request-send-confirm-modal'}
        onClick={(e) => e.stopPropagation()}
      >
        <div className={styles.header}>
          <div className={styles.iconWrapper}>
            <Icon size={16} />
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
            data-tour-id={mode === 'relaunch' ? 'institution-request-relaunch-confirm-btn' : 'institution-request-send-confirm-btn'}
            onClick={onConfirm}
            disabled={loading}
          >
            {loading ? 'Envoi…' : (
              <>
                <Icon size={12} />
                {copy.confirmLabel}
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

export default ConfirmSendModal;
