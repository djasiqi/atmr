import React from 'react';
import Modal from './Modal';
import styles from './ConfirmationModal.module.css';

const ConfirmationModal = ({
  isOpen,
  onClose,
  onConfirm,
  title,
  children, // On remplace "message" par "children" pour plus de flexibilité
  confirmText = "Confirmer",
  cancelText = "Annuler"
}) => {
  if (!isOpen) return null;

  return (
    <Modal onClose={onClose}>
      <div className={styles.container}>
        <h3 className={styles.title}>{title}</h3>
        {/* On affiche ici le contenu passé au composant */}
        <div className={styles.messageContent}>
          {children}
        </div>
        <div className={styles.actions}>
          <button onClick={onClose} className={`${styles.button} ${styles.cancelButton}`}>
            {cancelText}
          </button>
          <button onClick={onConfirm} className={`${styles.button} ${styles.confirmButton}`}>
            {confirmText}
          </button>
        </div>
      </div>
    </Modal>
  );
};

export default ConfirmationModal;