import React from 'react';
import { FiAlertTriangle, FiX } from 'react-icons/fi';
import Modal from './Modal';
import styles from './ConfirmationModal.module.css';

const ConfirmationModal = ({
  isOpen,
  onClose,
  onConfirm,
  title,
  message,
  children,
  confirmText = 'Confirmer',
  cancelText = 'Annuler',
  confirmButtonVariant = 'primary',
}) => {
  if (!isOpen) return null;

  const isDanger = confirmButtonVariant === 'danger' ||
    title?.toLowerCase().includes('supprimer') ||
    title?.toLowerCase().includes('annuler');

  const confirmClass = isDanger ? styles.dangerBtn : styles.primaryBtn;

  return (
    <Modal onClose={onClose} size="sm">
      <div className={styles.modal}>
        <div className={styles.header}>
          <div className={styles.headerLeft}>
            <div className={isDanger ? styles.iconDanger : styles.iconPrimary}>
              <FiAlertTriangle size={16} />
            </div>
            <h3 className={styles.title}>{title}</h3>
          </div>
          <button type="button" className={styles.closeBtn} onClick={onClose} aria-label="Fermer">
            <FiX size={16} />
          </button>
        </div>

        <div className={styles.body}>
          {message && <p className={styles.message}>{message}</p>}
          {children}
        </div>

        <div className={styles.footer}>
          <button type="button" onClick={onClose} className={styles.cancelBtn}>
            {cancelText}
          </button>
          <button type="button" onClick={onConfirm} className={confirmClass}>
            {confirmText}
          </button>
        </div>
      </div>
    </Modal>
  );
};

export default ConfirmationModal;
