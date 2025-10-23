import React from 'react';
import Modal from './Modal';

const ConfirmationModal = ({
  isOpen,
  onClose,
  onConfirm,
  title,
  children, // On remplace "message" par "children" pour plus de flexibilité
  confirmText = 'Confirmer',
  cancelText = 'Annuler',
}) => {
  if (!isOpen) return null;

  return (
    <Modal onClose={onClose}>
      <div className="modal-header">
        <h3 className="modal-title">{title}</h3>
      </div>
      <div className="modal-body">{children}</div>
      <div className="modal-footer">
        <button onClick={onClose} className="btn btn-secondary">
          {cancelText}
        </button>
        <button onClick={onConfirm} className="btn btn-primary">
          {confirmText}
        </button>
      </div>
    </Modal>
  );
};

export default ConfirmationModal;
