// src/components/common/Modal.jsx
import React from 'react';

const Modal = ({ children, onClose, size = 'lg' }) => {
  const handleClickOutside = (e) => {
    // Ferme le modal si on clique sur l'overlay (pas sur le contenu)
    if (e.target.classList.contains('modal-overlay')) {
      onClose();
    }
  };

  return (
    <div className={`modal-overlay modal-${size}`} onClick={handleClickOutside}>
      <div className="modal-content">{children}</div>
    </div>
  );
};

export default Modal;
