// src/components/common/Modal.jsx
import React from 'react';

const Modal = ({ children, onClose, size = 'lg', className = '' }) => {
  const handleClickOutside = (e) => {
    // Ferme le modal si on clique sur l'overlay (pas sur le contenu)
    if (e.target.classList.contains('modal-overlay')) {
      onClose();
    }
  };

  const overlayClasses = ['modal-overlay', `modal-${size}`, className].filter(Boolean).join(' ');

  return (
    <div className={overlayClasses} onClick={handleClickOutside}>
      <div className="modal-content">{children}</div>
    </div>
  );
};

export default Modal;
